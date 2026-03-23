"""
Export GPT-SoVITS models to GGUF format for C++ inference.

This script exports all components of the GPT-SoVITS TTS system:
- CNHubert: Chinese HuBERT for extracting semantic features from audio
- SSL Proj & Quantizer: Projects CNHubert features and codebook
- Reference Encoder: Encodes reference audio
- Text Encoder: Encodes phonemes with BERT features
- Flow: Residual flow blocks
- Generator: HiFi-GAN vocoder
- VITS AR Model: Autoregressive transformer

Usage:
    python export_gptsovits.py [--all | --cnhubert | --ssl | --ref | --codebook | --text | --flow | --generator | --vits]

Example:
    python export_gptsovits.py --all
"""

import sys
import os
from pathlib import Path

# Setup paths
sys.path.append("src")
from voice_dialogue.config.paths import load_third_party
load_third_party()

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(
        1, str(Path("/home/wjr/mount/code/openvla.cpp/gguf-py").parent / 'gguf-py'))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import gguf
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from moyoyo_tts import TTSModule, TTS_Config
from voice_dialogue.config.speaker_config import get_tts_config_by_speaker_name


# ============================================================================
# Shared utilities
# ============================================================================

def add_config(gguf_writer: gguf.GGUFWriter, model_cfg: dict):
    """Add configuration parameters to GGUF file."""
    for k, v in model_cfg.items():
        if isinstance(v, bool):
            gguf_writer.add_bool(k, v)
        elif isinstance(v, float):
            gguf_writer.add_float32(k, v)
        elif isinstance(v, int):
            gguf_writer.add_uint32(k, v)
        elif isinstance(v, str):
            gguf_writer.add_string(k, v)
        elif isinstance(v, list):
            if len(v) > 0:
                gguf_writer.add_array(k, v)
        else:
            print(f"Warning: Skipping config {k} with unsupported type {type(v)}")


def add_params(gguf_writer: gguf.GGUFWriter, state_dict: dict, prefix: str = ""):
    """Add model parameters to GGUF file."""
    for tensor_name, param in state_dict.items():
        full_name = f"{prefix}{tensor_name}" if prefix else tensor_name

        # Convert bias and norm layers to float32
        if param.dim() <= 1:
            param = param.to(torch.float32)
        elif tensor_name.endswith(("_norm.weight", "layer_norm.weight", "LayerNorm.weight")):
            param = param.to(torch.float32)

        # Check if it's a conv weight
        is_conv_weight = (
            tensor_name.endswith(".weight") and
            param.dim() >= 3
        )

        # For attention layers with kernel_size=1, convert to linear
        if is_conv_weight and any(key in tensor_name for key in ("attn_layers", "cross_attention")):
            shape_prod_2 = np.prod(param.shape[:2])
            shape_prod_all = np.prod(param.shape)
            if shape_prod_2 == shape_prod_all:
                param = param.squeeze()

        if not is_conv_weight:
            param = param.squeeze()

        gguf_writer.add_tensor(full_name, param.cpu().numpy())


def merge_weight_norm(state_dict: dict) -> dict:
    """
    Merge weight norm parametrizations into standard weights.

    Weight norm decomposes weight as: w = g * v / ||v||
    where:
      - g (original0): magnitude scalar per output channel
      - v (original1): direction tensor
    """
    merged_dict = {}
    processed_keys = set()

    for key in state_dict.keys():
        if ".parametrizations.weight.original0" in key:
            base_name = key.replace(".parametrizations.weight.original0", "")
            original0_key = key
            original1_key = key.replace(".original0", ".original1")

            if original1_key in state_dict:
                g = state_dict[original0_key]
                v = state_dict[original1_key]
                weight = torch._weight_norm(v, g, 2)

                merged_key = f"{base_name}.weight"
                merged_dict[merged_key] = weight
                processed_keys.add(original0_key)
                processed_keys.add(original1_key)

                print(f"  Merged weight_norm: {original0_key} + {original1_key} -> {merged_key}")

    for key, value in state_dict.items():
        if key not in processed_keys:
            merged_dict[key] = value

    return merged_dict


def create_gguf_writer(model_name: str, output_dir: str = "gguf") -> gguf.GGUFWriter:
    """Create a GGUF writer with standard settings."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{model_name}.gguf"
    gguf_writer = gguf.GGUFWriter(output_path, model_name)
    gguf_writer.add_type(gguf.GGUFType.MODEL)
    gguf_writer.add_name(model_name)
    gguf_writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    gguf_writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    return gguf_writer, output_path


def finalize_gguf(gguf_writer: gguf.GGUFWriter, output_path: str):
    """Write and close GGUF file."""
    print(f"Writing to {output_path}...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file(progress=True)
    gguf_writer.close()
    print(f"Successfully exported to {output_path}")


# ============================================================================
# CNHubert export functions
# ============================================================================

def export_cnhubert(tts_module):
    """
    Export CNHubert model to GGUF format.

    CNHubert architecture (based on HuBERT Base):
    - Feature extractor: 7-layer 1D CNN
    - Feature projection: Linear(512, 768) + LayerNorm
    - Encoder: 12-layer Transformer
    """
    print("\n=== Exporting CNHubert ===")

    model = tts_module.tts_pipeline.cnhuhbert_model.model
    model.eval()
    config = model.config

    gguf_writer, output_path = create_gguf_writer("cnhubert")

    model_cfg = {
        "n_layer": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "n_heads": config.num_attention_heads,
        "intermediate_size": config.intermediate_size,
        "layer_norm_eps": float(config.layer_norm_eps),
        "num_feat_extract_layers": config.num_feat_extract_layers,
        "conv_dim": list(config.conv_dim),
        "conv_kernel": list(config.conv_kernel),
        "conv_stride": list(config.conv_stride),
    }

    print(f"Model config: {model_cfg}")
    add_config(gguf_writer, model_cfg)

    state_dict = model.state_dict()
    print("Merging weight norm parametrizations...")
    state_dict = merge_weight_norm(state_dict)

    print(f"Exporting {len(state_dict)} tensors...")
    add_params(gguf_writer, state_dict)

    finalize_gguf(gguf_writer, output_path)


def export_ssl_proj_quantizer(tts_module):
    """
    Export ssl_proj and quantizer codebook from VITS model.

    ssl_proj: Conv1D(768, 768, kernel=1) - projects CNHubert features
    quantizer: ResidualVectorQuantizer codebook (1024, 768)
    """
    print("\n=== Exporting SSL Proj & Quantizer ===")

    vits_model = tts_module.tts_pipeline.vits_model

    ssl_proj = vits_model.ssl_proj
    quantizer = vits_model.quantizer
    codebook = quantizer.vq.layers[0]._codebook.embed

    gguf_writer, output_path = create_gguf_writer("ssl_proj_quantizer")

    model_cfg = {
        "ssl_dim": 768,
        "codebook_size": 1024,
        "n_q": 1,
    }
    add_config(gguf_writer, model_cfg)

    ssl_proj_weight = ssl_proj.weight
    ssl_proj_bias = ssl_proj.bias if ssl_proj.bias is not None else None

    gguf_writer.add_tensor("ssl_proj.weight", ssl_proj_weight.detach().cpu().numpy())
    if ssl_proj_bias is not None:
        gguf_writer.add_tensor("ssl_proj.bias", ssl_proj_bias.detach().cpu().numpy())

    codebook_squeezed = codebook.squeeze(0)
    gguf_writer.add_tensor("quantizer.codebook", codebook_squeezed.detach().cpu().numpy())

    finalize_gguf(gguf_writer, output_path)


# ============================================================================
# SynthesizerTrn export functions
# ============================================================================

def export_ref_encoder(vits_model):
    """Export reference encoder."""
    print("\n=== Exporting Reference Encoder ===")

    ref_enc = vits_model.ref_enc
    state_dict = ref_enc.state_dict()

    gguf_writer, output_path = create_gguf_writer("ref_enc")
    add_params(gguf_writer, state_dict)

    one = torch.ones(1, dtype=torch.float32)
    gguf_writer.add_tensor("one", one.cpu().numpy())

    finalize_gguf(gguf_writer, output_path)


def export_codebook(vits_model):
    """Export quantizer codebook."""
    print("\n=== Exporting Codebook ===")

    class CodeBook(torch.nn.Module):
        def __init__(self, codebook):
            super().__init__()
            self.embed = torch.nn.Parameter(codebook.embed, requires_grad=False)

    code_book = vits_model.quantizer.vq.layers[0]._codebook
    state_dict = CodeBook(code_book).state_dict()

    gguf_writer, output_path = create_gguf_writer("codebook")
    add_params(gguf_writer, state_dict)

    finalize_gguf(gguf_writer, output_path)


def export_text_encoder(vits_model):
    """Export text encoder."""
    print("\n=== Exporting Text Encoder ===")

    state_dict = vits_model.enc_p.state_dict()
    model_cfgs = {
        "n_layer": len(vits_model.enc_p.encoder_text.attn_layers),
        "n_q_heads": 2,
        "n_kv_heads": 2,
        "norm_eps": 1e-5,
        "n_heads_mrte": vits_model.enc_p.mrte.cross_attention.n_heads
    }

    gguf_writer, output_path = create_gguf_writer("text_encoder")
    add_config(gguf_writer, model_cfgs)
    add_params(gguf_writer, state_dict)

    finalize_gguf(gguf_writer, output_path)


def export_flow(vits_model):
    """Export residual flow blocks."""
    print("\n=== Exporting Flow ===")

    from moyoyo_tts.module.modules import WN
    model = vits_model.flow
    for name, module in model.named_modules():
        if isinstance(module, WN):
            module.remove_weight_norm()

    state_dict = model.state_dict()

    gguf_writer, output_path = create_gguf_writer("flow")
    add_params(gguf_writer, state_dict)

    finalize_gguf(gguf_writer, output_path)


def export_generator(vits_model):
    """Export HiFi-GAN generator."""
    print("\n=== Exporting Generator ===")

    dec = vits_model.dec
    dec.eval()
    dec.remove_weight_norm()

    state_dict = dec.state_dict()

    gguf_writer, output_path = create_gguf_writer("generator")

    model_cfgs = {
        "num_upsamples": len(dec.ups),
        "num_kernels": dec.num_kernels,
        "slope": 0.1
    }
    add_config(gguf_writer, model_cfgs)
    add_params(gguf_writer, state_dict)

    finalize_gguf(gguf_writer, output_path)


# ============================================================================
# VITS AR model export functions
# ============================================================================

class VitsTextEncoder(torch.nn.Module):
    def __init__(self, t2s_model):
        super(VitsTextEncoder, self).__init__()
        self.ar_text_embedding = t2s_model.model.ar_text_embedding
        self.bert_proj = t2s_model.model.bert_proj
        self.ar_text_position_alpha = t2s_model.model.ar_text_position.alpha.detach().item()
        self.ar_text_position_x_scale = t2s_model.model.ar_text_position.x_scale
        self.ar_text_position_pe = torch.nn.Parameter(torch.tensor(
            t2s_model.model.ar_text_position.pe[0]), requires_grad=False)
        del t2s_model.model.bert_proj
        del t2s_model.model.ar_text_embedding
        del t2s_model.model.ar_text_position

    @torch.no_grad()
    def forward(self, x, bert_feature, pos=None):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature)
        x = x * self.ar_text_position_x_scale + \
            self.ar_text_position_alpha * \
            self.ar_text_position_pe[pos] if pos is not None else self.ar_text_position_pe[:x.size(1)]
        return x


def export_vits_text_model(model):
    """Export VITS text encoder model."""
    print("\n=== Exporting VITS Text Model ===")

    gguf_writer, output_path = create_gguf_writer("vits_text")

    gguf_writer.add_embedding_length(model.ar_text_position_pe.shape[1])

    model_cfg = {
        "alpha": model.ar_text_position_alpha,
        "x_scale": model.ar_text_position_x_scale,
    }
    add_config(gguf_writer, model_cfg)
    add_params(gguf_writer, model.state_dict())

    finalize_gguf(gguf_writer, output_path)


def export_vits_ar_model(tts_module):
    """Export VITS autoregressive model."""
    print("\n=== Exporting VITS AR Model ===")

    t2s_model = tts_module.tts_pipeline.t2s_model

    # Extract model info before modification
    num_head = t2s_model.model.num_head
    num_kv_head = num_head
    num_blocks = len(t2s_model.model.h.layers)
    max_ctx = t2s_model.model.ar_audio_position.pe.shape[1]
    vocab_size, emb_dim = t2s_model.model.ar_audio_embedding.word_embeddings.weight.shape
    norm_eps = t2s_model.model.h.layers[0].norm1.eps
    intermediate_size = t2s_model.model.h.layers[0].linear1.weight.shape[0]

    # Create text encoder (this modifies t2s_model)
    vits_text_encoder = VitsTextEncoder(t2s_model)
    export_vits_text_model(vits_text_encoder)

    # Build tensor name mapping
    tensor_name_maps = {
        "ar_audio_embedding.word_embeddings.weight": "token_embd.weight",
        "ar_audio_position_pe": "position_embd.weight",
        "ar_predict_layer.weight": "output.weight",
        "ar_predict_layer.bias": "output.bias",
    }
    blk_params = {
        "h.layers.%d.self_attn.in_proj_weight": "blk.%d.attn_qkv.weight",
        "h.layers.%d.self_attn.in_proj_bias": "blk.%d.attn_qkv.bias",
        "h.layers.%d.self_attn.out_proj.weight": "blk.%d.attn_output.weight",
        "h.layers.%d.self_attn.out_proj.bias": "blk.%d.attn_output.bias",
        "h.layers.%d.linear1.weight": "blk.%d.ffn_up.weight",
        "h.layers.%d.linear1.bias": "blk.%d.ffn_up.bias",
        "h.layers.%d.linear2.weight": "blk.%d.ffn_down.weight",
        "h.layers.%d.linear2.bias": "blk.%d.ffn_down.bias",
        "h.layers.%d.norm1.weight": "blk.%d.attn_norm.weight",
        "h.layers.%d.norm1.bias": "blk.%d.attn_norm.bias",
        "h.layers.%d.norm2.weight": "blk.%d.ffn_norm.weight",
        "h.layers.%d.norm2.bias": "blk.%d.ffn_norm.bias",
    }

    for b_id in range(num_blocks):
        for k, v in blk_params.items():
            old_k = k % b_id
            new_k = v % b_id
            tensor_name_maps[old_k] = new_k

    # Prepare model
    vits_model = t2s_model.model
    vits_model.ar_audio_position_alpha = vits_model.ar_audio_position.alpha.detach().item()
    vits_model.ar_audio_position_x_scale = vits_model.ar_audio_position.x_scale
    vits_model.ar_audio_position_pe = torch.nn.Parameter(torch.tensor(
        vits_model.ar_audio_position.pe[0]), requires_grad=False)
    del vits_model.ar_audio_position

    # Rename state dict keys
    vits_state_dict = vits_model.state_dict()
    new_state_dict = {}
    for k, v in vits_state_dict.items():
        if k in tensor_name_maps:
            new_state_dict[tensor_name_maps[k]] = v
        else:
            new_state_dict[k] = v

    # Create GGUF
    print("\n=== Exporting VITS Main Model ===")
    gguf_writer, output_path = create_gguf_writer("vits")

    model_cfgs = {
        "vits.vocab_size": vocab_size,
        "alpha": vits_model.ar_audio_position_alpha,
        "x_scale": vits_model.ar_audio_position_x_scale,
    }
    add_config(gguf_writer, model_cfgs)

    gguf_writer.add_embedding_length(emb_dim)
    gguf_writer.add_block_count(num_blocks)
    gguf_writer.add_context_length(max_ctx)
    gguf_writer.add_feed_forward_length(intermediate_size)
    gguf_writer.add_head_count(num_head)
    gguf_writer.add_head_count_kv(num_kv_head)
    gguf_writer.add_layer_norm_eps(norm_eps)

    add_params(gguf_writer, new_state_dict)

    finalize_gguf(gguf_writer, output_path)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Export GPT-SoVITS models to GGUF format")
    parser.add_argument("--all", action="store_true", help="Export all models")
    parser.add_argument("--cnhubert", action="store_true", help="Export CNHubert model")
    parser.add_argument("--ssl", action="store_true", help="Export SSL proj and quantizer")
    parser.add_argument("--ref", action="store_true", help="Export reference encoder")
    parser.add_argument("--codebook", action="store_true", help="Export codebook")
    parser.add_argument("--text", action="store_true", help="Export text encoder")
    parser.add_argument("--flow", action="store_true", help="Export flow model")
    parser.add_argument("--generator", action="store_true", help="Export generator")
    parser.add_argument("--vits", action="store_true", help="Export VITS AR model")
    parser.add_argument("--speaker", type=str, default="Ellen", help="Speaker name for TTS config")

    args = parser.parse_args()

    # Default to --all if no specific model is selected
    if not any([args.cnhubert, args.ssl, args.ref, args.codebook, args.text,
                args.flow, args.generator, args.vits]):
        args.all = True

    # Initialize TTS module
    print(f"Loading TTS module with speaker: {args.speaker}")
    tts_speaker_config = get_tts_config_by_speaker_name(args.speaker)
    tts_config = TTS_Config(tts_speaker_config.get_runtime_config())
    tts_module = TTSModule(tts_config)
    tts_module.setup_inference_params(
        ref_audio=tts_speaker_config.reference_audio_path,
        parallel_infer=False,
        **tts_speaker_config.inference_parameters.model_dump()
    )

    vits_model = tts_module.tts_pipeline.vits_model

    # Export models based on arguments
    if args.all or args.cnhubert:
        export_cnhubert(tts_module)

    if args.all or args.ssl:
        export_ssl_proj_quantizer(tts_module)

    if args.all or args.ref:
        export_ref_encoder(vits_model)

    if args.all or args.codebook:
        export_codebook(vits_model)

    if args.all or args.text:
        export_text_encoder(vits_model)

    if args.all or args.flow:
        export_flow(vits_model)

    if args.all or args.generator:
        export_generator(vits_model)

    if args.all or args.vits:
        export_vits_ar_model(tts_module)

    print("\n=== Export Complete ===")


if __name__ == "__main__":
    main()
