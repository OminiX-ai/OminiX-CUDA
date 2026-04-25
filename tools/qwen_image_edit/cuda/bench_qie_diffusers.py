#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit_plus as qie_edit_plus
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    CONDITION_IMAGE_SIZE,
    VAE_IMAGE_SIZE,
    XLA_AVAILABLE,
    calculate_dimensions,
    calculate_shift,
    retrieve_timesteps,
)


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def patch_cuda_integer_prod():
    # PyTorch 2.13.0.dev20260424+cu130 on GB10 fails integer Tensor.prod(dim)
    # with "CUDA driver error: invalid argument". Qwen2.5-VL uses this on the
    # tiny image_grid_thw tensor during prompt/image encoding.
    original_prod = torch.Tensor.prod

    def safe_prod(self, *args, **kwargs):
        if self.is_cuda and not (self.dtype.is_floating_point or self.dtype.is_complex):
            return original_prod(self.cpu(), *args, **kwargs).to(self.device)
        return original_prod(self, *args, **kwargs)

    torch.Tensor.prod = safe_prod


def bytes_to_gib(value):
    return value / (1024**3)


def summarize(values):
    if not values:
        return {}
    return {
        "count": len(values),
        "mean_s": statistics.fmean(values),
        "median_s": statistics.median(values),
        "min_s": min(values),
        "max_s": max(values),
        "total_s": sum(values),
    }


def clone_tensors(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(clone_tensors(item) for item in value)
    if isinstance(value, list):
        return [clone_tensors(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_tensors(item) for key, item in value.items()}
    return value


def parse_module_list(value):
    if value is None:
        return None
    modules = [item.strip() for item in value.split(",") if item.strip()]
    return modules or None


def build_torchao_fp8_config(backend, modules_to_not_convert):
    from diffusers import TorchAoConfig
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, Float8WeightOnlyConfig

    if backend == "torchao-float8wo":
        quant_type = Float8WeightOnlyConfig(weight_dtype=torch.float8_e4m3fn)
    elif backend == "torchao-float8dq":
        quant_type = Float8DynamicActivationFloat8WeightConfig(
            activation_dtype=torch.float8_e4m3fn,
            weight_dtype=torch.float8_e4m3fn,
        )
    else:
        raise ValueError(f"Unsupported TorchAO FP8 backend: {backend}")

    return TorchAoConfig(quant_type, modules_to_not_convert=modules_to_not_convert)


def combine_cfg_conditioning(
    negative_embeds,
    prompt_embeds,
    negative_mask=None,
    prompt_mask=None,
    force_no_padding_mask=False,
):
    """Stack uncond + cond embeddings, padding sequence length only when needed."""
    neg_seq = negative_embeds.shape[1]
    prompt_seq = prompt_embeds.shape[1]
    max_seq = max(neg_seq, prompt_seq)

    def pad_embeds(embeds):
        seq_pad = max_seq - embeds.shape[1]
        if seq_pad == 0:
            return embeds
        return F.pad(embeds, (0, 0, 0, seq_pad))

    def pad_mask(mask, batch_size, seq_len, device):
        if mask is None:
            mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        else:
            mask = mask.to(device=device, dtype=torch.bool)
        seq_pad = max_seq - seq_len
        if seq_pad == 0:
            return mask
        return F.pad(mask, (0, seq_pad), value=False)

    batched_embeds = torch.cat([pad_embeds(negative_embeds), pad_embeds(prompt_embeds)], dim=0)
    if force_no_padding_mask:
        return batched_embeds, None

    if negative_mask is None and prompt_mask is None and neg_seq == prompt_seq:
        return batched_embeds, None

    batched_mask = torch.cat(
        [
            pad_mask(negative_mask, negative_embeds.shape[0], neg_seq, negative_embeds.device),
            pad_mask(prompt_mask, prompt_embeds.shape[0], prompt_seq, prompt_embeds.device),
        ],
        dim=0,
    )
    return batched_embeds, batched_mask


@torch.no_grad()
def qwen_edit_plus_call_batched_cfg(
    self,
    image=None,
    prompt=None,
    negative_prompt=None,
    true_cfg_scale=4.0,
    height=None,
    width=None,
    num_inference_steps=50,
    sigmas=None,
    guidance_scale=None,
    num_images_per_prompt=1,
    generator=None,
    latents=None,
    prompt_embeds=None,
    prompt_embeds_mask=None,
    negative_prompt_embeds=None,
    negative_prompt_embeds_mask=None,
    output_type="pil",
    return_dict=True,
    attention_kwargs=None,
    callback_on_step_end=None,
    callback_on_step_end_tensor_inputs=["latents"],
    max_sequence_length=512,
    cfg_force_no_padding_mask=False,
):
    image_size = image[-1].size if isinstance(image, list) else image.size
    calculated_width, calculated_height = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
    height = height or calculated_height
    width = width or calculated_width

    multiple_of = self.vae_scale_factor * 2
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of

    self.check_inputs(
        prompt,
        height,
        width,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if batch_size > 1:
        raise ValueError(
            f"QwenImageEditPlusPipeline currently only supports batch_size=1, but received batch_size={batch_size}. "
            "Please process prompts one at a time."
        )

    device = self._execution_device
    condition_images = None
    vae_images = None
    vae_image_sizes = []

    if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
        if not isinstance(image, list):
            image = [image]
        condition_images = []
        vae_images = []
        for img in image:
            image_width, image_height = img.size
            condition_width, condition_height = calculate_dimensions(CONDITION_IMAGE_SIZE, image_width / image_height)
            vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
            vae_image_sizes.append((vae_width, vae_height))
            condition_images.append(self.image_processor.resize(img, condition_height, condition_width))
            vae_images.append(self.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))

    has_neg_prompt = negative_prompt is not None or negative_prompt_embeds is not None
    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    if not do_true_cfg:
        raise ValueError("--cfg-batching requires true CFG: pass true_cfg_scale > 1 and a negative prompt.")

    prompt_embeds, prompt_embeds_mask = self.encode_prompt(
        image=condition_images,
        prompt=prompt,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )
    negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
        image=condition_images,
        prompt=negative_prompt,
        prompt_embeds=negative_prompt_embeds,
        prompt_embeds_mask=negative_prompt_embeds_mask,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )

    cfg_encoder_hidden_states, cfg_encoder_hidden_states_mask = combine_cfg_conditioning(
        negative_prompt_embeds,
        prompt_embeds,
        negative_prompt_embeds_mask,
        prompt_embeds_mask,
        force_no_padding_mask=cfg_force_no_padding_mask,
    )
    print(
        "CFG_BATCH_CONDITIONING "
        f"negative_shape={tuple(negative_prompt_embeds.shape)} "
        f"prompt_shape={tuple(prompt_embeds.shape)} "
        f"combined_shape={tuple(cfg_encoder_hidden_states.shape)} "
        f"mask={'none' if cfg_encoder_hidden_states_mask is None else tuple(cfg_encoder_hidden_states_mask.shape)} "
        f"force_no_padding_mask={cfg_force_no_padding_mask}",
        flush=True,
    )

    num_channels_latents = self.transformer.config.in_channels // 4
    latents, image_latents = self.prepare_latents(
        vae_images,
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    img_shapes = [
        [
            (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
            *[
                (1, vae_height // self.vae_scale_factor // 2, vae_width // self.vae_scale_factor // 2)
                for vae_width, vae_height in vae_image_sizes
            ],
        ]
    ] * batch_size

    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.get("base_image_seq_len", 256),
        self.scheduler.config.get("max_image_seq_len", 4096),
        self.scheduler.config.get("base_shift", 0.5),
        self.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    if self.transformer.config.guidance_embeds and guidance_scale is None:
        raise ValueError("guidance_scale is required for guidance-distilled model.")
    elif self.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    elif not self.transformer.config.guidance_embeds and guidance_scale is not None:
        guidance = None
    else:
        guidance = None

    if self.attention_kwargs is None:
        self._attention_kwargs = {}

    self.scheduler.set_begin_index(0)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t
            latent_model_input = latents
            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1)

            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            latent_model_input = torch.cat([latent_model_input, latent_model_input], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)
            cfg_img_shapes = img_shapes + img_shapes
            cfg_guidance = torch.cat([guidance, guidance], dim=0) if guidance is not None else None

            with self.transformer.cache_context("batched_cfg"):
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=cfg_guidance,
                    encoder_hidden_states_mask=cfg_encoder_hidden_states_mask,
                    encoder_hidden_states=cfg_encoder_hidden_states,
                    img_shapes=cfg_img_shapes,
                    attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, : latents.size(1)]

            neg_noise_pred, noise_pred = noise_pred.chunk(2, dim=0)
            comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            noise_pred = comb_pred * (cond_norm / noise_norm)

            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

            if XLA_AVAILABLE:
                qie_edit_plus.xm.mark_step()

    self._current_timestep = None
    if output_type == "latent":
        image = latents
    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return QwenImagePipelineOutput(images=image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="convert to black and white")
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--no-cuda-int-prod-patch", action="store_true")
    parser.add_argument("--cfg-batching", action="store_true")
    parser.add_argument("--cfg-batching-no-mask-padding", action="store_true")
    parser.add_argument(
        "--fp8-backend",
        choices=["none", "torchao-float8wo", "torchao-float8dq"],
        default="none",
        help="Optional FP8 quantization backend for the QwenImage transformer.",
    )
    parser.add_argument(
        "--fp8-modules-to-not-convert",
        default=None,
        help="Comma-separated module name substrings to leave unquantized for TorchAO FP8.",
    )
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--torch-compile-mode", default="max-autotune")
    parser.add_argument("--torch-compile-fullgraph", action="store_true")
    parser.add_argument(
        "--compile-warmup-runs",
        type=int,
        default=0,
        help="Run same-shape generations before the measured run. Useful because torch.compile is lazy.",
    )
    args = parser.parse_args()

    output_path = Path(args.output).expanduser()
    metrics_path = Path(args.metrics).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    if not args.no_cuda_int_prod_patch:
        patch_cuda_integer_prod()

    fp8_modules_to_not_convert = parse_module_list(args.fp8_modules_to_not_convert)

    quantized_transformer = None
    transformer_load_wall = None
    if args.fp8_backend.startswith("torchao-"):
        quantization_config = build_torchao_fp8_config(args.fp8_backend, fp8_modules_to_not_convert)
        transformer_t0 = time.perf_counter()
        quantized_transformer = QwenImageTransformer2DModel.from_pretrained(
            args.model,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            local_files_only=args.local_files_only,
        )
        transformer_t1 = time.perf_counter()
        transformer_load_wall = transformer_t1 - transformer_t0
        print(
            f"FP8_TRANSFORMER_LOADED backend={args.fp8_backend} wall_s={transformer_load_wall:.6f} "
            f"modules_to_not_convert={fp8_modules_to_not_convert}",
            flush=True,
        )

    t0 = time.perf_counter()
    pipe_kwargs = {}
    if quantized_transformer is not None:
        pipe_kwargs["transformer"] = quantized_transformer
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
        **pipe_kwargs,
    )
    t1 = time.perf_counter()
    pipe.to("cuda")
    sync_cuda()
    t2 = time.perf_counter()
    pipe.set_progress_bar_config(disable=args.disable_progress)

    compile_wrap_wall = None
    if args.torch_compile:
        compile_t0 = time.perf_counter()
        pipe.transformer = torch.compile(
            pipe.transformer,
            mode=args.torch_compile_mode,
            fullgraph=args.torch_compile_fullgraph,
        )
        compile_t1 = time.perf_counter()
        compile_wrap_wall = compile_t1 - compile_t0
        print(
            "TORCH_COMPILE_WRAPPED "
            f"mode={args.torch_compile_mode} fullgraph={args.torch_compile_fullgraph} "
            f"wall_s={compile_wrap_wall:.6f}",
            flush=True,
        )

    metrics = {
        "model": args.model,
        "image": args.image,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "true_cfg_scale": args.true_cfg_scale,
        "guidance_scale": args.guidance_scale,
        "cfg_mode": "batched" if args.cfg_batching else "sequential",
        "cfg_batched_no_mask_padding": args.cfg_batching_no_mask_padding,
        "fp8_backend": args.fp8_backend,
        "fp8_modules_to_not_convert": fp8_modules_to_not_convert,
        "fp8_transformer_load_wall_s": transformer_load_wall,
        "torch_compile": args.torch_compile,
        "torch_compile_mode": args.torch_compile_mode if args.torch_compile else None,
        "torch_compile_fullgraph": args.torch_compile_fullgraph if args.torch_compile else None,
        "torch_compile_wrap_wall_s": compile_wrap_wall,
        "compile_warmup_runs": args.compile_warmup_runs,
        "seed": args.seed,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "load_wall_s": t1 - t0,
        "to_cuda_wall_s": t2 - t1,
    }

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        metrics["mem_after_load_free_gib"] = bytes_to_gib(free)
        metrics["mem_after_load_total_gib"] = bytes_to_gib(total)

    if args.load_only:
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
        return

    original_forward = pipe.transformer.forward
    transformer_call_times = []

    def timed_transformer_forward(*forward_args, **forward_kwargs):
        sync_cuda()
        if args.torch_compile and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        start = time.perf_counter()
        result = original_forward(*forward_args, **forward_kwargs)
        if args.torch_compile:
            result = clone_tensors(result)
        sync_cuda()
        elapsed = time.perf_counter() - start
        transformer_call_times.append(elapsed)
        print(f"TRANSFORMER_CALL {len(transformer_call_times)} wall_s={elapsed:.6f}", flush=True)
        return result

    pipe.transformer.forward = timed_transformer_forward

    image = Image.open(Path(args.image).expanduser()).convert("RGB")

    def run_generation(run_label, save_path=None):
        callback_step_times = []
        callback_last = {"time": None}
        infer_start_holder = {"time": None}

        def on_step_end(_pipe, step, timestep, callback_kwargs):
            sync_cuda()
            now = time.perf_counter()
            infer_start = infer_start_holder["time"]
            if callback_last["time"] is None:
                elapsed = now - infer_start
            else:
                elapsed = now - callback_last["time"]
            callback_last["time"] = now
            callback_step_times.append(elapsed)
            timestep_value = timestep.item() if hasattr(timestep, "item") else timestep
            print(
                f"{run_label}_STEP_END {step + 1}/{args.steps} timestep={timestep_value} "
                f"callback_delta_s={elapsed:.6f}",
                flush=True,
            )
            return callback_kwargs

        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        call = qwen_edit_plus_call_batched_cfg if args.cfg_batching else QwenImageEditPlusPipeline.__call__
        pipeline_kwargs = {
            "image": [image],
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "true_cfg_scale": args.true_cfg_scale,
            "guidance_scale": args.guidance_scale,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.steps,
            "num_images_per_prompt": 1,
            "generator": generator,
            "max_sequence_length": args.max_sequence_length,
            "callback_on_step_end": on_step_end,
        }
        if args.cfg_batching:
            pipeline_kwargs["cfg_force_no_padding_mask"] = args.cfg_batching_no_mask_padding

        sync_cuda()
        infer_start_holder["time"] = time.perf_counter()
        with torch.inference_mode():
            result = call(pipe, **pipeline_kwargs)
        sync_cuda()
        infer_end = time.perf_counter()
        if save_path is not None:
            result.images[0].save(save_path)
        return infer_end - infer_start_holder["time"], callback_step_times

    warmup_metrics = []
    for warmup_index in range(args.compile_warmup_runs):
        before_calls = len(transformer_call_times)
        warmup_wall, warmup_callback_step_times = run_generation(f"WARMUP{warmup_index + 1}")
        warmup_call_times = transformer_call_times[before_calls:]
        warmup_metrics.append(
            {
                "run": warmup_index + 1,
                "infer_wall_s": warmup_wall,
                "transformer_call_times_s": warmup_call_times,
                "transformer_calls": summarize(warmup_call_times),
                "callback_step_deltas_s": warmup_callback_step_times,
                "callback_steps": summarize(warmup_callback_step_times),
            }
        )
        transformer_call_times.clear()

    torch.cuda.reset_peak_memory_stats()
    infer_wall, callback_step_times = run_generation("STEP", output_path)

    calls_per_step = 1 if args.cfg_batching else (2 if args.true_cfg_scale and args.true_cfg_scale > 1.0 else 1)
    grouped_transformer_steps = [
        sum(transformer_call_times[i : i + calls_per_step])
        for i in range(0, len(transformer_call_times), calls_per_step)
        if len(transformer_call_times[i : i + calls_per_step]) == calls_per_step
    ]

    metrics.update(
        {
            "compile_warmups": warmup_metrics,
            "infer_wall_s": infer_wall,
            "output": str(output_path),
            "transformer_call_times_s": transformer_call_times,
            "transformer_calls": summarize(transformer_call_times),
            "transformer_grouped_step_times_s": grouped_transformer_steps,
            "transformer_grouped_steps": summarize(grouped_transformer_steps),
            "callback_step_deltas_s": callback_step_times,
            "callback_steps": summarize(callback_step_times),
            "peak_allocated_gib": bytes_to_gib(torch.cuda.max_memory_allocated()),
            "peak_reserved_gib": bytes_to_gib(torch.cuda.max_memory_reserved()),
        }
    )
    free, total = torch.cuda.mem_get_info()
    metrics["mem_after_infer_free_gib"] = bytes_to_gib(free)
    metrics["mem_after_infer_total_gib"] = bytes_to_gib(total)

    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
