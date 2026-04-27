// ============================================================================
// Phase 4.2 smoke test — AudioEncoderCudaEngine forward.
//
// Usage:
//   test_audio_encoder_cuda <audio_encoder_gguf> <wav_path>
//
// Loads a WAV via qwen_common::load_wav, computes the mel spectrogram on the
// CPU, runs the CUDA audio encoder forward, then prints the output shape +
// stats and verifies NaN/Inf-freeness. Exits 0 on success.
// ============================================================================

#include "audio_encoder_cuda_engine.h"
#include "mel_spectrogram.h"

#include "../../qwen_common/audio_io.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: test_audio_encoder_cuda <audio_encoder_gguf> "
                "<wav_path>\n");
        return 2;
    }
    const std::string gguf_path = argv[1];
    const std::string wav_path  = argv[2];

    // 1. Init engine.
    ominix_cuda::AudioEncoderCudaEngine eng;
    if (!eng.init_from_gguf(gguf_path, /*device=*/0)) {
        fprintf(stderr, "[smoke] init_from_gguf FAILED\n");
        return 1;
    }
    fprintf(stderr,
            "[smoke] engine ready: layers=%d d_model=%d heads=%d ffn=%d "
            "mels=%d out=%d max_pos=%d\n",
            eng.encoder_layers(), eng.d_model(), eng.encoder_heads(),
            eng.encoder_ffn_dim(), eng.num_mel_bins(), eng.output_dim(),
            eng.max_source_pos());

    // 2. Load WAV.
    audio_io::AudioData audio;
    if (!audio_io::load_wav(wav_path, audio)) {
        fprintf(stderr, "[smoke] load_wav FAILED: %s\n", wav_path.c_str());
        return 1;
    }
    fprintf(stderr,
            "[smoke] WAV loaded: samples=%zu  sr=%d\n",
            audio.samples.size(), audio.sample_rate);
    if (audio.sample_rate != 16000) {
        fprintf(stderr,
                "[smoke] WARNING: expected sr=16000, got %d — proceeding.\n",
                audio.sample_rate);
    }

    // 3. Compute mel spectrogram on host.
    //    Phase 4.5 parity-debug: if OMINIX_ASR_USE_MEL_BIN=<path> is set, load
    //    raw F32 mel of shape (n_mels, mel_T) from that file instead of
    //    computing from the WAV. mel_T is inferred from file size.
    std::vector<float> mel;
    int mel_T = 0;
    const char *mel_bin_env = std::getenv("OMINIX_ASR_USE_MEL_BIN");
    if (mel_bin_env && *mel_bin_env) {
        FILE *f = std::fopen(mel_bin_env, "rb");
        if (!f) {
            fprintf(stderr, "[smoke] cannot open mel bin: %s\n", mel_bin_env);
            return 1;
        }
        std::fseek(f, 0, SEEK_END);
        long bytes = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        size_t nfloats = (size_t)bytes / sizeof(float);
        mel_T = (int)(nfloats / (size_t)eng.num_mel_bins());
        mel.resize(nfloats);
        std::fread(mel.data(), sizeof(float), nfloats, f);
        std::fclose(f);
        fprintf(stderr,
                "[smoke] mel LOADED from %s: shape=(%d, %d) elems=%zu\n",
                mel_bin_env, eng.num_mel_bins(), mel_T, mel.size());
    } else {
        MelSpectrogram mel_spec(/*sample_rate=*/16000, /*n_fft=*/400,
                                 /*hop_length=*/160, /*n_mels=*/eng.num_mel_bins());
        // Phase 4.6: load HF whisper filterbank (slaney) — built-in HTK bank
        // diverges from HF reference (cossim ~0.07 on filters).
        {
            const char *envp = std::getenv("OMINIX_ASR_MEL_FILTERS");
            std::string mfp = envp && *envp
                ? std::string(envp)
                : std::string("tools/qwen_asr/verify_data/mel_filters_whisper.npy");
            if (!mel_spec.load_mel_filterbank(mfp)) {
                fprintf(stderr,
                        "[smoke] WARN: failed to load mel filters from %s — "
                        "proceeding with built-in HTK (will diverge from HF).\n",
                        mfp.c_str());
            }
        }
        if (!mel_spec.compute(audio.samples, mel, mel_T)) {
            fprintf(stderr, "[smoke] mel_spec.compute FAILED\n");
            return 1;
        }
        fprintf(stderr,
                "[smoke] mel computed: shape=(%d, %d) elems=%zu\n",
                eng.num_mel_bins(), mel_T, mel.size());
    }

    // 4. Run encoder forward.
    int max_frames = mel_T;   // generous upper bound (output ≤ mel_T / 4)
    std::vector<float> embeds((size_t)max_frames * eng.output_dim(), 0.0f);
    int num_frames = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = eng.encode(mel.data(), eng.num_mel_bins(), mel_T,
                          embeds.data(), num_frames);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (!ok) {
        fprintf(stderr, "[smoke] encode FAILED\n");
        return 1;
    }
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr,
            "[smoke] encode OK: num_frames=%d  output_dim=%d  wall=%.3f ms\n",
            num_frames, eng.output_dim(), ms);

    // 5. Stats / NaN/Inf check on the first num_frames * output_dim values.
    size_t valid_n = (size_t)num_frames * eng.output_dim();
    size_t n_nan = 0, n_inf = 0;
    double sum = 0.0, sum_sq = 0.0;
    float vmin = +INFINITY, vmax = -INFINITY;
    for (size_t i = 0; i < valid_n; ++i) {
        float v = embeds[i];
        if (std::isnan(v)) { ++n_nan; continue; }
        if (std::isinf(v)) { ++n_inf; continue; }
        sum    += (double)v;
        sum_sq += (double)v * (double)v;
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }
    size_t finite_n = valid_n - n_nan - n_inf;
    double mean = finite_n > 0 ? sum / (double)finite_n : 0.0;
    double var  = finite_n > 0 ? (sum_sq / (double)finite_n) - mean * mean : 0.0;
    if (var < 0.0) var = 0.0;
    double std_dev = std::sqrt(var);
    fprintf(stderr,
            "[smoke] embeds stats: min=%.4f max=%.4f mean=%.4f std=%.4f "
            "NaN=%zu Inf=%zu\n",
            vmin, vmax, (float)mean, (float)std_dev, n_nan, n_inf);

    if (n_nan > 0 || n_inf > 0) {
        fprintf(stderr, "[smoke] FAIL: non-finite values present.\n");
        return 1;
    }
    if (num_frames <= 0) {
        fprintf(stderr, "[smoke] FAIL: num_frames=0.\n");
        return 1;
    }
    fprintf(stderr,
            "[smoke] Phase 4.2 audio encoder forward PASS  "
            "shape=(%d, %d)  finite_n=%zu\n",
            num_frames, eng.output_dim(), finite_n);
    return 0;
}
