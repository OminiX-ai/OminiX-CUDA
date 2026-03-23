#include "qwen_asr.h"
#include <cstdio>
#include <cstring>
#include <string>

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  --model_dir DIR       Model directory (default: Qwen/Qwen3-ASR-1.7B)\n");
    printf("  --audio FILE          Input audio file (WAV)\n");
    printf("  --encoder FILE        Audio encoder GGUF path\n");
    printf("  --decoder FILE        Text decoder GGUF path (llama.cpp format)\n");
    printf("  --vocab FILE          vocab.json path\n");
    printf("  --merges FILE         merges.txt path\n");
    printf("  --device DEV          Backend device (default: CPU)\n");
    printf("  --threads N           Number of threads (default: 4)\n");
    printf("  --gpu_layers N        GPU layers for decoder (default: 0)\n");
    printf("  --max_tokens N        Max output tokens (default: 256)\n");
    printf("  -h, --help            Show this help\n");
}

int main(int argc, char *argv[]) {
    std::string model_dir = "Qwen/Qwen3-ASR-1.7B";
    std::string audio_path = "ellen_ref.wav";
    std::string encoder_path;
    std::string decoder_path;
    std::string vocab_path;
    std::string merges_path;
    std::string mel_filters_path;
    std::string device = "CPU";
    int n_threads = 4;
    int n_gpu_layers = 0;
    int max_tokens = 256;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model_dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--audio") == 0 && i + 1 < argc) {
            audio_path = argv[++i];
        } else if (strcmp(argv[i], "--encoder") == 0 && i + 1 < argc) {
            encoder_path = argv[++i];
        } else if (strcmp(argv[i], "--decoder") == 0 && i + 1 < argc) {
            decoder_path = argv[++i];
        } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (strcmp(argv[i], "--merges") == 0 && i + 1 < argc) {
            merges_path = argv[++i];
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device = argv[++i];
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gpu_layers") == 0 && i + 1 < argc) {
            n_gpu_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max_tokens") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    // Set default paths if not specified
    if (encoder_path.empty())
        encoder_path = "tools/qwen_asr/gguf/qwen_asr_audio_encoder.gguf";
    if (decoder_path.empty())
        decoder_path = "tools/qwen_asr/gguf/qwen_asr_decoder_q8_0.gguf";
    if (vocab_path.empty())
        vocab_path = model_dir + "/vocab.json";
    if (merges_path.empty())
        merges_path = model_dir + "/merges.txt";
    if (mel_filters_path.empty())
        mel_filters_path = "tools/qwen_asr/verify_data/mel_filters_whisper.npy";

    printf("=== Qwen3-ASR C++ Inference ===\n");
    printf("Model dir: %s\n", model_dir.c_str());
    printf("Audio: %s\n", audio_path.c_str());
    printf("Encoder: %s\n", encoder_path.c_str());
    printf("Decoder: %s\n", decoder_path.c_str());
    printf("Device: %s, threads: %d, gpu_layers: %d\n",
           device.c_str(), n_threads, n_gpu_layers);
    printf("\n");

    QwenASRParams params;
    params.model_dir = model_dir;
    params.audio_encoder_path = encoder_path;
    params.decoder_path = decoder_path;
    params.vocab_path = vocab_path;
    params.merges_path = merges_path;
    params.mel_filters_path = mel_filters_path;
    params.device = device;
    params.n_threads = n_threads;
    params.n_gpu_layers = n_gpu_layers;
    params.max_new_tokens = max_tokens;

    QwenASR asr;
    if (!asr.load(params)) {
        printf("Failed to load model\n");
        return 1;
    }

    printf("\n--- Transcribing ---\n");
    std::string output_text;

    if (!asr.transcribe(audio_path, output_text)) {
        printf("Transcription failed\n");
        return 1;
    }

    printf("\n=== Result ===\n%s\n", output_text.c_str());
    return 0;
}
