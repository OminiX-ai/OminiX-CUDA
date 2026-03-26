#include "qwen_tts.h"
#include "audio_io.h"
#include <chrono>
#include <cstdio>

int main() {
    QwenTTSParams params;
    params.model_dir = "tools/qwen_tts/gguf/";
    params.tokenizer_dir = "Qwen/Qwen3-TTS-12Hz-1.7B-Base/";
    params.ref_audio = "tools/qwen_tts/data/test_ref.wav";
    params.ref_text = "The key to life is not accumulation. It's contribution.";
    params.target_lang = "English";
    params.talker_model = "tools/qwen_tts/gguf/qwen_tts_talker_llama_q8_0.gguf";
    params.cp_model = "tools/qwen_tts/gguf/qwen_tts_cp_llama.gguf";
    params.n_gpu_layers = 29;
    params.n_threads = 8;

    QwenTTS tts;
    auto t_load = std::chrono::high_resolution_clock::now();
    if (!tts.load(params)) return 1;
    auto t_loaded = std::chrono::high_resolution_clock::now();
    printf("\n=== Models loaded in %.1fs ===\n\n",
        std::chrono::duration<double>(t_loaded - t_load).count());

    printf("%-6s %9s %8s  %s\n", "Round", "GenTime", "Audio", "Decoder");
    printf("──────────────────────────────────────────\n");

    params.max_new_tokens = 60;  // limit frames to stay within CANN decoder limit

    for (int r = 0; r < 5; r++) {
        params.text = "Hello, how are you?";
        params.output = "output/warmup_r" + std::to_string(r+1) + ".wav";
        set_sampling_seed(42);

        std::vector<float> audio;
        auto t0 = std::chrono::high_resolution_clock::now();
        tts.generate(params, audio);
        auto t1 = std::chrono::high_resolution_clock::now();
        double total = std::chrono::duration<double>(t1 - t0).count();

        if (!audio.empty())
            audio_io::save_wav(params.output, audio, 24000);

        printf("  %-4d  %7.2fs  %5.1fs\n", r+1, total, audio.size()/24000.0f);
    }
    return 0;
}
