// Quick test to dump mel spectrogram from C++ for comparison
#include "mel_spectrogram.h"
#include "audio_io.h"
#include <cstdio>
#include <fstream>

int main(int argc, char *argv[]) {
    std::string audio_path = argc > 1 ? argv[1] : "ellen_ref.wav";

    std::vector<float> audio_16k;
    if (!audio_io::load_audio(audio_path, 16000, audio_16k)) {
        printf("Failed to load audio\n");
        return 1;
    }
    printf("Audio: %zu samples\n", audio_16k.size());

    MelSpectrogram mel_spec;
    // Try loading Python mel filterbank for exact match
    mel_spec.load_mel_filterbank("tools/qwen_asr/verify_data/mel_filters_whisper.npy");
    std::vector<float> mel;
    int mel_T = 0;
    if (!mel_spec.compute(audio_16k, mel, mel_T)) {
        printf("Mel failed\n");
        return 1;
    }
    printf("Mel: %d bins x %d frames\n", mel_spec.get_n_mels(), mel_T);

    // Print first few values for comparison
    printf("mel[:3,:5]:\n");
    for (int m = 0; m < 3; m++) {
        for (int t = 0; t < 5; t++) {
            printf("%.6f ", mel[m * mel_T + t]);
        }
        printf("\n");
    }

    // Save to binary file for Python comparison
    std::ofstream fout("tools/qwen_asr/verify_data/mel_cpp.bin", std::ios::binary);
    int dims[2] = {128, mel_T};
    fout.write((char*)dims, sizeof(dims));
    fout.write((char*)mel.data(), mel.size() * sizeof(float));
    fout.close();
    printf("Saved to mel_cpp.bin\n");

    return 0;
}
