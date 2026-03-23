// Test: decode Python reference codes to check decoder quality
#include "speech_tokenizer_decoder.h"
#include "audio_io.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <string>

// Simple NPY loader for int64 arrays
bool load_npy_int64(const std::string &path, std::vector<std::vector<int>> &codes,
                     int &n_frames, int &n_groups) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    char magic[6];
    f.read(magic, 6);
    uint8_t major, minor;
    f.read((char*)&major, 1);
    f.read((char*)&minor, 1);
    uint16_t header_len;
    f.read((char*)&header_len, 2);

    std::string header(header_len, '\0');
    f.read(header.data(), header_len);

    // Parse shape from header like "{'descr': '<i8', 'fortran_order': False, 'shape': (271, 16), }"
    auto pos = header.find("shape");
    if (pos == std::string::npos) return false;
    pos = header.find('(', pos);
    auto end = header.find(')', pos);
    std::string shape_str = header.substr(pos + 1, end - pos - 1);
    // Parse "271, 16"
    sscanf(shape_str.c_str(), "%d, %d", &n_frames, &n_groups);

    printf("Loading %s: shape=(%d, %d)\n", path.c_str(), n_frames, n_groups);

    codes.resize(n_groups);
    for (int g = 0; g < n_groups; g++) codes[g].resize(n_frames);

    // Read data: shape (n_frames, n_groups), dtype int64
    for (int t = 0; t < n_frames; t++) {
        for (int g = 0; g < n_groups; g++) {
            int64_t val;
            f.read((char*)&val, 8);
            codes[g][t] = (int)val;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    const char *gguf_dir = "tools/qwen_tts/gguf/";
    const char *npy_path = "logs/python_ref_codec_tokens.npy";
    const char *output_path = "output_decoder_test.wav";

    if (argc > 1) npy_path = argv[1];
    if (argc > 2) output_path = argv[2];

    // Load decoder
    SpeechTokenizerDecoder decoder;
    std::string dec_path = std::string(gguf_dir) + "qwen_tts_tokenizer_dec.gguf";
    ContextParams ctx_params;
    if (!decoder.load(dec_path, ctx_params)) {
        printf("Failed to load decoder\n");
        return 1;
    }

    // Load Python ref codes
    int n_frames, n_groups;
    std::vector<std::vector<int>> codes;
    if (!load_npy_int64(npy_path, codes, n_frames, n_groups)) {
        printf("Failed to load %s\n", npy_path);
        return 1;
    }

    // Also load ref_codes and prepend
    int ref_frames = 0, ref_groups = 0;
    std::vector<std::vector<int>> ref_codes;
    if (load_npy_int64("logs/py_ref_codes.npy", ref_codes, ref_frames, ref_groups)) {
        printf("Prepending %d ref frames\n", ref_frames);
        std::vector<std::vector<int>> full_codes(n_groups);
        for (int g = 0; g < n_groups; g++) {
            full_codes[g].reserve(ref_frames + n_frames);
            if (g < ref_groups)
                full_codes[g].insert(full_codes[g].end(), ref_codes[g].begin(), ref_codes[g].end());
            full_codes[g].insert(full_codes[g].end(), codes[g].begin(), codes[g].end());
        }
        codes = full_codes;
        int total = ref_frames + n_frames;
        printf("Total frames: %d (ref=%d + gen=%d)\n", total, ref_frames, n_frames);

        // Decode
        std::vector<float> audio;
        if (!decoder.decode(codes, audio)) {
            printf("Decode failed\n");
            return 1;
        }

        // Strip ref portion (proportional)
        int cut = (int)((long long)ref_frames * audio.size() / total);
        std::vector<float> gen_audio(audio.begin() + cut, audio.end());
        printf("Decoded: %zu total samples, cut=%d, output=%zu samples (%.2f sec)\n",
               audio.size(), cut, gen_audio.size(), gen_audio.size() / 24000.0);

        audio_io::save_wav(output_path, gen_audio, 24000);
        printf("Saved to %s\n", output_path);
    }

    return 0;
}
