#include "mel_spectrogram.h"
#include "../../vendor/kissfft/kiss_fft.h"
#include "../../vendor/kissfft/kiss_fftr.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

// ============================================================================
// Mel scale conversion (HTK formula)
// ============================================================================

static float hz_to_mel_htk(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

static float mel_to_hz_htk(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

// ============================================================================
// MelSpectrogram implementation
// ============================================================================

MelSpectrogram::MelSpectrogram(int sample_rate, int n_fft,
                               int hop_length, int n_mels)
    : sample_rate_(sample_rate), n_fft_(n_fft),
      hop_length_(hop_length), n_mels_(n_mels) {
    init_hann_window();
    init_mel_filterbank();
}

void MelSpectrogram::init_hann_window() {
    // Periodic Hann window (matching torch.hann_window with periodic=True)
    hann_window_.resize(n_fft_);
    const double pi = 3.14159265358979323846;
    for (int i = 0; i < n_fft_; i++) {
        hann_window_[i] = 0.5f * (1.0f - std::cos(2.0 * pi * i / n_fft_));
    }
}

void MelSpectrogram::init_mel_filterbank() {
    // Create mel filterbank matching HuggingFace's mel_filter_bank()
    // with norm="slaney", mel_scale="htk"
    // Layout: (n_freqs, n_mels) - same as HuggingFace's mel_filters
    int n_freqs = n_fft_ / 2 + 1;
    mel_filterbank_.resize(n_freqs * n_mels_, 0.0f);

    float fmin = 0.0f;
    float fmax = (float)sample_rate_ / 2.0f;

    float mel_min = hz_to_mel_htk(fmin);
    float mel_max = hz_to_mel_htk(fmax);

    // n_mels + 2 equally spaced points in mel scale
    std::vector<float> mel_pts(n_mels_ + 2);
    for (int i = 0; i < n_mels_ + 2; i++) {
        mel_pts[i] = mel_min + (mel_max - mel_min) * i / (n_mels_ + 1);
    }

    // Convert back to Hz
    std::vector<float> hz_pts(n_mels_ + 2);
    for (int i = 0; i < n_mels_ + 2; i++) {
        hz_pts[i] = mel_to_hz_htk(mel_pts[i]);
    }

    // FFT frequency bins
    std::vector<float> fft_freqs(n_freqs);
    for (int i = 0; i < n_freqs; i++) {
        fft_freqs[i] = (float)sample_rate_ * i / n_fft_;
    }

    // Triangular filters with Slaney normalization
    // Store in (n_freqs, n_mels) layout to match HuggingFace
    for (int m = 0; m < n_mels_; m++) {
        float left = hz_pts[m];
        float center = hz_pts[m + 1];
        float right = hz_pts[m + 2];

        // Slaney normalization factor: 2 / (right - left)
        float enorm = 2.0f / (right - left);

        for (int f = 0; f < n_freqs; f++) {
            float freq = fft_freqs[f];
            float val = 0.0f;
            if (freq >= left && freq <= center && center > left) {
                val = enorm * (freq - left) / (center - left);
            } else if (freq > center && freq <= right && right > center) {
                val = enorm * (right - freq) / (right - center);
            }
            // Layout: (n_freqs, n_mels), index [f, m]
            mel_filterbank_[f * n_mels_ + m] = val;
        }
    }
}

bool MelSpectrogram::compute(const std::vector<float> &audio,
                              std::vector<float> &output, int &num_frames) {
    if (audio.empty()) {
        fprintf(stderr, "MelSpectrogram: empty input audio\n");
        return false;
    }

    // ========================================================================
    // Step 1: Pad audio (reflect padding for center=True STFT)
    // ========================================================================
    int pad_amount = n_fft_ / 2;
    int padded_len = (int)audio.size() + 2 * pad_amount;

    std::vector<float> padded_audio(padded_len);

    // Copy original audio
    std::memcpy(padded_audio.data() + pad_amount, audio.data(),
                audio.size() * sizeof(float));

    // Reflect padding on left: audio[1], audio[2], ..., audio[pad]
    for (int i = 0; i < pad_amount; i++) {
        int src_idx = pad_amount - i; // 1, 2, ..., pad_amount
        if (src_idx >= (int)audio.size()) src_idx = (int)audio.size() - 1;
        padded_audio[i] = audio[src_idx];
    }

    // Reflect padding on right: audio[-2], audio[-3], ...
    for (int i = 0; i < pad_amount; i++) {
        int src_idx = (int)audio.size() - 2 - i;
        if (src_idx < 0) src_idx = 0;
        padded_audio[pad_amount + (int)audio.size() + i] = audio[src_idx];
    }

    // ========================================================================
    // Step 2: Compute STFT frames
    // WhisperFeatureExtractor drops the last frame: log_spec = log_spec[:, :-1]
    // So: stft produces N+1 frames, we keep N frames
    // ========================================================================
    int stft_frames = (padded_len - n_fft_) / hop_length_ + 1;
    num_frames = stft_frames - 1; // Drop last frame to match Whisper
    if (num_frames <= 0) {
        fprintf(stderr, "MelSpectrogram: audio too short (%zu samples)\n", audio.size());
        return false;
    }

    int n_freqs = n_fft_ / 2 + 1;

    // Initialize kissfft for real FFT
    kiss_fftr_cfg cfg = kiss_fftr_alloc(n_fft_, 0, nullptr, nullptr);
    if (!cfg) {
        fprintf(stderr, "MelSpectrogram: failed to allocate kissfft config\n");
        return false;
    }

    // Compute STFT and apply mel filterbank in one pass
    // mel_spec: (n_mels, num_frames) row-major
    std::vector<float> mel_spec(n_mels_ * num_frames, 0.0f);

    std::vector<float> frame(n_fft_);
    std::vector<kiss_fft_cpx> fft_out(n_freqs);

    for (int t = 0; t < num_frames; t++) {
        int start = t * hop_length_;

        // Extract frame and apply Hann window
        for (int i = 0; i < n_fft_; i++) {
            frame[i] = padded_audio[start + i] * hann_window_[i];
        }

        // Compute FFT
        kiss_fftr(cfg, frame.data(), fft_out.data());

        // Compute power spectrum and apply mel filterbank
        // mel_spec[m, t] = sum_f(mel_filterbank[f, m] * |FFT[f]|^2)
        for (int f = 0; f < n_freqs; f++) {
            float power = fft_out[f].r * fft_out[f].r + fft_out[f].i * fft_out[f].i;
            // mel_filterbank layout: (n_freqs, n_mels), index [f, m]
            const float *filters = mel_filterbank_.data() + f * n_mels_;
            for (int m = 0; m < n_mels_; m++) {
                mel_spec[m * num_frames + t] += filters[m] * power;
            }
        }
    }

    kiss_fft_free(cfg);

    // ========================================================================
    // Step 3: Log mel + Whisper normalization
    // log_spec = log10(max(mel, 1e-10))
    // log_spec = max(log_spec, max(log_spec) - 8.0)
    // log_spec = (log_spec + 4.0) / 4.0
    // ========================================================================
    output.resize(n_mels_ * num_frames);

    // First pass: log10
    float global_max = -1e20f;
    for (int i = 0; i < n_mels_ * num_frames; i++) {
        float val = std::log10(std::max(mel_spec[i], 1e-10f));
        output[i] = val;
        if (val > global_max) global_max = val;
    }

    // Second pass: clamp and normalize
    float clamp_min = global_max - 8.0f;
    for (int i = 0; i < n_mels_ * num_frames; i++) {
        float val = std::max(output[i], clamp_min);
        output[i] = (val + 4.0f) / 4.0f;
    }

    printf("MelSpectrogram: %d frames, %d mels, global_max=%.4f\n",
           num_frames, n_mels_, global_max);

    return true;
}

bool MelSpectrogram::load_mel_filterbank(const std::string &npy_path) {
    // Simple .npy loader for float32 2D arrays
    std::ifstream f(npy_path, std::ios::binary);
    if (!f.is_open()) return false;

    // Read .npy header
    char magic[6];
    f.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") return false;

    uint8_t major, minor;
    f.read((char*)&major, 1);
    f.read((char*)&minor, 1);

    uint16_t header_len;
    if (major == 1) {
        f.read((char*)&header_len, 2);
    } else {
        uint32_t hl;
        f.read((char*)&hl, 4);
        header_len = (uint16_t)hl;
    }

    std::string header(header_len, '\0');
    f.read(&header[0], header_len);

    // Parse shape from header
    auto shape_pos = header.find("'shape': (");
    if (shape_pos == std::string::npos) return false;
    shape_pos += 10;
    auto shape_end = header.find(")", shape_pos);
    std::string shape_str = header.substr(shape_pos, shape_end - shape_pos);

    int dim0 = 0, dim1 = 0;
    auto comma = shape_str.find(",");
    dim0 = std::stoi(shape_str.substr(0, comma));
    if (comma != std::string::npos && comma + 1 < shape_str.size()) {
        std::string s2 = shape_str.substr(comma + 1);
        while (!s2.empty() && s2[0] == ' ') s2 = s2.substr(1);
        if (!s2.empty()) dim1 = std::stoi(s2);
    }

    printf("MelSpectrogram: loaded filterbank from %s: (%d, %d)\n",
           npy_path.c_str(), dim0, dim1);

    // Read data - shape should be (n_freqs=201, n_mels=128)
    int n_freqs = n_fft_ / 2 + 1;
    if (dim0 != n_freqs || dim1 != n_mels_) {
        printf("MelSpectrogram: unexpected filterbank shape (%d, %d), expected (%d, %d)\n",
               dim0, dim1, n_freqs, n_mels_);
        return false;
    }

    mel_filterbank_.resize(dim0 * dim1);
    f.read((char*)mel_filterbank_.data(), dim0 * dim1 * sizeof(float));
    f.close();
    return true;
}
