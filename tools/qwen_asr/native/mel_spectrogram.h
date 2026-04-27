#pragma once

#include <string>
#include <vector>

// Whisper-style mel spectrogram extraction for Qwen3-ASR
// Parameters: sr=16000, n_fft=400, hop_length=160, n_mels=128
//
// Normalization follows WhisperFeatureExtractor:
//   log_spec = log10(max(mel, 1e-10))
//   log_spec = max(log_spec, max(log_spec) - 8.0)
//   log_spec = (log_spec + 4.0) / 4.0

class MelSpectrogram {
public:
    MelSpectrogram(int sample_rate = 16000, int n_fft = 400,
                   int hop_length = 160, int n_mels = 128);

    // Compute mel spectrogram from audio samples
    // audio: PCM float samples (mono, at sample_rate)
    // output: (n_mels, num_frames) in row-major
    // num_frames: output frame count
    // Returns true on success
    bool compute(const std::vector<float> &audio,
                 std::vector<float> &output, int &num_frames);

    int get_n_mels() const { return n_mels_; }
    int get_sample_rate() const { return sample_rate_; }

    // Load mel filterbank from a numpy .npy file (for exact match with Python)
    bool load_mel_filterbank(const std::string &npy_path);

private:
    int sample_rate_;
    int n_fft_;
    int hop_length_;
    int n_mels_;
    std::vector<float> hann_window_;
    std::vector<float> mel_filterbank_; // (n_mels, n_fft/2+1) row-major

    void init_hann_window();
    void init_mel_filterbank();
};
