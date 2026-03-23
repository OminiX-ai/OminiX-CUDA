#pragma once

#include <cmath>
#include <vector>

namespace stft {

struct STFTParams {
  int n_fft = 2048;
  int hop_length = 640;
  int win_length = 2048;
  int sampling_rate = 32000;
};

// Default params for GPT-SoVITS
inline STFTParams get_default_params() {
  return STFTParams{
      .n_fft = 2048,
      .hop_length = 640,
      .win_length = 2048,
      .sampling_rate = 32000,
  };
}

// Compute Hann window
std::vector<float> hann_window(int size);

// Reflect padding (like PyTorch's reflect mode)
// Pads input on both sides using reflected values
void reflect_pad(const std::vector<float> &input, int pad_left, int pad_right,
                 std::vector<float> &output);

// Compute STFT magnitude spectrum
// Input: audio (1D float), params
// Output: magnitude spectrum (n_fft/2+1 x n_frames, row-major)
// Returns false on error
bool compute_stft(const std::vector<float> &audio, const STFTParams &params,
                  std::vector<float> &magnitude, int &n_frames);

// Convenience function: compute spectrogram matching PyTorch's spectrogram_torch
// Parameters: n_fft=2048, hop_length=640, win_length=2048, center=False
// Output shape: 1025 x n_frames (row-major, 1025 = n_fft/2+1)
bool spectrogram_torch(const std::vector<float> &audio_32k,
                       std::vector<float> &spec, int &n_frames);

} // namespace stft
