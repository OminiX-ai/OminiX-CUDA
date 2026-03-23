#include "stft.h"
#include "../../vendor/kissfft/kiss_fft.h"
#include "../../vendor/kissfft/kiss_fftr.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace stft {

std::vector<float> hann_window(int size) {
  std::vector<float> window(size);
  const double pi = 3.14159265358979323846;
  for (int i = 0; i < size; ++i) {
    // Periodic Hann window (like PyTorch's torch.hann_window)
    window[i] = 0.5f * (1.0f - std::cos(2.0 * pi * i / size));
  }
  return window;
}

void reflect_pad(const std::vector<float> &input, int pad_left, int pad_right,
                 std::vector<float> &output) {
  int input_size = static_cast<int>(input.size());
  int output_size = input_size + pad_left + pad_right;
  output.resize(output_size);

  // Copy original data
  std::copy(input.begin(), input.end(), output.begin() + pad_left);

  // Left padding (reflect)
  for (int i = 0; i < pad_left; ++i) {
    int src_idx = pad_left - i;
    if (src_idx >= input_size) {
      src_idx = input_size - 1;
    }
    output[i] = input[src_idx];
  }

  // Right padding (reflect)
  for (int i = 0; i < pad_right; ++i) {
    int src_idx = input_size - 2 - i;
    if (src_idx < 0) {
      src_idx = 0;
    }
    output[pad_left + input_size + i] = input[src_idx];
  }
}

bool compute_stft(const std::vector<float> &audio, const STFTParams &params,
                  std::vector<float> &magnitude, int &n_frames) {
  int n_fft = params.n_fft;
  int hop_length = params.hop_length;
  int win_length = params.win_length;

  if (audio.empty()) {
    fprintf(stderr, "STFT: empty input audio\n");
    return false;
  }

  // Compute padding (like PyTorch with center=False, using reflect padding)
  int pad_amount = (n_fft - hop_length) / 2;

  // Reflect pad the audio
  std::vector<float> padded_audio;
  reflect_pad(audio, pad_amount, pad_amount, padded_audio);

  // Compute number of frames
  int padded_len = static_cast<int>(padded_audio.size());
  n_frames = (padded_len - n_fft) / hop_length + 1;

  if (n_frames <= 0) {
    fprintf(stderr, "STFT: audio too short for given parameters\n");
    return false;
  }

  // Prepare Hann window
  std::vector<float> window = hann_window(win_length);

  // Initialize kissfft for real FFT
  kiss_fftr_cfg cfg = kiss_fftr_alloc(n_fft, 0, nullptr, nullptr);
  if (!cfg) {
    fprintf(stderr, "STFT: failed to allocate kissfft config\n");
    return false;
  }

  // Output frequency bins = n_fft/2 + 1
  int n_bins = n_fft / 2 + 1;
  magnitude.resize(n_bins * n_frames);

  // Temporary buffers
  std::vector<float> frame(n_fft);
  std::vector<kiss_fft_cpx> fft_out(n_bins);

  // Process each frame
  for (int f = 0; f < n_frames; ++f) {
    int start = f * hop_length;

    // Extract frame and apply window
    for (int i = 0; i < n_fft; ++i) {
      if (i < win_length) {
        frame[i] = padded_audio[start + i] * window[i];
      } else {
        frame[i] = 0.0f; // Zero pad if n_fft > win_length
      }
    }

    // Compute FFT
    kiss_fftr(cfg, frame.data(), fft_out.data());

    // Compute magnitude: sqrt(real^2 + imag^2 + 1e-6)
    for (int b = 0; b < n_bins; ++b) {
      float real = fft_out[b].r;
      float imag = fft_out[b].i;
      float mag = std::sqrt(real * real + imag * imag + 1e-6f);
      // Store in row-major order: magnitude[bin][frame]
      magnitude[b * n_frames + f] = mag;
    }
  }

  kiss_fft_free(cfg);
  return true;
}

bool spectrogram_torch(const std::vector<float> &audio_32k,
                       std::vector<float> &spec, int &n_frames) {
  STFTParams params = get_default_params();
  return compute_stft(audio_32k, params, spec, n_frames);
}

} // namespace stft
