#pragma once

#include <cmath>
#include <string>
#include <vector>

namespace audio_io {

struct AudioData {
  std::vector<float> samples; // normalized to [-1, 1]
  int sample_rate;
  int channels;
};

// Load WAV file and convert to mono
bool load_wav(const std::string &path, AudioData &audio);

// Resample audio to target sample rate
bool resample(const AudioData &input, int target_sr, std::vector<float> &output);

// Load audio file and resample to target sample rate (convenience function)
bool load_audio(const std::string &path, int target_sr,
                std::vector<float> &output);

// Normalize volume: if max > 1, divide by min(2, max)
void normalize_volume(std::vector<float> &audio);

// Add zero padding at the end
void pad_zeros(std::vector<float> &audio, int num_samples);

// Validate audio duration is within [min_sec, max_sec]
bool validate_duration(const std::vector<float> &audio, int sample_rate,
                       float min_sec, float max_sec);

// Get audio duration in seconds
inline float get_duration(const std::vector<float> &audio, int sample_rate) {
  return static_cast<float>(audio.size()) / static_cast<float>(sample_rate);
}

// Save audio samples to WAV file (int16_t samples)
bool save_wav(const std::string &path, const std::vector<int16_t> &samples,
              int sample_rate, int channels = 1);

// Save audio samples to WAV file (float samples, will be converted to int16)
bool save_wav(const std::string &path, const std::vector<float> &samples,
              int sample_rate, int channels = 1);

} // namespace audio_io
