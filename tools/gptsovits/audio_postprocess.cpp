#include "audio_postprocess.h"

namespace audio_post {

std::pair<int, std::vector<int16_t>> AudioPostProcessor::process(
    std::vector<std::vector<std::vector<float>>> &audio, int sampling_rate,
    const std::vector<std::vector<int>> &batch_index_list, float speed_factor,
    bool split_bucket, float fragment_interval) {
  // Create silence padding (zero vector), length = sampling_rate * fragment_interval
  size_t zero_wav_len = static_cast<size_t>(sampling_rate * fragment_interval);
  std::vector<float> zero_wav(zero_wav_len, 0.0f);

  // Process each audio fragment in each batch
  for (size_t i = 0; i < audio.size(); ++i) {
    for (size_t j = 0; j < audio[i].size(); ++j) {
      auto &audio_fragment = audio[i][j];

      // Calculate maximum absolute value (prevent 16-bit clipping)
      float max_audio = 0.0f;
      for (const float &sample : audio_fragment) {
        float abs_val = std::abs(sample);
        if (abs_val > max_audio) {
          max_audio = abs_val;
        }
      }

      // Normalize if maximum value exceeds 1
      if (max_audio > 1.0f) {
        for (float &sample : audio_fragment) {
          sample /= max_audio;
        }
      }

      // Add silence padding at the end of audio fragment
      audio_fragment.insert(audio_fragment.end(), zero_wav.begin(),
                            zero_wav.end());
    }
  }

  // Process audio order based on split_bucket mode
  std::vector<std::vector<float>> ordered_audio;
  if (split_bucket) {
    // Restore original order
    ordered_audio = recovery_order(audio, batch_index_list);
  } else {
    // Flatten directly
    ordered_audio = flatten(audio);
  }

  // Calculate total sample count
  size_t total_samples = 0;
  for (const auto &fragment : ordered_audio) {
    total_samples += fragment.size();
  }

  // Allocate output buffer
  std::vector<int16_t> output(total_samples);

  // Concatenate all audio fragments and convert to int16
  // NOTE: numpy's astype(np.int16) wraps around for out-of-range values
  // instead of clipping. e.g., 32768.0 becomes -32768, -32769.0 becomes 32767
  // To maintain consistency with Python, we need to simulate this behavior
  size_t offset = 0;
  for (const auto &fragment : ordered_audio) {
    for (const float &sample : fragment) {
      // Multiply float by 32768
      float scaled = sample * 32768.0f;
      // Truncate to integer (consistent with Python's astype behavior)
      int32_t int_val = static_cast<int32_t>(scaled);
      // Convert to int16 (will wrap around automatically)
      output[offset++] = static_cast<int16_t>(int_val);
    }
  }

  return {sampling_rate, output};
}

std::vector<std::vector<float>> AudioPostProcessor::recovery_order(
    const std::vector<std::vector<std::vector<float>>> &audio,
    const std::vector<std::vector<int>> &batch_index_list) {
  // Calculate total data amount
  size_t total_length = 0;
  for (const auto &batch : batch_index_list) {
    total_length += batch.size();
  }

  // Initialize output array
  std::vector<std::vector<float>> result(total_length);

  // Restore order based on indices
  for (size_t i = 0; i < batch_index_list.size(); ++i) {
    const auto &index_list = batch_index_list[i];
    for (size_t j = 0; j < index_list.size(); ++j) {
      int original_index = index_list[j];
      if (original_index >= 0 &&
          static_cast<size_t>(original_index) < result.size()) {
        if (i < audio.size() && j < audio[i].size()) {
          result[original_index] = audio[i][j];
        }
      }
    }
  }

  return result;
}

std::vector<std::vector<float>> AudioPostProcessor::flatten(
    const std::vector<std::vector<std::vector<float>>> &audio) {
  std::vector<std::vector<float>> result;
  for (const auto &batch : audio) {
    for (const auto &fragment : batch) {
      result.push_back(fragment);
    }
  }
  return result;
}

} // namespace audio_post
