#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace audio_post {

/**
 * @brief Audio post-processing class
 *
 * This class implements the same functionality as Python TTS.audio_postprocess:
 * 1. Normalize each audio fragment (prevent 16-bit clipping)
 * 2. Add silence interval at the end of each fragment
 * 3. Restore audio order after batch processing (if split_bucket is enabled)
 * 4. Concatenate all fragments and convert to int16 format
 */
class AudioPostProcessor {
public:
  /**
   * @brief Audio post-processing
   *
   * @param audio Input audio data, 2D structure: audio[batch_idx][fragment_idx] = float vector
   * @param sampling_rate Sampling rate
   * @param batch_index_list Batch index list for restoring original order (optional)
   * @param speed_factor Speed factor (reserved parameter, currently unused)
   * @param split_bucket Whether to enable bucket splitting mode
   * @param fragment_interval Fragment interval time (seconds)
   * @return std::pair<int, std::vector<int16_t>> Sampling rate and processed audio data
   */
  static std::pair<int, std::vector<int16_t>>
  process(std::vector<std::vector<std::vector<float>>> &audio,
          int sampling_rate,
          const std::vector<std::vector<int>> &batch_index_list,
          float speed_factor, bool split_bucket, float fragment_interval);

private:
  /**
   * @brief Restore original order of audio fragments
   *
   * @param audio Audio fragments grouped by batch
   * @param batch_index_list Batch index list
   * @return std::vector<std::vector<float>> Audio fragment list with restored order
   */
  static std::vector<std::vector<float>>
  recovery_order(const std::vector<std::vector<std::vector<float>>> &audio,
                 const std::vector<std::vector<int>> &batch_index_list);

  /**
   * @brief Flatten 2D audio array to 1D
   *
   * @param audio 2D audio array
   * @return std::vector<std::vector<float>> Flattened audio fragment list
   */
  static std::vector<std::vector<float>>
  flatten(const std::vector<std::vector<std::vector<float>>> &audio);
};

} // namespace audio_post
