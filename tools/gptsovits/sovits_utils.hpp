#pragma once
#include "llama.h"
#include <vector>
#include <algorithm>
#include <numeric>
void f(float speed_factor, std::vector<llama_token> & tokens, std::vector<llama_token>& phones){
    /*
    phones = {51, 12, 62, 68,  1, 63, 22, 64, 42, 63, 55, 93, 92, 10, 65, 63, 55, 65,
         1, 64, 22, 75, 80, 88, 63, 58, 80, 92, 88,  0}
    */
    std::vector<int> upsample_rates = {10, 8, 2, 2, 2};
    // upsample_rate = math.prod(self.vits_model.upsample_rates)
    int upsample_rate = std::accumulate(upsample_rates.begin(), upsample_rates.end(), 1, std::multiplies<int>());
    int audio_frag_idx = tokens.size() * 2 * upsample_rate;

}