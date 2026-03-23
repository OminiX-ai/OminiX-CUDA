#pragma once

/*
model_loader.h implements GGUF model file loading functionality. Key member attributes include ctx_gguf and ctx_meta.
It also provides value reading function interfaces.
ctx_meta ownership will be transferred during load_tensors operation, reducing ggml_context creation overhead and allowing ModelLoader to be released promptly.
*/
#include "ggml-cpp.h"
#include "utils.h"
#include <string>

class ModelLoader {
public:
  ModelLoader(const std::string &fname);
  void get_bool(const std::string &key, bool &output,
                bool required = true) const;
  void get_i32(const std::string &key, int &output, bool required = true) const;
  void get_u32(const std::string &key, int &output, bool required = true) const;

  void get_f32(const std::string &key, float &output,
               bool required = true) const;
  void get_string(const std::string &key, std::string &output,
                  bool required = true) const;
  void get_arr_int(const std::string &key, std::vector<int> &output,
                   bool required = true) const;

  // private:
  std::string fname_;
  size_t model_size_ = 0; // in bytes
  gguf_context_ptr ctx_gguf_ = nullptr;
  // Ownership will be transferred to ContextManager object later
  ggml_context_ptr ctx_meta_ = nullptr;
};
