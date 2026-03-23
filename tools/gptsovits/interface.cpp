#include "openvla.h"
#include "utils.h"
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

class OpenvlaPipeline {
public:
  OpenvlaPipeline(const std::string &dinov2_model_path,
                  const std::string &siglip_model_path,
                  const std::string &proj_model_path,
                  const std::string &llm_model_path,
                  const std::string &tokenizer_path = "",
                  const std::string &device_name = "CUDA0", int n_threads = 4,
                  int max_nodes = 2048, int ngl = 99, int n_ctx = 300) {
    ContextParams ctx_params = {.device_name = device_name,
                                .n_threads = n_threads,
                                .max_nodes = max_nodes};
    LlmParam llm_params = {.ngl = ngl,
                           .n_ctx = n_ctx,
                           .tokenizer_path = tokenizer_path,
                           .embeddings = false};
    openvla_ = std::make_unique<Openvla>(dinov2_model_path, siglip_model_path,
                                        proj_model_path, llm_model_path,
                                        ctx_params, llm_params);
  }
  py::array_t<float> run(const std::string &image_path,
                         const std::string &instruction) {
    std::vector<float> output;
    openvla_->run(image_path, instruction, output);
    ssize_t n = output.size();
    return py::array_t<float>(n, output.data());
  }

private:
  std::unique_ptr<Openvla> openvla_;
};

PYBIND11_MODULE(openvla, m) {
  m.doc() = "Python binding for OpenVLA C++ pipeline";

  py::class_<OpenvlaPipeline>(m, "OpenvlaPipeline")
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &, const std::string &, int, int, int,
                    int>(),
           py::arg("dinov2_model_path"), py::arg("siglip_model_path"),
           py::arg("proj_model_path"), py::arg("llm_model_path"),
           py::arg("tokenizer_path") = "", py::arg("device_name") = "CUDA0",
           py::arg("n_threads") = 4, py::arg("max_nodes") = 2048,
           py::arg("ngl") = 99, py::arg("n_ctx") = 300)
      .def("run", &OpenvlaPipeline::run, py::arg("image_path"),
           py::arg("instruction"),
           "Run OpenVLA on a given image and instruction");
}