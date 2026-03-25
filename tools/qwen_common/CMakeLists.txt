add_library(qwen_common STATIC
    bpe_tokenizer.cpp
    audio_io.cpp
    utils.cpp
    model_loader.cpp
    ctx_manager.cpp
    build_graph.cpp
)
target_link_libraries(qwen_common PUBLIC ggml)
target_include_directories(qwen_common PUBLIC .)
target_include_directories(qwen_common PRIVATE ../.. ../../vendor)
target_compile_features(qwen_common PRIVATE cxx_std_17)
