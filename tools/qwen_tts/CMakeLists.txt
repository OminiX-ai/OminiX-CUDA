# Qwen3-TTS Voice Clone Implementation

find_package(Threads REQUIRED)
find_package(OpenMP)

set(TARGET qwen_tts)
set(SRC_FILES
    main.cpp
    qwen_tts.cpp
    bpe_tokenizer.cpp
    talker.cpp
    speaker_encoder.cpp
    speech_tokenizer_encoder.cpp
    speech_tokenizer_decoder.cpp
    audio_io.cpp
    utils.cpp
    model_loader.cpp
    model_defs.cpp
    ctx_manager.cpp
    build_graph.cpp
    stft.cpp
)

# kissfft library for STFT computation
set(KISSFFT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/kissfft)
set(KISSFFT_SRC
    ${KISSFFT_DIR}/kiss_fft.c
    ${KISSFFT_DIR}/kiss_fftr.c
)
list(APPEND SRC_FILES ${KISSFFT_SRC})

add_executable(${TARGET} ${SRC_FILES})

target_link_libraries     (${TARGET} PUBLIC ggml llama)
target_link_libraries     (${TARGET} PRIVATE Threads::Threads)
target_include_directories(${TARGET} PUBLIC  .)
target_include_directories(${TARGET} PRIVATE ../.. ../../vendor ${KISSFFT_DIR})
target_compile_features   (${TARGET} PRIVATE cxx_std_17)

# Enable NEON F16 intrinsics for ARM
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(${TARGET} PRIVATE -march=armv8.2-a+fp16)
endif()

# OpenMP for parallel CP matvec
if(OpenMP_CXX_FOUND)
    target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties     (${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_BUILD)
    target_compile_definitions(${TARGET} PUBLIC  LLAMA_SHARED)
endif()

set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME ${TARGET})
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
target_link_libraries  (${TARGET} PRIVATE common Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Test: Code Predictor verification
set(TEST_CP_SRC
    test_code_predictor.cpp
    talker.cpp
    model_loader.cpp
    model_defs.cpp
    ctx_manager.cpp
    build_graph.cpp
    utils.cpp
)
add_executable(test_code_predictor ${TEST_CP_SRC})
target_link_libraries     (test_code_predictor PUBLIC ggml llama)
target_link_libraries     (test_code_predictor PRIVATE Threads::Threads common)
target_include_directories(test_code_predictor PUBLIC  .)
target_include_directories(test_code_predictor PRIVATE ../.. ../../vendor)
target_compile_features   (test_code_predictor PRIVATE cxx_std_17)

# Test: Talker LLM integration
set(TEST_TALKER_SRC
    test_talker.cpp
    talker.cpp
    model_loader.cpp
    model_defs.cpp
    ctx_manager.cpp
    build_graph.cpp
    utils.cpp
)
add_executable(test_talker ${TEST_TALKER_SRC})
target_link_libraries     (test_talker PUBLIC ggml llama)
target_link_libraries     (test_talker PRIVATE Threads::Threads common)
target_include_directories(test_talker PUBLIC  .)
target_include_directories(test_talker PRIVATE ../.. ../../vendor)
target_compile_features   (test_talker PRIVATE cxx_std_17)

# Test: Speech Tokenizer Encoder
set(TEST_ENCODER_SRC
    test_encoder.cpp
    speech_tokenizer_encoder.cpp
    model_loader.cpp
    model_defs.cpp
    ctx_manager.cpp
    build_graph.cpp
    utils.cpp
)
add_executable(test_encoder ${TEST_ENCODER_SRC})
target_link_libraries     (test_encoder PUBLIC ggml llama)
target_link_libraries     (test_encoder PRIVATE Threads::Threads common)
target_include_directories(test_encoder PUBLIC  .)
target_include_directories(test_encoder PRIVATE ../.. ../../vendor)
target_compile_features   (test_encoder PRIVATE cxx_std_17)

# Test: CP autoregressive flow
set(TEST_CP_FLOW_SRC
    test_cp_flow.cpp
    talker.cpp
    model_loader.cpp
    model_defs.cpp
    ctx_manager.cpp
    build_graph.cpp
    utils.cpp
)
add_executable(test_cp_flow ${TEST_CP_FLOW_SRC})
target_link_libraries     (test_cp_flow PUBLIC ggml llama)
target_link_libraries     (test_cp_flow PRIVATE Threads::Threads common)
target_include_directories(test_cp_flow PUBLIC  .)
target_include_directories(test_cp_flow PRIVATE ../.. ../../vendor)
target_compile_features   (test_cp_flow PRIVATE cxx_std_17)

# Test: BPE Tokenizer
add_executable(test_tokenizer test_tokenizer.cpp bpe_tokenizer.cpp)
target_link_libraries     (test_tokenizer PRIVATE Threads::Threads)
target_include_directories(test_tokenizer PRIVATE . ../.. ../../vendor)
target_compile_features   (test_tokenizer PRIVATE cxx_std_17)
