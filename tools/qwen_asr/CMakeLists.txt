# Qwen3-ASR Speech Recognition Implementation

find_package(Threads REQUIRED)

set(TARGET qwen_asr)

# Reuse shared components from qwen_tts
set(QWEN_TTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../qwen_tts)

set(SRC_FILES
    main.cpp
    qwen_asr.cpp
    audio_encoder.cpp
    mel_spectrogram.cpp
    # Shared components from qwen_tts
    ${QWEN_TTS_DIR}/bpe_tokenizer.cpp
    ${QWEN_TTS_DIR}/audio_io.cpp
    ${QWEN_TTS_DIR}/utils.cpp
    ${QWEN_TTS_DIR}/model_loader.cpp
    ${QWEN_TTS_DIR}/model_defs.cpp
    ${QWEN_TTS_DIR}/ctx_manager.cpp
    ${QWEN_TTS_DIR}/build_graph.cpp
    ${QWEN_TTS_DIR}/stft.cpp
)

# kissfft library for mel spectrogram
set(KISSFFT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/kissfft)
set(KISSFFT_SRC
    ${KISSFFT_DIR}/kiss_fft.c
    ${KISSFFT_DIR}/kiss_fftr.c
)
list(APPEND SRC_FILES ${KISSFFT_SRC})

add_executable(${TARGET} ${SRC_FILES})

target_link_libraries     (${TARGET} PUBLIC ggml llama)
target_link_libraries     (${TARGET} PRIVATE Threads::Threads)
target_include_directories(${TARGET} PUBLIC  . ${QWEN_TTS_DIR})
target_include_directories(${TARGET} PRIVATE ../.. ../../vendor ${KISSFFT_DIR})
target_compile_features   (${TARGET} PRIVATE cxx_std_17)

# Enable NEON F16 intrinsics for ARM
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(${TARGET} PRIVATE -march=armv8.2-a+fp16)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_SHARED LLAMA_BUILD)
endif()

install(TARGETS ${TARGET} RUNTIME)

# Test executable for step-by-step encoder verification
set(TEST_ENC_TARGET test_encoder_asr)
set(TEST_ENC_SRC
    test_encoder.cpp
    audio_encoder.cpp
    ${QWEN_TTS_DIR}/utils.cpp
    ${QWEN_TTS_DIR}/model_loader.cpp
    ${QWEN_TTS_DIR}/model_defs.cpp
    ${QWEN_TTS_DIR}/ctx_manager.cpp
    ${QWEN_TTS_DIR}/build_graph.cpp
)

add_executable(${TEST_ENC_TARGET} ${TEST_ENC_SRC})
target_link_libraries     (${TEST_ENC_TARGET} PUBLIC ggml)
target_link_libraries     (${TEST_ENC_TARGET} PRIVATE Threads::Threads)
target_include_directories(${TEST_ENC_TARGET} PUBLIC  . ${QWEN_TTS_DIR})
target_include_directories(${TEST_ENC_TARGET} PRIVATE ../.. ../../vendor)
target_compile_features   (${TEST_ENC_TARGET} PRIVATE cxx_std_17)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(${TEST_ENC_TARGET} PRIVATE -march=armv8.2-a+fp16)
endif()

# Minimal conv2d test
set(TEST_CONV_TARGET test_conv2d_minimal)
set(TEST_CONV_SRC
    test_conv2d_minimal.cpp
    ${QWEN_TTS_DIR}/model_loader.cpp
    ${QWEN_TTS_DIR}/utils.cpp
)
add_executable(${TEST_CONV_TARGET} ${TEST_CONV_SRC})
target_link_libraries     (${TEST_CONV_TARGET} PUBLIC ggml)
target_link_libraries     (${TEST_CONV_TARGET} PRIVATE Threads::Threads)
target_include_directories(${TEST_CONV_TARGET} PUBLIC  . ${QWEN_TTS_DIR})
target_include_directories(${TEST_CONV_TARGET} PRIVATE ../.. ../../vendor)
target_compile_features   (${TEST_CONV_TARGET} PRIVATE cxx_std_17)
