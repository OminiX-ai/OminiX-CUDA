# Qwen3-ASR Speech Recognition Implementation

find_package(Threads REQUIRED)

set(TARGET qwen_asr)

set(SRC_FILES
    main.cpp
    qwen_asr.cpp
    audio_encoder.cpp
    mel_spectrogram.cpp
)

# kissfft library for mel spectrogram
set(KISSFFT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/kissfft)
set(KISSFFT_SRC
    ${KISSFFT_DIR}/kiss_fft.c
    ${KISSFFT_DIR}/kiss_fftr.c
)
list(APPEND SRC_FILES ${KISSFFT_SRC})

add_executable(${TARGET} ${SRC_FILES})

target_link_libraries     (${TARGET} PUBLIC ggml llama qwen_common)
target_link_libraries     (${TARGET} PRIVATE Threads::Threads)
target_include_directories(${TARGET} PUBLIC  .)
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
)

add_executable(${TEST_ENC_TARGET} ${TEST_ENC_SRC})
target_link_libraries     (${TEST_ENC_TARGET} PUBLIC ggml qwen_common)
target_link_libraries     (${TEST_ENC_TARGET} PRIVATE Threads::Threads)
target_include_directories(${TEST_ENC_TARGET} PUBLIC  .)
target_include_directories(${TEST_ENC_TARGET} PRIVATE ../.. ../../vendor)
target_compile_features   (${TEST_ENC_TARGET} PRIVATE cxx_std_17)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(${TEST_ENC_TARGET} PRIVATE -march=armv8.2-a+fp16)
endif()

# Minimal conv2d test
set(TEST_CONV_TARGET test_conv2d_minimal)
set(TEST_CONV_SRC
    test_conv2d_minimal.cpp
)
add_executable(${TEST_CONV_TARGET} ${TEST_CONV_SRC})
target_link_libraries     (${TEST_CONV_TARGET} PUBLIC ggml qwen_common)
target_link_libraries     (${TEST_CONV_TARGET} PRIVATE Threads::Threads)
target_include_directories(${TEST_CONV_TARGET} PUBLIC  .)
target_include_directories(${TEST_CONV_TARGET} PRIVATE ../.. ../../vendor)
target_compile_features   (${TEST_CONV_TARGET} PRIVATE cxx_std_17)

# =============================================================================
# qwen_asr_native — A1b-v2: native text-decoder engine on Ascend NPU.
#
# Reuses TalkerCannEngine (tools/qwen_tts) verbatim for the 28L transformer
# body and adds a host-side embedding + NPU lm_head side-car. Gated on the
# same condition that enables the CP-CANN engine in qwen_tts — if we're on
# an Ascend host with acl.h available. Outside Ascend this target is skipped.
# =============================================================================
if(FALSE)
    set(NATIVE_TARGET qwen_asr_native)
    set(NATIVE_SRC
        native/main_native.cpp
        native/asr_text_decoder_engine.cpp
        # Shared with qwen_asr:
        audio_encoder.cpp
        mel_spectrogram.cpp
        # Reused verbatim from qwen_tts:
        ../qwen_tts/talker_cann_engine.cpp
        ../qwen_tts/cp_cann_symbols.cpp
        ${KISSFFT_SRC}
    )
    add_executable(${NATIVE_TARGET} ${NATIVE_SRC})

    target_link_libraries     (${NATIVE_TARGET} PUBLIC ggml qwen_common)
    target_link_libraries     (${NATIVE_TARGET} PRIVATE Threads::Threads
                                                          ${CMAKE_DL_LIBS})
    target_include_directories(${NATIVE_TARGET} PUBLIC  .)
    target_include_directories(${NATIVE_TARGET} PRIVATE
        ../..
        ../../vendor
        ${KISSFFT_DIR}
        ${CMAKE_SOURCE_DIR}/include   # llama.h (talker.h includes it as a type decl)
        $ENV{ASCEND_TOOLKIT_HOME}/include
        $ENV{ASCEND_TOOLKIT_HOME}/aarch64-linux/include)
    target_compile_features   (${NATIVE_TARGET} PRIVATE cxx_std_17)
    target_compile_definitions(${NATIVE_TARGET} PRIVATE QWEN_TTS_HAS_CP_CANN=1)

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        target_compile_options(${NATIVE_TARGET} PRIVATE -march=armv8.2-a+fp16)
    endif()

    if (BUILD_SHARED_LIBS)
        set_target_properties(${NATIVE_TARGET}
            PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()

    install(TARGETS ${NATIVE_TARGET} RUNTIME)
    message(STATUS "qwen_asr: native engine ENABLED (qwen_asr_native)")
else()
    message(STATUS "qwen_asr: native engine disabled (ominix-cuda Phase 4 CUDA port pending)")
endif()
