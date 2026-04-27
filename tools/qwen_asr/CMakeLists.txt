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

# =============================================================================
# OminiX-CUDA Phase 4.1 — native ASR scaffold on NVIDIA Blackwell.
#
# Builds a static library with the host-side mel spectrogram, the CUDA audio
# encoder scaffold, and the top-level AsrCudaEngine. Reuses
# tools/qwen_tts/native/talker_cuda_engine for the 28L Qwen3 text decoder,
# identical to the TTS talker.
#
# Phase 4.1 deliverable: scaffold + GGUF parse + smoke test.
#   Phase 4.2: audio encoder forward kernels.
#   Phase 4.3: split-prefill helper (prefix_text + audio_embeds + suffix_text).
#   Phase 4.4: end-to-end transcribe.
# =============================================================================
option(QWEN_ASR_CUDA "Build native CUDA ASR engine (Phase 4)" OFF)
if(QWEN_ASR_CUDA)
    if(NOT QWEN_TTS_CUDA)
        message(FATAL_ERROR
            "QWEN_ASR_CUDA=ON requires -DQWEN_TTS_CUDA=ON (the ASR text decoder "
            "reuses TalkerCudaEngine).")
    endif()

    # CUDA kernels static lib (audio-encoder-specific kernels).
    enable_language(CUDA)
    add_library(qwen_asr_cuda_kernels STATIC
        native/cuda_kernels/audio_encoder_kernels.cu
    )
    set_target_properties(qwen_asr_cuda_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON)
    target_include_directories(qwen_asr_cuda_kernels PUBLIC
        native native/cuda_kernels)

    # Static library: mel spectrogram (CPU) + audio encoder (CUDA) +
    # AsrCudaEngine (top-level orchestration).
    add_library(qwen_asr_cuda STATIC
        native/mel_spectrogram.cpp
        native/audio_encoder_cuda_engine.cpp
        native/asr_cuda_engine.cpp
        ${KISSFFT_SRC}
    )
    target_link_libraries     (qwen_asr_cuda PUBLIC  ggml qwen_common)
    target_link_libraries     (qwen_asr_cuda PRIVATE Threads::Threads
                                                      CUDA::cudart CUDA::cublas
                                                      qwen_tts_cuda_kernels
                                                      qwen_asr_cuda_kernels)
    target_include_directories(qwen_asr_cuda PUBLIC  native)
    target_include_directories(qwen_asr_cuda PRIVATE
        ../.. ../../vendor ${KISSFFT_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/tools/qwen_tts
        ${CMAKE_SOURCE_DIR}/tools/qwen_tts/native)
    target_compile_features   (qwen_asr_cuda PUBLIC  cxx_std_17)
    target_compile_definitions(qwen_asr_cuda PUBLIC  QWEN_ASR_HAS_CUDA=1
                                                      QWEN_TTS_HAS_CUDA=1)

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        target_compile_options(qwen_asr_cuda PRIVATE -march=armv8.2-a+fp16)
    endif()
    if (BUILD_SHARED_LIBS)
        set_target_properties(qwen_asr_cuda
            PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()

    # ---------------------------------------------------------------------
    # Phase 4.1 smoke harness — test_qwen_asr_cuda_init
    #
    # Init AsrCudaEngine from <audio_encoder_gguf> + <text_decoder_gguf>,
    # report counts and dimensions, exit 0.
    # ---------------------------------------------------------------------
    add_executable(test_qwen_asr_cuda_init
        native/test_qwen_asr_cuda_init.cpp
        ../qwen_tts/native/talker_cuda_engine.cpp
    )
    target_link_libraries     (test_qwen_asr_cuda_init PUBLIC  ggml qwen_common)
    target_link_libraries     (test_qwen_asr_cuda_init PRIVATE Threads::Threads
                                                                qwen_asr_cuda
                                                                CUDA::cudart
                                                                CUDA::cublas
                                                                qwen_tts_cuda_kernels)
    target_include_directories(test_qwen_asr_cuda_init PRIVATE
        ../.. ../../vendor native
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/tools/qwen_tts
        ${CMAKE_SOURCE_DIR}/tools/qwen_tts/native)
    target_compile_definitions(test_qwen_asr_cuda_init PRIVATE
                                QWEN_ASR_HAS_CUDA=1 QWEN_TTS_HAS_CUDA=1)
    target_compile_features   (test_qwen_asr_cuda_init PRIVATE cxx_std_17)
    if(OMINIX_CUDA_USE_CUDNN)
        target_link_libraries     (test_qwen_asr_cuda_init PRIVATE CUDA::cudnn)
        target_compile_definitions(test_qwen_asr_cuda_init PRIVATE
                                    OMINIX_CUDA_USE_CUDNN=1)
    endif()
    # ---------------------------------------------------------------------
    # Phase 4.2 smoke harness — test_audio_encoder_cuda
    #
    # Loads a WAV, computes mel on the CPU, calls AudioEncoderCudaEngine.encode,
    # prints output stats and verifies NaN/Inf-freeness.
    # ---------------------------------------------------------------------
    add_executable(test_audio_encoder_cuda
        native/test_audio_encoder_cuda.cpp
    )
    target_link_libraries     (test_audio_encoder_cuda PUBLIC  ggml qwen_common)
    target_link_libraries     (test_audio_encoder_cuda PRIVATE Threads::Threads
                                                                 qwen_asr_cuda
                                                                 qwen_asr_cuda_kernels
                                                                 CUDA::cudart
                                                                 CUDA::cublas
                                                                 qwen_tts_cuda_kernels)
    target_include_directories(test_audio_encoder_cuda PRIVATE
        ../.. ../../vendor native
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/tools)
    target_compile_definitions(test_audio_encoder_cuda PRIVATE
                                QWEN_ASR_HAS_CUDA=1)
    target_compile_features   (test_audio_encoder_cuda PRIVATE cxx_std_17)

    message(STATUS
        "qwen_asr: OminiX-CUDA Phase 4.1+4.2 ASR ENABLED "
        "(qwen_asr_cuda + test_qwen_asr_cuda_init + test_audio_encoder_cuda)")
endif()
