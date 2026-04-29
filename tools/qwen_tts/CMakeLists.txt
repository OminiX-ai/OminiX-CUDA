# Qwen3-TTS Voice Clone Implementation

find_package(Threads REQUIRED)
find_package(OpenMP)

set(TARGET qwen_tts)
set(SRC_FILES
    main.cpp
    qwen_tts.cpp
    talker.cpp
    tts_transformer.cpp
    speaker_encoder.cpp
    speech_tokenizer_encoder.cpp
    speech_tokenizer_decoder.cpp
    model_defs.cpp
    stft.cpp
)

# Native CANN engines were stripped for ominix-cuda; Phase 2 will add the
# CUDA-native talker path.
option(QWEN_TTS_CP_CANN "Build native CP CANN engine (unavailable in ominix-cuda)" OFF)
if(QWEN_TTS_CP_CANN)
    message(FATAL_ERROR "QWEN_TTS_CP_CANN is not available in ominix-cuda")
endif()

# -----------------------------------------------------------------------------
# Path C W4.1: AscendC custom kernels.
#
# The fused_attn_sublayer kernel collapses the CP transformer's attention
# sublayer (12 aclnn ops) into a single AscendC dispatch. Built into a
# STATIC library via the ascendc_library() cmake macro from the CANN
# toolkit. The library is named `qwen_tts_ascendc_kernels`; downstream
# targets (qwen_tts executable, qwen_tts_api shared lib, and the offline
# 4.1.2 diff harness) link it as an internal dep.
#
# Gated on QWEN_TTS_CP_CANN — the kernels are only useful when the CP
# CANN engine is also built. Default off when ASCEND_TOOLKIT_HOME is
# absent (macOS / non-Ascend hosts); default on when we're on an Ascend
# host. Source file lives at tools/qwen_tts/ascendc/.
# -----------------------------------------------------------------------------
option(QWEN_TTS_ASCENDC "Build AscendC custom kernels (W4.1)" OFF)
if(QWEN_TTS_ASCENDC)
    message(FATAL_ERROR "QWEN_TTS_ASCENDC is not available in ominix-cuda")
endif()

# Native CUDA Talker engine (Phase 2 of the OminiX-CUDA contract). Default
# OFF on hosts without CUDA. Mirror of QWEN_TTS_CP_CANN on the Ascend side:
# when ON, the executable + shared lib pull in the new
# tools/qwen_tts/native/talker_cuda_engine.* sources and link cuBLAS (and
# cuDNN if the user opts into FMHA helpers via OMINIX_CUDA_USE_CUDNN).
option(QWEN_TTS_CUDA "Build native CUDA Talker engine (Phase 2)" OFF)
if(QWEN_TTS_CUDA)
    if(NOT GGML_CUDA)
        message(WARNING "QWEN_TTS_CUDA=ON requires -DGGML_CUDA=ON for matching "
                        "ggml-cuda backend; building anyway since the native "
                        "engine dispatches cuBLAS directly")
    endif()
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        # GB10 is sm_121 (Blackwell). Allow override from the cmake CLI.
        set(CMAKE_CUDA_ARCHITECTURES 121)
    endif()
    message(STATUS "qwen_tts: native CUDA Talker engine ENABLED "
                   "(CUDA ${CUDAToolkit_VERSION}, sm=${CMAKE_CUDA_ARCHITECTURES})")

    # Static library bundling the custom CUDA kernels (RmsNorm, RoPE,
    # SwiGLU, GQA attention, F32<->F16 casts, F16 add). Linked into the
    # qwen_tts executable, the qwen_tts_api shared lib, and the Phase 2.x
    # smoke harnesses.
    add_library(qwen_tts_cuda_kernels STATIC
        native/cuda_kernels/elementwise.cu
        native/cuda_kernels/rmsnorm.cu
        native/cuda_kernels/rope_neox.cu
        native/cuda_kernels/swiglu.cu
        native/cuda_kernels/attn_gqa.cu
        native/cuda_kernels/sampler.cu
        native/cuda_kernels/embedding_lookup.cu
    )
    target_include_directories(qwen_tts_cuda_kernels PUBLIC native)
    set_target_properties(qwen_tts_cuda_kernels PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION OFF
    )
    target_compile_features(qwen_tts_cuda_kernels PUBLIC cxx_std_17)
endif()

# llama.cpp fallback Talker path: optional, default OFF. When OFF, the
# `--llama_fallback` runtime flag prints a clear error and exits; the
# native TalkerCannEngine is the only backbone path. When ON, llama.cpp
# is linked and `--llama_fallback` runs the original hybrid code path.
option(QWEN_TTS_LLAMA "Link llama.cpp fallback into qwen_tts" OFF)
if(QWEN_TTS_LLAMA)
    message(STATUS "qwen_tts: llama.cpp fallback ENABLED (QWEN_TTS_LLAMA=ON)")
else()
    message(STATUS "qwen_tts: llama.cpp fallback DISABLED "
                   "(native-only; enable with -DQWEN_TTS_LLAMA=ON)")
endif()

# kissfft library for STFT computation
set(KISSFFT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/kissfft)
set(KISSFFT_SRC
    ${KISSFFT_DIR}/kiss_fft.c
    ${KISSFFT_DIR}/kiss_fftr.c
)
list(APPEND SRC_FILES ${KISSFFT_SRC})

# Native CUDA engine sources (Phase 2 of OminiX-CUDA contract).
if(QWEN_TTS_CUDA)
    list(APPEND SRC_FILES
        native/talker_cuda_engine.cpp
    )
endif()

add_executable(${TARGET} ${SRC_FILES})

# Base deps. llama is only linked when QWEN_TTS_LLAMA=ON. The `llama.h`
# header is still included by talker.h for type declarations (llama_batch
# field, etc.) so we keep the include path visible in both configs; the
# actual library is skipped in the OFF case.
target_link_libraries     (${TARGET} PUBLIC ggml qwen_common)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/include)
if(QWEN_TTS_LLAMA)
    target_link_libraries     (${TARGET} PUBLIC llama)
    target_compile_definitions(${TARGET} PRIVATE QWEN_TTS_LLAMA=1)
endif()

# Native CP CANN engine: implemented via dlopen/dlsym at runtime to sidestep
# CANN stub-lib circular deps. We only need libdl + the toolkit headers.
target_link_libraries     (${TARGET} PRIVATE Threads::Threads)
if(QWEN_TTS_CP_CANN)
    target_link_libraries(${TARGET} PRIVATE ${CMAKE_DL_LIBS})
    target_compile_definitions(${TARGET} PRIVATE QWEN_TTS_HAS_CP_CANN=1)
    target_include_directories(${TARGET} PRIVATE
        $ENV{ASCEND_TOOLKIT_HOME}/aarch64-linux/include)
endif()
if(QWEN_TTS_ASCENDC)
    # Static link with the AscendC library + the auto-generated host stub
    # header (exposes `aclrtlaunch_fused_attn_sublayer`). The runtime
    # static libs come in via the ascendc.cmake's add_link_deps.
    target_link_libraries(${TARGET} PRIVATE qwen_tts_ascendc_kernels)
    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_BINARY_DIR}/auto_gen/qwen_tts_ascendc_kernels
        ${CMAKE_BINARY_DIR}/include/qwen_tts_ascendc_kernels)
    target_compile_definitions(${TARGET} PRIVATE QWEN_TTS_HAS_ASCENDC=1)
endif()
if(QWEN_TTS_CUDA)
    target_link_libraries     (${TARGET} PRIVATE
        CUDA::cudart CUDA::cublas CUDA::cublasLt qwen_tts_cuda_kernels)
    target_compile_definitions(${TARGET} PRIVATE QWEN_TTS_HAS_CUDA=1)
    target_include_directories(${TARGET} PRIVATE native)
    if(OMINIX_CUDA_USE_CUDNN)
        target_link_libraries     (${TARGET} PRIVATE CUDA::cudnn)
        target_compile_definitions(${TARGET} PRIVATE OMINIX_CUDA_USE_CUDNN=1)
    endif()
endif()
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
# `common` is llama.cpp's utility lib (PUBLIC-links libllama). Only needed
# when we're actually on the llama.cpp path. qwen_tts sources don't use it.
if(QWEN_TTS_LLAMA)
    target_link_libraries  (${TARGET} PRIVATE common)
endif()
target_link_libraries  (${TARGET} PRIVATE Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# =============================================================================
# qwen_tts_api — Shared library exposing the C ABI in qwen_tts_api.h.
#
# Built for OminiX-API FFI (Ascend API Bridge Contract §5 B1). Links the SAME
# internals as the qwen_tts executable (BPETokenizer, TalkerLLM, SpeechTokenizer
# encoder/decoder, SpeakerEncoder, ggml, ggml-cann, kissfft) but exposes ONLY
# the 14 qwen_tts_* C symbols via a linker version script to prevent ggml /
# ggml-cann symbol pollution in downstream binaries (risk register §7).
#
# Does NOT include main.cpp / qwen_tts.cpp (CLI entry points) or the llama.cpp
# fallback path — the library is a pure compute surface. The native CP-CANN
# Talker engine (cp_cann_engine.cpp, cp_cann_symbols.cpp, talker_cann_engine.cpp)
# is included when QWEN_TTS_CP_CANN is ON so Ascend clients get the same
# native-Talker path the executable uses.
# =============================================================================
set(API_TARGET qwen_tts_api)
set(API_SRC_FILES
    qwen_tts_api.cpp
    qwen_tts.cpp                     # B5: QwenTTS::generate* used by qwen_tts_synthesize
    talker.cpp
    tts_transformer.cpp
    speaker_encoder.cpp
    speech_tokenizer_encoder.cpp
    speech_tokenizer_decoder.cpp
    model_defs.cpp
    stft.cpp
    ${KISSFFT_SRC}
)
if(QWEN_TTS_CP_CANN)
    list(APPEND API_SRC_FILES
        cp_cann_engine.cpp
        cp_cann_symbols.cpp
        talker_cann_engine.cpp
    )
endif()
if(QWEN_TTS_CUDA)
    list(APPEND API_SRC_FILES
        native/talker_cuda_engine.cpp
    )
endif()

add_library(${API_TARGET} SHARED ${API_SRC_FILES})

# qwen_common is a STATIC dep and is linked into a SHARED lib here, so its
# object files must be PIC. qwen_common's own CMakeLists doesn't set PIC
# (its only other consumer is the qwen_tts executable, which doesn't need
# it). We flip the flag on the existing target from here — this touches
# build flags, not source code in tools/qwen_common/.
set_property(TARGET qwen_common PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries     (${API_TARGET} PUBLIC ggml qwen_common)
target_link_libraries     (${API_TARGET} PRIVATE Threads::Threads)
target_include_directories(${API_TARGET} PUBLIC  .)
target_include_directories(${API_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/include   # llama.h visible for talker.h (fallback types)
    ../.. ../../vendor ${KISSFFT_DIR})
target_compile_features   (${API_TARGET} PRIVATE cxx_std_17)
set_target_properties     (${API_TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(QWEN_TTS_LLAMA)
    target_link_libraries     (${API_TARGET} PUBLIC llama)
    target_compile_definitions(${API_TARGET} PRIVATE QWEN_TTS_LLAMA=1)
endif()

if(QWEN_TTS_CP_CANN)
    target_link_libraries(${API_TARGET} PRIVATE ${CMAKE_DL_LIBS})
    target_compile_definitions(${API_TARGET} PRIVATE QWEN_TTS_HAS_CP_CANN=1)
    target_include_directories(${API_TARGET} PRIVATE
        $ENV{ASCEND_TOOLKIT_HOME}/aarch64-linux/include)
endif()
if(QWEN_TTS_ASCENDC)
    target_link_libraries(${API_TARGET} PRIVATE qwen_tts_ascendc_kernels)
    target_include_directories(${API_TARGET} PRIVATE
        ${CMAKE_BINARY_DIR}/auto_gen/qwen_tts_ascendc_kernels
        ${CMAKE_BINARY_DIR}/include/qwen_tts_ascendc_kernels)
    target_compile_definitions(${API_TARGET} PRIVATE QWEN_TTS_HAS_ASCENDC=1)
endif()
if(QWEN_TTS_CUDA)
    target_link_libraries     (${API_TARGET} PRIVATE
        CUDA::cudart CUDA::cublas CUDA::cublasLt qwen_tts_cuda_kernels)
    target_compile_definitions(${API_TARGET} PRIVATE QWEN_TTS_HAS_CUDA=1)
    target_include_directories(${API_TARGET} PRIVATE native)
    if(OMINIX_CUDA_USE_CUDNN)
        target_link_libraries     (${API_TARGET} PRIVATE CUDA::cudnn)
        target_compile_definitions(${API_TARGET} PRIVATE OMINIX_CUDA_USE_CUDNN=1)
    endif()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(${API_TARGET} PRIVATE -march=armv8.2-a+fp16)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(${API_TARGET} PRIVATE OpenMP::OpenMP_CXX)
endif()

# SONAME versioning: libqwen_tts_api.so.1.1.0 with .so.1 and .so symlinks.
# Bumped MINOR (1.0.0 -> 1.1.0) for B5's additive qwen_tts_synthesize +
# qwen_tts_pcm_free exports. SOVERSION stays at 1 — existing consumers
# linked against the 14-symbol ABI resolve cleanly against this build.
set_target_properties(${API_TARGET} PROPERTIES
    OUTPUT_NAME qwen_tts_api
    VERSION     1.1.0
    SOVERSION   1
)

# Hide every symbol that isn't in the version script. Prevents ggml/ggml-cann
# internals from leaking into consumer binaries.
set(API_VERSION_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/qwen_tts_api.version")
if(UNIX AND NOT APPLE AND EXISTS "${API_VERSION_SCRIPT}")
    target_link_options(${API_TARGET} PRIVATE
        "-Wl,--version-script=${API_VERSION_SCRIPT}"
        "-Wl,--no-undefined")
    set_property(TARGET ${API_TARGET} APPEND PROPERTY
        LINK_DEPENDS "${API_VERSION_SCRIPT}")
    # NOTE: We deliberately do NOT use -fvisibility=hidden at compile time.
    # The version script's `local: *;` already suppresses every non-API symbol
    # at link time, including symbols from static-linked qwen_common / kissfft.
    # Adding -fvisibility=hidden would hide the extern "C" qwen_tts_* symbols
    # as well (they have no __attribute__((visibility("default"))) annotation),
    # leaving nothing for the version script to export.
endif()

# Install library to lib/, public header to include/.
install(TARGETS ${API_TARGET}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)
install(FILES qwen_tts_api.h DESTINATION include)

# Test: Code Predictor verification
set(TEST_CP_SRC
    test_code_predictor.cpp
    talker.cpp
    tts_transformer.cpp
    model_defs.cpp
)
add_executable(test_code_predictor ${TEST_CP_SRC})
target_link_libraries     (test_code_predictor PUBLIC ggml llama qwen_common)
target_link_libraries     (test_code_predictor PRIVATE Threads::Threads common)
target_include_directories(test_code_predictor PUBLIC  .)
target_include_directories(test_code_predictor PRIVATE ../.. ../../vendor)
target_compile_definitions(test_code_predictor PRIVATE QWEN_TTS_LLAMA=1)
target_compile_features   (test_code_predictor PRIVATE cxx_std_17)

# Test: Talker LLM integration
set(TEST_TALKER_SRC
    test_talker.cpp
    talker.cpp
    tts_transformer.cpp
    model_defs.cpp
)
add_executable(test_talker ${TEST_TALKER_SRC})
target_link_libraries     (test_talker PUBLIC ggml llama qwen_common)
target_link_libraries     (test_talker PRIVATE Threads::Threads common)
target_include_directories(test_talker PUBLIC  .)
target_include_directories(test_talker PRIVATE ../.. ../../vendor)
target_compile_definitions(test_talker PRIVATE QWEN_TTS_LLAMA=1)
target_compile_features   (test_talker PRIVATE cxx_std_17)

# Test: Speech Tokenizer Encoder
set(TEST_ENCODER_SRC
    test_encoder.cpp
    speech_tokenizer_encoder.cpp
    model_defs.cpp
)
add_executable(test_encoder ${TEST_ENCODER_SRC})
target_link_libraries     (test_encoder PUBLIC ggml llama qwen_common)
target_link_libraries     (test_encoder PRIVATE Threads::Threads common)
target_include_directories(test_encoder PUBLIC  .)
target_include_directories(test_encoder PRIVATE ../.. ../../vendor)
target_compile_features   (test_encoder PRIVATE cxx_std_17)

# Test: CP autoregressive flow
set(TEST_CP_FLOW_SRC
    test_cp_flow.cpp
    talker.cpp
    tts_transformer.cpp
    model_defs.cpp
)
add_executable(test_cp_flow ${TEST_CP_FLOW_SRC})
target_link_libraries     (test_cp_flow PUBLIC ggml llama qwen_common)
target_link_libraries     (test_cp_flow PRIVATE Threads::Threads common)
target_include_directories(test_cp_flow PUBLIC  .)
target_include_directories(test_cp_flow PRIVATE ../.. ../../vendor)
target_compile_definitions(test_cp_flow PRIVATE QWEN_TTS_LLAMA=1)
target_compile_features   (test_cp_flow PRIVATE cxx_std_17)

# Test: BPE Tokenizer
add_executable(test_tokenizer test_tokenizer.cpp)
target_link_libraries     (test_tokenizer PRIVATE Threads::Threads qwen_common)
target_include_directories(test_tokenizer PRIVATE . ../.. ../../vendor)
target_compile_features   (test_tokenizer PRIVATE cxx_std_17)

# Test: native TalkerCannEngine (Ascend only). Only built when the CANN
# engine source is present — it depends on cp_cann_symbols + the aclnn
# headers. The test pulls in the engine and its symbol loader directly; it
# does NOT link talker.cpp (no need for the llama.cpp Talker path).
if(QWEN_TTS_CP_CANN)
    add_executable(test_talker_native
        test_talker_native.cpp
        talker_cann_engine.cpp
        cp_cann_symbols.cpp
        model_defs.cpp
    )
    target_link_libraries     (test_talker_native PUBLIC ggml llama qwen_common)
    target_link_libraries     (test_talker_native PRIVATE Threads::Threads common
                                                           ${CMAKE_DL_LIBS})
    target_include_directories(test_talker_native PUBLIC  .)
    target_include_directories(test_talker_native PRIVATE ../.. ../../vendor
        $ENV{ASCEND_TOOLKIT_HOME}/aarch64-linux/include
        $ENV{ASCEND_TOOLKIT_HOME}/include)
    target_compile_definitions(test_talker_native PRIVATE QWEN_TTS_HAS_CP_CANN=1)
    target_compile_features   (test_talker_native PRIVATE cxx_std_17)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        target_compile_options(test_talker_native PRIVATE -march=armv8.2-a+fp16)
    endif()

    # Diagnostic harness: batched vs iterative forward_prefill cos-sim on
    # synthetic scales + real-embedding-dump replay (see Track A §8 entry
    # in NATIVE_TTS_CONTRACT.md for the use case).
    add_executable(test_prefill_diff
        test_prefill_diff.cpp
        talker_cann_engine.cpp
        cp_cann_symbols.cpp
        model_defs.cpp
    )
    target_link_libraries     (test_prefill_diff PUBLIC ggml llama qwen_common)
    target_link_libraries     (test_prefill_diff PRIVATE Threads::Threads common
                                                           ${CMAKE_DL_LIBS})
    target_include_directories(test_prefill_diff PUBLIC  .)
    target_include_directories(test_prefill_diff PRIVATE ../.. ../../vendor
        $ENV{ASCEND_TOOLKIT_HOME}/aarch64-linux/include
        $ENV{ASCEND_TOOLKIT_HOME}/include)
    target_compile_definitions(test_prefill_diff PRIVATE QWEN_TTS_HAS_CP_CANN=1)
    target_compile_features   (test_prefill_diff PRIVATE cxx_std_17)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        target_compile_options(test_prefill_diff PRIVATE -march=armv8.2-a+fp16)
    endif()

    # W4.1.2 offline-diff harness: dispatch the fused_attn_sublayer AscendC
    # kernel on a synthetic fixture and compare its output against the same
    # math run in host-F32. Gate: max-abs-diff ≤ 1e-2.
    if(QWEN_TTS_ASCENDC)
        add_executable(test_fused_attn_diff
            ascendc/test_fused_attn_diff.cpp
            cp_cann_symbols.cpp
        )
        target_link_libraries     (test_fused_attn_diff PUBLIC ggml llama qwen_common)
        target_link_libraries     (test_fused_attn_diff PRIVATE Threads::Threads common
                                                                 ${CMAKE_DL_LIBS}
                                                                 qwen_tts_ascendc_kernels)
        target_include_directories(test_fused_attn_diff PUBLIC  .)
        target_include_directories(test_fused_attn_diff PRIVATE ../.. ../../vendor
            $ENV{ASCEND_TOOLKIT_HOME}/aarch64-linux/include
            $ENV{ASCEND_TOOLKIT_HOME}/include
            ${CMAKE_BINARY_DIR}/include/qwen_tts_ascendc_kernels)
        target_compile_definitions(test_fused_attn_diff PRIVATE QWEN_TTS_HAS_CP_CANN=1
                                                                 QWEN_TTS_HAS_ASCENDC=1)
        target_compile_features   (test_fused_attn_diff PRIVATE cxx_std_17)
        # The ascendc_library() macro produces PIC object files; any target
        # that links it must also be PIC.
        set_target_properties     (test_fused_attn_diff PROPERTIES
                                    POSITION_INDEPENDENT_CODE ON)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            target_compile_options(test_fused_attn_diff PRIVATE -march=armv8.2-a+fp16)
        endif()
    endif()

    # Diagnostic: per-row aclnnMm(W, X_col) vs batched aclnnMm(X, W^T strided).
    # Verifies the strided-weight view pattern used in forward_prefill matches
    # the contiguous-weight pattern used in forward_decode.
    add_executable(test_mm_diff
        test_mm_diff.cpp
        cp_cann_symbols.cpp
    )
    target_link_libraries     (test_mm_diff PUBLIC ggml llama qwen_common)
    target_link_libraries     (test_mm_diff PRIVATE Threads::Threads common
                                                           ${CMAKE_DL_LIBS})
    target_include_directories(test_mm_diff PUBLIC  .)
    target_include_directories(test_mm_diff PRIVATE ../.. ../../vendor
        $ENV{ASCEND_TOOLKIT_HOME}/aarch64-linux/include
        $ENV{ASCEND_TOOLKIT_HOME}/include)
    target_compile_definitions(test_mm_diff PRIVATE QWEN_TTS_HAS_CP_CANN=1)
    target_compile_features   (test_mm_diff PRIVATE cxx_std_17)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        target_compile_options(test_mm_diff PRIVATE -march=armv8.2-a+fp16)
    endif()

    # -------------------------------------------------------------------------
    # G1 HARD GATE feasibility probe: test_aclgraph_smoke.
    #
    # Answers the single open question from G0: does
    # `aclmdlRICaptureTaskUpdateBegin/End` accept rebinding of RoPE slice
    # ptrs / KV-cache write offset / FIAv2 `seq_len` on an already-captured
    # `aclOpExecutor`? See ACLGRAPH_CONTRACT.md §5 G1 for the gate logic.
    #
    # Gated on QWEN_TTS_ACLGRAPH_SMOKE=ON (default OFF; only built when the
    # G1 agent is running). No behavioural impact on the main qwen_tts
    # binary — this is a standalone harness.
    # -------------------------------------------------------------------------
    option(QWEN_TTS_ACLGRAPH_SMOKE "Build G1 aclGraph capture/replay smoke test" OFF)
    if(QWEN_TTS_ACLGRAPH_SMOKE)
        add_executable(test_aclgraph_smoke
            test_aclgraph_smoke.cpp
            cp_cann_symbols.cpp
        )
        target_link_libraries     (test_aclgraph_smoke PUBLIC ggml llama qwen_common)
        target_link_libraries     (test_aclgraph_smoke PRIVATE Threads::Threads common
                                                                 ${CMAKE_DL_LIBS})
        target_include_directories(test_aclgraph_smoke PUBLIC  .)
        target_include_directories(test_aclgraph_smoke PRIVATE ../.. ../../vendor
            $ENV{ASCEND_TOOLKIT_HOME}/aarch64-linux/include
            $ENV{ASCEND_TOOLKIT_HOME}/include)
        target_compile_definitions(test_aclgraph_smoke PRIVATE QWEN_TTS_HAS_CP_CANN=1
                                                                TALKER_CP_ACLGRAPH_SMOKE=1)
        target_compile_features   (test_aclgraph_smoke PRIVATE cxx_std_17)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            target_compile_options(test_aclgraph_smoke PRIVATE -march=armv8.2-a+fp16)
        endif()
        message(STATUS "qwen_tts: G1 aclGraph smoke harness ENABLED")
    endif()
endif()

# -------------------------------------------------------------------------
# OminiX-CUDA Phase 2.1 smoke harness — test_talker_cuda_init
#
# Builds a tiny standalone executable that constructs `TalkerCudaEngine`,
# pulls TalkerConfig defaults, calls init_from_gguf() against a path passed
# on argv[1] (or `--noweights` to skip the file open and exercise scratch
# alloc + RoPE table build alone), and prints a pass/fail line.
#
# Phase 2.1 deliverable: this binary builds + runs on GB10 #1, prints
# scaffold-init OK with the configured 28/16/8/2048 dims and exits 0.
# -------------------------------------------------------------------------
if(QWEN_TTS_CUDA)
    add_executable(test_talker_cuda_init
        native/test_talker_cuda_init.cpp
        native/talker_cuda_engine.cpp
    )
    target_link_libraries     (test_talker_cuda_init PUBLIC  ggml qwen_common)
    target_link_libraries     (test_talker_cuda_init PRIVATE Threads::Threads
                                                              CUDA::cudart CUDA::cublas CUDA::cublasLt
                                                              qwen_tts_cuda_kernels)
    target_include_directories(test_talker_cuda_init PUBLIC  .)
    target_include_directories(test_talker_cuda_init PRIVATE
        ../.. ../../vendor native
        ${CMAKE_SOURCE_DIR}/include)   # talker.h includes <llama.h>
    target_compile_definitions(test_talker_cuda_init PRIVATE QWEN_TTS_HAS_CUDA=1)
    target_compile_features   (test_talker_cuda_init PRIVATE cxx_std_17)
    if(OMINIX_CUDA_USE_CUDNN)
        target_link_libraries     (test_talker_cuda_init PRIVATE CUDA::cudnn)
        target_compile_definitions(test_talker_cuda_init PRIVATE OMINIX_CUDA_USE_CUDNN=1)
    endif()
    message(STATUS "qwen_tts: OminiX-CUDA Phase 2.1 smoke harness ENABLED")

    # ---------------------------------------------------------------------
    # Phase 2.2 smoke harness — test_talker_cuda_decode
    #
    # Drives the full forward_decode body for 10 sequential positions.
    # Asserts no NaN/inf and reports avg wall-clock per step. Same GGUF
    # path as Phase 2.1; reuses the kernel static lib.
    # ---------------------------------------------------------------------
    add_executable(test_talker_cuda_decode
        native/test_talker_cuda_decode.cpp
        native/talker_cuda_engine.cpp
    )
    target_link_libraries     (test_talker_cuda_decode PUBLIC  ggml qwen_common)
    target_link_libraries     (test_talker_cuda_decode PRIVATE Threads::Threads
                                                                 CUDA::cudart CUDA::cublas CUDA::cublasLt
                                                                 qwen_tts_cuda_kernels)
    target_include_directories(test_talker_cuda_decode PUBLIC  .)
    target_include_directories(test_talker_cuda_decode PRIVATE
        ../.. ../../vendor native
        ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(test_talker_cuda_decode PRIVATE QWEN_TTS_HAS_CUDA=1)
    target_compile_features   (test_talker_cuda_decode PRIVATE cxx_std_17)
    if(OMINIX_CUDA_USE_CUDNN)
        target_link_libraries     (test_talker_cuda_decode PRIVATE CUDA::cudnn)
        target_compile_definitions(test_talker_cuda_decode PRIVATE OMINIX_CUDA_USE_CUDNN=1)
    endif()
    message(STATUS "qwen_tts: OminiX-CUDA Phase 2.2 decode smoke ENABLED")

    # ---------------------------------------------------------------------
    # Phase 2.3 smoke harness — test_talker_cuda_autoreg
    #
    # Drives a true autoregressive loop: token_embd lookup → forward_decode
    # → CPU lm_head matvec → argmax → next-token. Reports tokens/sec and
    # token-sequence sanity (not-all-same, not-degenerate runs). The lm_head
    # matmul stays on host for Phase 2.3; Phase 2.4 will move it onto GPU.
    # ---------------------------------------------------------------------
    add_executable(test_talker_cuda_autoreg
        native/test_talker_cuda_autoreg.cpp
        native/talker_cuda_engine.cpp
    )
    target_link_libraries     (test_talker_cuda_autoreg PUBLIC  ggml qwen_common)
    target_link_libraries     (test_talker_cuda_autoreg PRIVATE Threads::Threads
                                                                  CUDA::cudart CUDA::cublas CUDA::cublasLt
                                                                  qwen_tts_cuda_kernels)
    target_include_directories(test_talker_cuda_autoreg PUBLIC  .)
    target_include_directories(test_talker_cuda_autoreg PRIVATE
        ../.. ../../vendor native
        ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(test_talker_cuda_autoreg PRIVATE QWEN_TTS_HAS_CUDA=1)
    target_compile_features   (test_talker_cuda_autoreg PRIVATE cxx_std_17)
    if(OMINIX_CUDA_USE_CUDNN)
        target_link_libraries     (test_talker_cuda_autoreg PRIVATE CUDA::cudnn)
        target_compile_definitions(test_talker_cuda_autoreg PRIVATE OMINIX_CUDA_USE_CUDNN=1)
    endif()
    message(STATUS "qwen_tts: OminiX-CUDA Phase 2.3 autoreg smoke ENABLED")

    # ---------------------------------------------------------------------
    # Phase 2.4 smoke harness — test_codec_cuda
    #
    # The qwen3_tts_predictor.gguf shipped on zgx-3675 is a flat llama.cpp
    # Qwen3 transformer export with the SAME tensor schema as the Talker
    # GGUF (blk.<L>.attn_q.weight etc.); only the hyperparameters differ
    # (5 layers / hidden 1024 / FFN 3072 / vocab 30720 = 15 groups × 2048).
    # Phase 2.4 reuses TalkerCudaEngine directly with a TalkerConfig
    # populated from CodePredictorConfig values, then drives a 256-step
    # autoreg loop to prove the transformer engine works on codec dims.
    # No new engine source needed; binary just links the existing
    # talker_cuda_engine.cpp + qwen_tts_cuda_kernels static lib.
    # ---------------------------------------------------------------------
    add_executable(test_codec_cuda
        native/test_codec_cuda.cpp
        native/talker_cuda_engine.cpp
    )
    target_link_libraries     (test_codec_cuda PUBLIC  ggml qwen_common)
    target_link_libraries     (test_codec_cuda PRIVATE Threads::Threads
                                                        CUDA::cudart CUDA::cublas CUDA::cublasLt
                                                        qwen_tts_cuda_kernels)
    target_include_directories(test_codec_cuda PUBLIC  .)
    target_include_directories(test_codec_cuda PRIVATE
        ../.. ../../vendor native
        ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(test_codec_cuda PRIVATE QWEN_TTS_HAS_CUDA=1)
    target_compile_features   (test_codec_cuda PRIVATE cxx_std_17)
    if(OMINIX_CUDA_USE_CUDNN)
        target_link_libraries     (test_codec_cuda PRIVATE CUDA::cudnn)
        target_compile_definitions(test_codec_cuda PRIVATE OMINIX_CUDA_USE_CUDNN=1)
    endif()
    message(STATUS "qwen_tts: OminiX-CUDA Phase 2.4 codec smoke ENABLED")

    # ---------------------------------------------------------------------
    # Phase 2.7a — SpeechTokenizerDecoder scaffold + RVQ decode.
    #
    # Builds the SpeechTokenizerDecoderCudaEngine static lib (GGUF parse +
    # RVQ decode only; pre_conv / pre_transformer / upsample / vocoder are
    # scaffolded but abort) and a tiny smoke harness that drives init +
    # rvq_decode against the qwen_tts_tokenizer_dec.gguf shipped on
    # zgx-3675. Phase 2.7b will add pre_conv + upsample; 2.7c lands the
    # SnakeBeta vocoder + final conv.
    # ---------------------------------------------------------------------
    add_library(qwen_tts_speech_decoder_cuda STATIC
        native/speech_tokenizer_decoder_cuda_engine.cpp
        native/cuda_kernels/decoder_ops.cu
    )
    set_target_properties(qwen_tts_speech_decoder_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
    )
    target_link_libraries     (qwen_tts_speech_decoder_cuda PUBLIC ggml)
    target_link_libraries     (qwen_tts_speech_decoder_cuda PRIVATE
                                                              CUDA::cudart CUDA::cublas CUDA::cublasLt)
    target_include_directories(qwen_tts_speech_decoder_cuda PUBLIC native)
    target_include_directories(qwen_tts_speech_decoder_cuda PRIVATE
                                                              ../.. ../../vendor)
    target_compile_features   (qwen_tts_speech_decoder_cuda PUBLIC cxx_std_17)
    set_target_properties     (qwen_tts_speech_decoder_cuda PROPERTIES
                                                              POSITION_INDEPENDENT_CODE ON)

    add_executable(test_speech_tokenizer_decoder_init
        native/test_speech_tokenizer_decoder_init.cpp
    )
    target_link_libraries     (test_speech_tokenizer_decoder_init PUBLIC
                                                              ggml qwen_tts_speech_decoder_cuda)
    target_link_libraries     (test_speech_tokenizer_decoder_init PRIVATE
                                                              Threads::Threads
                                                              CUDA::cudart CUDA::cublas CUDA::cublasLt)
    target_include_directories(test_speech_tokenizer_decoder_init PUBLIC  .)
    target_include_directories(test_speech_tokenizer_decoder_init PRIVATE
                                                              ../.. ../../vendor native)
    target_compile_definitions(test_speech_tokenizer_decoder_init PRIVATE QWEN_TTS_HAS_CUDA=1)
    target_compile_features   (test_speech_tokenizer_decoder_init PRIVATE cxx_std_17)
    message(STATUS "qwen_tts: OminiX-CUDA Phase 2.7a speech-decoder smoke ENABLED")
    # ---------------------------------------------------------------------
    # Phase 2.7d E2E smoke harness — test_qwen_tts_e2e
    #
    # Drives the full text -> codec tokens -> audio waveform pipeline:
    #   text -> BPE -> seed codec token -> TalkerCudaEngine autoreg ->
    #   CodePredictor (15 acoustic groups, Pattern B sequential) ->
    #   SpeechTokenizerDecoder.decode_audio -> WAV save.
    # ---------------------------------------------------------------------
    add_executable(test_qwen_tts_e2e
        native/test_qwen_tts_e2e.cpp
        native/talker_cuda_engine.cpp
    )
    target_link_libraries     (test_qwen_tts_e2e PUBLIC
                                                              ggml qwen_common qwen_tts_speech_decoder_cuda)
    target_link_libraries     (test_qwen_tts_e2e PRIVATE
                                                              Threads::Threads
                                                              CUDA::cudart CUDA::cublas CUDA::cublasLt
                                                              qwen_tts_cuda_kernels)
    target_include_directories(test_qwen_tts_e2e PUBLIC  .)
    target_include_directories(test_qwen_tts_e2e PRIVATE
                                                              ../.. ../../vendor native
                                                              ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(test_qwen_tts_e2e PRIVATE QWEN_TTS_HAS_CUDA=1)
    target_compile_features   (test_qwen_tts_e2e PRIVATE cxx_std_17)
    if(OMINIX_CUDA_USE_CUDNN)
        target_link_libraries     (test_qwen_tts_e2e PRIVATE CUDA::cudnn)
        target_compile_definitions(test_qwen_tts_e2e PRIVATE OMINIX_CUDA_USE_CUDNN=1)
    endif()
    message(STATUS "qwen_tts: OminiX-CUDA Phase 2.7d E2E smoke ENABLED")

    # ---------------------------------------------------------------------
    # Phase 2.10 — TTS warm-start daemon (tts_server)
    # ---------------------------------------------------------------------
    # Persistent process that loads all weights once and serves synthesis
    # requests over a TCP line-protocol on 127.0.0.1:OMINIX_TTS_SERVER_PORT
    # (default 7777). Amortizes the ~11 s of init+weight-load that
    # dominates cold-start in test_qwen_tts_e2e. Same engines, same
    # numerics; only the entry-point and per-request lifecycle change.
    # ---------------------------------------------------------------------
    add_executable(tts_server
        native/tts_server.cpp
        native/talker_cuda_engine.cpp
    )
    target_link_libraries     (tts_server PUBLIC
                                                              ggml qwen_common qwen_tts_speech_decoder_cuda)
    target_link_libraries     (tts_server PRIVATE
                                                              Threads::Threads
                                                              CUDA::cudart CUDA::cublas CUDA::cublasLt
                                                              CUDA::cublasLt
                                                              qwen_tts_cuda_kernels)
    target_include_directories(tts_server PUBLIC  .)
    target_include_directories(tts_server PRIVATE
                                                              ../.. ../../vendor native
                                                              ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(tts_server PRIVATE QWEN_TTS_HAS_CUDA=1)
    target_compile_features   (tts_server PRIVATE cxx_std_17)
    if(OMINIX_CUDA_USE_CUDNN)
        target_link_libraries     (tts_server PRIVATE CUDA::cudnn)
        target_compile_definitions(tts_server PRIVATE OMINIX_CUDA_USE_CUDNN=1)
    endif()
    message(STATUS "qwen_tts: OminiX-CUDA Phase 2.10 warm-start daemon ENABLED")
endif()
