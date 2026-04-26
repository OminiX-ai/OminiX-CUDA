# Qwen-Image-Edit native DiT engine (ominix-cuda Phase 3 — CUDA port).
#
# Phase 3.1 (THIS): scaffold target `test_image_diffusion_cuda_init` that
# constructs `ominix_cuda::ImageDiffusionCudaEngine`, opens a QIE-Edit GGUF,
# uploads every transformer_blocks.* weight + global head/tail to GPU,
# and prints a PASS/FAIL line. Phase 3.2 fills the per-block forward; Phase
# 3.3 wires the full denoising loop + VAE decode hand-off.
#
# Gated on QWEN_IMAGE_EDIT_CUDA. Default OFF on hosts without CUDA toolkit;
# turned ON in the GB10 build presets.

find_package(Threads REQUIRED)

option(QWEN_IMAGE_EDIT_CUDA "Build native CUDA Image Diffusion engine (Phase 3)" OFF)

if(NOT QWEN_IMAGE_EDIT_CUDA)
    message(STATUS "qwen_image_edit: native CUDA engine DISABLED "
                   "(enable with -DQWEN_IMAGE_EDIT_CUDA=ON)")
    return()
endif()

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    # GB10 is sm_121 (Blackwell). Allow override from the cmake CLI.
    set(CMAKE_CUDA_ARCHITECTURES 121)
endif()
message(STATUS "qwen_image_edit: native CUDA engine ENABLED "
               "(CUDA ${CUDAToolkit_VERSION}, sm=${CMAKE_CUDA_ARCHITECTURES})")

# ---------------------------------------------------------------------------
# Phase 3.1 smoke harness — test_image_diffusion_cuda_init
# ---------------------------------------------------------------------------
add_executable(test_image_diffusion_cuda_init
    native/test_image_diffusion_cuda_init.cpp
    native/image_diffusion_cuda_engine.cpp
)
target_link_libraries     (test_image_diffusion_cuda_init PUBLIC  ggml qwen_common)
target_link_libraries     (test_image_diffusion_cuda_init PRIVATE Threads::Threads
                                                                    CUDA::cudart CUDA::cublas)
target_include_directories(test_image_diffusion_cuda_init PUBLIC  .)
target_include_directories(test_image_diffusion_cuda_init PRIVATE
    ../.. ../../vendor native
    ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(test_image_diffusion_cuda_init PRIVATE
    QWEN_IMAGE_EDIT_HAS_CUDA=1)
target_compile_features   (test_image_diffusion_cuda_init PRIVATE cxx_std_17)
if(OMINIX_CUDA_USE_CUDNN)
    target_link_libraries     (test_image_diffusion_cuda_init PRIVATE CUDA::cudnn)
    target_compile_definitions(test_image_diffusion_cuda_init PRIVATE OMINIX_CUDA_USE_CUDNN=1)
endif()

if(BUILD_SHARED_LIBS)
    set_target_properties(test_image_diffusion_cuda_init PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
endif()

install(TARGETS test_image_diffusion_cuda_init RUNTIME)
message(STATUS "qwen_image_edit: Phase 3.1 smoke harness ENABLED "
               "(test_image_diffusion_cuda_init)")
