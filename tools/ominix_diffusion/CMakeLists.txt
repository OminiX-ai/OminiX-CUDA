# tools/ominix_diffusion - Stable-Diffusion 模块（库 + cli + server）

# SD 推理库 (libstable-diffusion.a)
add_subdirectory(src)

# 可执行程序
if(SD_BUILD_EXAMPLES)
    add_subdirectory(cli)
    add_subdirectory(server)
endif()
