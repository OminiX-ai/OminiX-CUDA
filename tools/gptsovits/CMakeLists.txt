# mtmd

find_package(Threads REQUIRED)

OPTION(BUILD_PYTHON "Build Python bindings" OFF)
OPTION(USE_CPP_PINYIN "Enable cpp-pinyin for Chinese G2P" ON)
OPTION(USE_ESPEAK "Enable espeak-ng for English G2P" OFF)

message(STATUS "Building Python bindings: ${BUILD_PYTHON}")
message(STATUS "Use cpp-pinyin: ${USE_CPP_PINYIN}")
message(STATUS "Use espeak-ng: ${USE_ESPEAK}")

set(TARGET gptsovits)
set(SRC_FILES ctx_manager.cpp build_graph.cpp llm.cpp model_loader.cpp model_defs.cpp utils.cpp openvla.cpp gptsovits.cpp audio_postprocess.cpp audio_io.cpp stft.cpp text_preprocessor.cpp)

# kissfft library for STFT computation
set(KISSFFT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/kissfft)
set(KISSFFT_SRC
    ${KISSFFT_DIR}/kiss_fft.c
    ${KISSFFT_DIR}/kiss_fftr.c
)
list(APPEND SRC_FILES ${KISSFFT_SRC})
if(BUILD_PYTHON)
    list(APPEND SRC_FILES interface.cpp)
    add_library(${TARGET} SHARED ${SRC_FILES})
    set_target_properties(${TARGET} PROPERTIES
        PREFIX ""
        SUFFIX ".so"
    )

    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    message(STATUS "Found Python: ${Python_EXECUTABLE} (found version \"${Python_VERSION}\")")
    message(STATUS "Python Include Dir: ${Python_INCLUDE_DIRS} Python Libraries: ${Python_LIBRARIES}")

    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pybind11, sys; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${PYBIND11_CMAKE_DIR})
    find_package(pybind11 CONFIG REQUIRED)
    target_include_directories(${TARGET} PRIVATE ${Python_INCLUDE_DIRS})
    target_link_directories(${TARGET} PRIVATE ${Python_LIBRARIES})
    target_link_libraries(${TARGET} PRIVATE pybind11::module)
    target_compile_definitions(${TARGET} PRIVATE BUILD_PYTHON_BINDINGS)
else()
    list(APPEND SRC_FILES main.cpp)
    add_executable(${TARGET} ${SRC_FILES})
endif()

# set(TARGET example)
# add_executable(${TARGET} ctx_manager.cpp build_graph.cpp llm.cpp model_loader.cpp model_defs.cpp utils.cpp openvla.cpp examples.cpp)

# cpp-pinyin integration
if(USE_CPP_PINYIN)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../xd4/CMakeLists.txt")
        add_subdirectory(../../vendor/cpp-pinyin cpp-pinyin EXCLUDE_FROM_ALL)
        target_compile_definitions(${TARGET} PRIVATE USE_CPP_PINYIN)
        target_link_libraries(${TARGET} PRIVATE cpp-pinyin::cpp-pinyin)
        message(STATUS "cpp-pinyin found and enabled")

        # Copy dictionary files
        add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/cpp-pinyin/res/dict
            $<TARGET_FILE_DIR:${TARGET}>/dict
            COMMENT "Copying cpp-pinyin dictionaries..."
        )

        # Copy G2P data files (opencpop-strict.txt and cmudict.rep)
        add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
            $<TARGET_FILE_DIR:${TARGET}>/tools/gptsovits/data
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/data/opencpop-strict.txt
            $<TARGET_FILE_DIR:${TARGET}>/tools/gptsovits/data/opencpop-strict.txt
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/data/cmudict.rep
            $<TARGET_FILE_DIR:${TARGET}>/tools/gptsovits/data/cmudict.rep
            COMMENT "Copying G2P data files..."
        )
    else()
        message(WARNING "cpp-pinyin not found, using fallback G2P")
        set(USE_CPP_PINYIN OFF)
    endif()
endif()

# espeak-ng integration (optional)
if(USE_ESPEAK)
    find_package(PkgConfig)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(ESPEAK espeak-ng)
        if(ESPEAK_FOUND)
            target_compile_definitions(${TARGET} PRIVATE USE_ESPEAK)
            target_include_directories(${TARGET} PRIVATE ${ESPEAK_INCLUDE_DIRS})
            target_link_libraries(${TARGET} PRIVATE ${ESPEAK_LIBRARIES})
            message(STATUS "espeak-ng found and enabled")
        else()
            message(WARNING "espeak-ng not found, using fallback for English G2P")
            set(USE_ESPEAK OFF)
        endif()
    else()
        set(USE_ESPEAK OFF)
    endif()
endif()

target_link_libraries     (${TARGET} PUBLIC ggml llama)
target_link_libraries     (${TARGET} PRIVATE Threads::Threads)
target_include_directories(${TARGET} PUBLIC  .)
target_include_directories(${TARGET} PRIVATE ../..)
target_include_directories(${TARGET} PRIVATE ../../vendor)
target_include_directories(${TARGET} PRIVATE ${KISSFFT_DIR})
target_compile_features   (${TARGET} PRIVATE cxx_std_17)

add_subdirectory(../../vendor/tokenizers-cpp tokenizers EXCLUDE_FROM_ALL)

if (BUILD_SHARED_LIBS)
    set_target_properties     (${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_BUILD)
    target_compile_definitions(${TARGET} PUBLIC  LLAMA_SHARED)
endif()

# set(MTMD_PUBLIC_HEADERS
#     ${CMAKE_CURRENT_SOURCE_DIR}/mtmd.h
#     ${CMAKE_CURRENT_SOURCE_DIR}/mtmd-helper.h
#     )

set_target_properties(${TARGET}
    PROPERTIES
    PUBLIC_HEADER "${MTMD_PUBLIC_HEADERS}")

install(TARGETS ${TARGET} LIBRARY PUBLIC_HEADER)

if (NOT MSVC)
    # for stb_image.h and miniaudio.h
    target_compile_options(${TARGET} PRIVATE -Wno-cast-qual)
endif()

if (TARGET BUILD_INFO)
    add_dependencies(${TARGET}        BUILD_INFO)
    # add_dependencies(mtmd-helper BUILD_INFO)
endif()

set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME ${TARGET})
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
target_link_libraries  (${TARGET} PRIVATE common Threads::Threads tokenizers_cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
