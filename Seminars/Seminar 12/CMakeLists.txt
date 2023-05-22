cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(torch_api)

set(CMAKE_CXX_STANDARD 17)
add_executable(torch_api_simple simple.cpp)
add_executable(torch_api_resnet resnet.cpp)

# Link matplotlib-cpp
include(FetchContent)
FetchContent_Declare(
    matplotlib_cpp
    GIT_REPOSITORY "https://github.com/nakhodnov17/matplotlib-cpp.git"
)
FetchContent_MakeAvailable(matplotlib_cpp)
include_directories(${matplotlib_cpp_SOURCE_DIR})

# Link OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Link torch
cmake_policy(SET CMP0054 NEW)
if(NOT DEFINED TORCH_LIB_PATH)
    if(WIN32)
        if(CMAKE_BUILD_TYPE MATCHES DEBUG)
            set(TORCH_SUFFIX "win_debug_${TORCH_VERSION}_${TORCH_CUDA_VERSION}")
            set(TORCH_URL "https://download.pytorch.org/libtorch/${TORCH_CUDA_VERSION}/libtorch-win-shared-with-deps-debug-${TORCH_VERSION}%2B${TORCH_CUDA_VERSION}.zip")
        else ()
            set(TORCH_SUFFIX "win_${TORCH_VERSION}_${TORCH_CUDA_VERSION}")
            set(TORCH_URL "https://download.pytorch.org/libtorch/${TORCH_CUDA_VERSION}/libtorch-win-shared-with-deps-${TORCH_VERSION}%2B${TORCH_CUDA_VERSION}.zip")
        endif ()
    elseif (APPLE)
        set(TORCH_SUFFIX "macos_${TORCH_VERSION}_${TORCH_CUDA_VERSION}")
        set(TORCH_URL "https://download.pytorch.org/libtorch/${TORCH_CUDA_VERSION}/libtorch-macos-${TORCH_VERSION}.zip")
    elseif (UNIX)
        set(TORCH_SUFFIX "unix_${TORCH_VERSION}_${TORCH_CUDA_VERSION}")
        set(TORCH_URL "https://download.pytorch.org/libtorch/${TORCH_CUDA_VERSION}/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2B${TORCH_CUDA_VERSION}.zip")
    endif()

    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/torch_${TORCH_SUFFIX}.zip")
        message("TORCH_LIB_PATH is not set. Downloading pytorch sources from ${TORCH_URL}")
        file(DOWNLOAD "${TORCH_URL}" "${CMAKE_SOURCE_DIR}/torch_${TORCH_SUFFIX}.zip")
    endif ()

    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/torch_${TORCH_SUFFIX}")
        message("Extract pytorch sources to ${CMAKE_SOURCE_DIR}/torch_${TORCH_SUFFIX}")
        file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/torch_${TORCH_SUFFIX}")
        execute_process(
                COMMAND ${CMAKE_COMMAND} -E tar xvf "${CMAKE_SOURCE_DIR}/torch_${TORCH_SUFFIX}.zip"
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/torch_${TORCH_SUFFIX}"
        )
    endif ()

    set(TORCH_LIB_PATH "${CMAKE_SOURCE_DIR}/torch_${TORCH_SUFFIX}/libtorch")
    set(CMAKE_PREFIX_PATH "${TORCH_LIB_PATH}/share/cmake/Torch")
else ()
    set(CMAKE_PREFIX_PATH "${TORCH_LIB_PATH}/share/cmake/Torch")
endif ()

set(CAFFE2_USE_CUDNN 1)
find_package(Torch REQUIRED PATHS "${TORCH_LIB_PATH}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

target_link_libraries(torch_api_simple matplotlib_cpp ${OpenCV_LIBS} ${TORCH_LIBRARIES})
target_link_libraries(torch_api_resnet matplotlib_cpp ${OpenCV_LIBS} ${TORCH_LIBRARIES})

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET torch_api_simple
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:torch_api_simple>)
endif()
if(MSVC)
   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
   add_custom_command(TARGET torch_api_resnet
           POST_BUILD
           COMMAND ${CMAKE_COMMAND} -E copy_if_different
           ${TORCH_DLLS}
           $<TARGET_FILE_DIR:torch_api_resnet>)
endif()
