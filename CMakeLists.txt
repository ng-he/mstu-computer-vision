cmake_minimum_required(VERSION 3.16)

project(mstu-computer-vision CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# The generated header is the whole of what this builds against: no library of
# the SDK is linked. Inside the centre checkout it is a sibling; a clone on its
# own is told where it is.
set(MSTU_SDK_ABI "" CACHE PATH "The folder holding mstu-sdk.h")

if(NOT MSTU_SDK_ABI)
    set(sibling "${CMAKE_CURRENT_SOURCE_DIR}/../../mstu-sdk/abi")

    if(EXISTS "${sibling}/mstu-sdk.h")
        set(MSTU_SDK_ABI "${sibling}")
    else()
        message(FATAL_ERROR
            "mstu-sdk.h not found. Build inside the mstu checkout, or pass "
            "-DMSTU_SDK_ABI=<the folder holding it>.")
    endif()
endif()

# OpenCV's dnn carries the whole way from a jpeg to a list of boxes. Without
# it the node still loads, describes itself and hands images on, and says in
# its own reading that it was built with nothing to detect with -- so a clone
# of the centre repository stays green on a machine that has not got it.
find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs dnn)

set(sources src/plugin.cpp src/schema.cpp src/worker.cpp src/labels.cpp
            src/preview.cpp)

if(OpenCV_FOUND)
    list(APPEND sources src/opencv.cpp)
    message(STATUS "mstu-computer-vision: detecting with OpenCV ${OpenCV_VERSION}")
else()
    list(APPEND sources src/nobackend.cpp)
    message(STATUS "mstu-computer-vision: no OpenCV, nothing will be detected")
endif()

function(mstu_target name)
    target_include_directories(${name} PRIVATE "${MSTU_SDK_ABI}" src)

    if(OpenCV_FOUND)
        target_link_libraries(${name} PRIVATE ${OpenCV_LIBS})

        # SYSTEM, or -Werror turns OpenCV's own deprecation warnings -- its
        # 4.6 headers add enums together -- into our build failing.
        target_include_directories(${name} SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_compile_definitions(${name} PRIVATE MSTU_WITH_OPENCV)
    endif()

    if(NOT MSVC)
        target_compile_options(${name} PRIVATE -Wall -Wextra -Werror)
    endif()
endfunction()

add_library(mstu_computer_vision SHARED ${sources})

mstu_target(mstu_computer_vision)

# collect.sh looks in dist/ for the library and the ui folder, the same as it
# does for a plugin cargo knows nothing about.
set_target_properties(mstu_computer_vision PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/dist"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/dist"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/dist"
)

enable_testing()

# The tests link the plugin's own sources rather than the library, so they can
# reach what the descriptor does not expose.
add_executable(tests tests/tests.cpp ${sources})

mstu_target(tests)

# Where model.sh puts what it fetched, so a test can look for it and say it
# was skipped rather than failing when it is not there.
target_compile_definitions(tests PRIVATE
    MSTU_MODELS="${CMAKE_CURRENT_SOURCE_DIR}/models")

add_test(NAME tests COMMAND tests)
