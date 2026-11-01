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

add_library(mstu_computer_vision SHARED
    src/plugin.cpp
    src/schema.cpp
)

target_include_directories(mstu_computer_vision PRIVATE "${MSTU_SDK_ABI}" src)

if(NOT MSVC)
    target_compile_options(mstu_computer_vision PRIVATE -Wall -Wextra -Werror)
endif()

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
add_executable(tests
    tests/tests.cpp
    src/plugin.cpp
    src/schema.cpp
)

target_include_directories(tests PRIVATE "${MSTU_SDK_ABI}" src)

if(NOT MSVC)
    target_compile_options(tests PRIVATE -Wall -Wextra -Werror)
endif()

add_test(NAME tests COMMAND tests)
