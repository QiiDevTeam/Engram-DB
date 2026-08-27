cmake_minimum_required(VERSION 3.16)
project(engram-cpp VERSION 0.2.0 LANGUAGES CXX)

# EngramDB C++ SDK (header-only RAII wrapper over stable C ABI).
#
# 独立仓库 / Monorepo 双模式：
# - Monorepo（主仓库）：C 头文件真源在 ../../db/include/engram.h，CMake 同时加入该目录与本 SDK 的 include/
# - 独立仓库（单独 git clone sdks/cpp）：本目录 include/engram.h 为 vendored 快照（与 db/include/engram.h 同步），
#   仅依赖本目录即可工作，无需主仓库。构建动态库仍需 cargo 产出的 engram_db.dll/.so，或用 LoadLibrary 动态加载（见 smoke）。
#
# 动态库由 cargo 产出：
#   cargo build -p engram-db            # -> target/debug|release/engram_db.dll/.so
# 亦可只做头文件集成，通过 dlopen/LoadLibrary 在运行时加载（examples/smoke.cpp 示例）。

set(ENGRAM_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
# 优先使用主仓库的真源头文件（若存在），否则回退到 vendored 快照
if(EXISTS "${ENGRAM_REPO_ROOT}/db/include/engram.h")
    set(ENGRAM_C_INCLUDE "${ENGRAM_REPO_ROOT}/db/include")
else()
    set(ENGRAM_C_INCLUDE "${CMAKE_CURRENT_LIST_DIR}/include")
    message(STATUS "EngramDB C++ SDK: using vendored engram.h (standalone mode)")
endif()

add_library(engram_cpp INTERFACE)
add_library(engram::cpp ALIAS engram_cpp)

target_include_directories(engram_cpp INTERFACE
    "${ENGRAM_C_INCLUDE}"
    "${CMAKE_CURRENT_LIST_DIR}/include"
)

target_compile_features(engram_cpp INTERFACE cxx_std_17)

# 仅当 examples/smoke.cpp 存在时才添加冒烟可执行文件（允许 SDK 被 add_subdirectory 嵌入）
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/examples/smoke.cpp")
    add_executable(engram_smoke examples/smoke.cpp)
    target_link_libraries(engram_smoke PRIVATE engram::cpp)
endif()
