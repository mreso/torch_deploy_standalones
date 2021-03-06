
cmake_minimum_required(VERSION 3.10)

project(cpp_backend_poc_eager)

find_package(Torch REQUIRED)

add_executable(already_in_use already_in_use.cpp)
target_compile_features(already_in_use PRIVATE cxx_std_14)

target_include_directories(already_in_use PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(already_in_use PRIVATE ${TORCH_LIBRARIES})

add_executable(out_of_memory out_of_memory.cpp)
target_compile_features(out_of_memory PRIVATE cxx_std_14)

target_include_directories(out_of_memory PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(out_of_memory PRIVATE ${TORCH_LIBRARIES})

add_executable(gaps gaps.cpp)
target_compile_features(gaps PRIVATE cxx_std_14)

target_include_directories(gaps PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(gaps PRIVATE ${TORCH_LIBRARIES})

add_executable(batched_threading batched_threading.cpp)
target_compile_features(batched_threading PRIVATE cxx_std_14)

target_include_directories(batched_threading PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(batched_threading PRIVATE ${TORCH_LIBRARIES})