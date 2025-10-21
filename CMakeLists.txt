cmake_minimum_required(VERSION 3.10)

project(VulkanRenderer)

find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(main main.cpp)

find_path(TINYGLTF_INCLUDE_DIRS "tiny_gltf.h")

target_include_directories(main PRIVATE ${TINYGLTF_INCLUDE_DIRS})
target_link_libraries(main PRIVATE glfw glm::glm nlohmann_json::nlohmann_json)