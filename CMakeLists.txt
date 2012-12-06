CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)

project(mini_project2)
find_package(OpenCV REQUIRED)
add_executable(mini_project2 main.cpp)
target_link_libraries(mini_project2 ${OpenCV_LIBS})
