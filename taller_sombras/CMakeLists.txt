cmake_minimum_required(VERSION 3.15)
project(TallerSombras CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

add_executable(taller_sombras
    src/main.cpp
)
target_link_libraries(taller_sombras ${OpenCV_LIBS})
