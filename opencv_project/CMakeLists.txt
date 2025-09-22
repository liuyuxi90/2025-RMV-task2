cmake_minimum_required(VERSION 3.10)

project(opencv_project VERSION 1.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(TESSERACT REQUIRED tesseract)

include_directories(${OpenCV_INCLUDE_DIRS} ${TESSERACT_INCLUDE_DIRS})
link_directories(${TESSERACT_LIBRARY_DIRS})

add_executable(${PROJECT_NAME} src/main.cpp)

# 链接 OpenCV 和 Tesseract
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${TESSERACT_LIBRARIES})

# 让 main.cpp 可以通过 RESOURCE_DIR 访问资源文件夹
target_compile_definitions(${PROJECT_NAME} PRIVATE
    RESOURCE_DIR="${CMAKE_SOURCE_DIR}/resources"
)

