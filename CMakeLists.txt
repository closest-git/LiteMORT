cmake_minimum_required(VERSION 2.8.12)
SET(PROJECT_NAME LiteMORT)
PROJECT(${PROJECT_NAME} LANGUAGES CXX)
message("Hello, It's ${PROJECT_NAME} by CYS!")
cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
set (CMAKE_CXX_STANDARD 11)

OPTION(USE_OPENMP "Enable OpenMP" ON)

if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    # Ignore unknown #pragma warning
    if( (CMAKE_CXX_COMPILER_ID MATCHES "[cC][lL][aA][nN][gG]")
      OR (CMAKE_CXX_COMPILER_ID MATCHES "[gG][nN][uU]"))
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    endif()
endif(USE_OPENMP)

if(MSVC)
    if(MSVC_VERSION LESS 1900)
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a newer MSVC.")
    endif()

    SET(variables
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_MINSIZEREL
        CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_MINSIZEREL
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
    )
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /O2 /Ob2 /Oi /Ot /Oy /GL")
else()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()


set (cLIB "../../lib/")
#set (PYTHON_INC "../../lib/")
SET(some_COMPILE_FLAGS "-std=c++11 -pthread -O3 -I${cLIB}" )

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${some_COMPILE_FLAGS}")

SET(SOURCE_DIR "src")
# Tell cmake that headers are in alse in source_dir
include_directories(${SOURCE_DIR})
SET(SOURCE_FILES "${SOURCE_DIR}/LiteMORT.cpp ${SOURCE_DIR}/python/pyMORT_DLL.cpp")
add_subdirectory(${SOURCE_DIR}/data_fold)
add_subdirectory(${SOURCE_DIR}/tree)
add_subdirectory(${SOURCE_DIR}/util)
add_subdirectory(${SOURCE_DIR}/learn)
add_subdirectory(${SOURCE_DIR}/EDA)

ADD_LIBRARY("${PROJECT_NAME}" SHARED ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} data_fold_ tree_ util_ learn_ eda_)
# SET(TEST_DIR "tests")
# SET(TESTS ${SOURCES}
#     "${TEST_DIR}/test_main.cpp"
#     "${TEST_DIR}/test_math.cpp")

# Generate a test executable
# include_directories(lib/catch/include)
# add_executable("${PROJECT_NAME}_test" ${SOURCE_FILES})


# Generate python module
# add_subdirectory(lib/pybind11)
# pybind11_add_module(python_cpp_example ${SOURCES} "${SOURCE_DIR}/bindings.cpp")
