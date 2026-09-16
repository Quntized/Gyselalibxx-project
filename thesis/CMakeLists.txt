cmake_minimum_required(VERSION 3.25)
project(czarnys LANGUAGES CXX)

if(NOT TARGET gslx::advection)
    set(GYSELALIBXX_BUILD_SIMULATIONS OFF CACHE BOOL "" FORCE)
    set(GYSELALIBXX_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(GYSELALIBXX_ENABLE_DEPRECATED OFF CACHE BOOL "" FORCE)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../external/gyselalibxx ${CMAKE_BINARY_DIR}/gyselalibxx)
endif()

add_executable(czarnys
    sim.cpp
)
target_include_directories(czarnys PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../external/gyselalibxx/simulations/utilities
)
find_package(DDC REQUIRED COMPONENTS pdi)
find_package(PDI REQUIRED)
find_package(paraconf REQUIRED COMPONENTS C)
target_link_libraries(czarnys
        PUBLIC
            DDC::core
            DDC::pdi

            paraconf::paraconf

            PDI::PDI_C

            gslx::advection
            gslx::diocotron_init_eq
            gslx::geometry_RTheta
            gslx::interpolation
            gslx::io
            gslx::paraconfpp
            gslx::poisson_RTheta
            gslx::quadrature
            gslx::time_integration_RTheta
            gslx::utils
    )
