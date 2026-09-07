add_library(gyselalibxx_project
    fluid_moments/fluid_moments.cpp
)
target_include_directories(gyselalibxx_project PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/fluid_moments>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(gyselalibxx_project PUBLIC
    gslx::quadrature
    gslx::utils
    gslx::speciesinfo
    gslx::io
    gslx::mpi_parallelisation
)