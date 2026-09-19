cmake_minimum_required(VERSION 3.25)

add_executable(thesis_r_theta
    core.cpp
    initialisation/maxwellianequilibrium.cpp
    poisson/chargedensitycalculator.cpp
    poisson/qnsolver.cpp
    time_integration/predcorr.cpp
    vlasov/splitvlasovsolver.cpp
)

target_include_directories(thesis_r_theta PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/geometry
    ${CMAKE_CURRENT_LIST_DIR}/initialisation
    ${CMAKE_CURRENT_LIST_DIR}/poisson
    ${CMAKE_CURRENT_LIST_DIR}/time_integration
    ${CMAKE_CURRENT_LIST_DIR}/vlasov
)
find_package(DDC REQUIRED COMPONENTS pdi)
find_package(PDI REQUIRED)
find_package(paraconf REQUIRED COMPONENTS C)

target_link_libraries(thesis_r_theta PUBLIC
    DDC::core
    DDC::pdi
    PDI::PDI_C
    paraconf::paraconf
    gslx::advection
    gslx::geometry_RTheta
    gslx::interpolation
    gslx::io
    gslx::paraconfpp
    gslx::pde_solvers
    gslx::quadrature
    gslx::speciesinfo
    gslx::timestepper
    gslx::utils
)