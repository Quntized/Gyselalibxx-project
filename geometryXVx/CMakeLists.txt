# SPDX-License-Identifier: MIT

set(BASIC_GEOMETRY_XVx_VARIANTS_LIST xperiod_vx xnonperiod_vx)


add_executable(vlasovpoisson_xvx_fem_uniform_xperiod_vx vlasovpoisson_xvx_fem_uniform.cpp)
target_link_libraries(vlasovpoisson_xvx_fem_uniform_xperiod_vx
    PUBLIC
        DDC::core
        DDC::pdi
        paraconf::paraconf
        PDI::PDI_C
        gslx::initialisation_xperiod_vx
        gslx::interpolation
        gslx::paraconfpp
        gslx::poisson_xperiod_vx
        gslx::speciesinfo
        gslx::time_integration_xperiod_vx
        gslx::boltzmann_xperiod_vx
        gslx::advection
        gslx::io
        gslx::pde_solvers
        gslx::utils

)
