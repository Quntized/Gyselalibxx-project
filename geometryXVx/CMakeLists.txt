# SPDX-License-Identifier: MIT

set(BASIC_GEOMETRY_XVx_VARIANTS_LIST xperiod_vx xnonperiod_vx)


add_executable(vlasovpoisson_xvx_fem_uniform_xperiod_vx vlasovpoisson_xvx_fem_uniform.cpp)
target_link_libraries(vlasovpoisson_xvx_fem_uniform_xperiod_vx
    PUBLIC
        DDC::core
        DDC::pdi
        paraconf::paraconf
        PDI::PDI_C
        gyselalibxx::initialisation_xperiod_vx
        gyselalibxx::interpolation
        gyselalibxx::paraconfpp
        gyselalibxx::poisson_xperiod_vx
        gyselalibxx::speciesinfo
        gyselalibxx::time_integration_xperiod_vx
        gyselalibxx::boltzmann_xperiod_vx
        gyselalibxx::advection
        gyselalibxx::io
        gyselalibxx::pde_solvers
        gyselalibxx::utils

)
