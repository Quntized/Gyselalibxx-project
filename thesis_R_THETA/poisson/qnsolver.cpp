// SPDX-License-Identifier: MIT

#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>

#include <ddc/ddc.hpp>

#include "ddc_alias_inline_functions.hpp"
//#include "geometry_rthetavrvtheta.hpp"
#include "qnsolver.hpp"
#include "vector_index_tools.hpp"

QNSolver::QNSolver(PoissonSolver const& solve_poisson, IChargeDensityCalculator const& compute_rho)
    : m_solve_poisson(solve_poisson)
    , m_compute_rho(compute_rho)
{
}

void QNSolver::operator()(
        DFieldRTheta const electrostatic_potential,
        DVectorFieldRTheta<X, Y> const electric_field,
        DConstFieldSpVrVthetaRTheta const allfdistribu) const
{
    Kokkos::Profiling::pushRegion("(GSLX) QNSolver");
    assert((get_idx_range(electrostatic_potential) == get_idx_range<GridR, GridTheta>(allfdistribu)));
    IdxRangeRTheta const idx_range_rtheta = get_idx_range(electrostatic_potential);

    // Compute the RHS of the Quasi-Neutrality equation.
    DFieldMemRTheta rho(idx_range_rtheta);
    DFieldMemVrVtheta contiguous_slice_vrvtheta(get_idx_range<GridVr, GridVtheta>(allfdistribu));
    m_compute_rho(get_field(rho), allfdistribu);

    m_solve_poisson(electrostatic_potential, get_const_field(rho));
    ddc::parallel_fill(ddcHelper::get<X>(electric_field), 0.);
    ddc::parallel_fill(ddcHelper::get<Y>(electric_field), 0.);

    Kokkos::Profiling::popRegion();
}