
#include <cmath>
#include <iostream>

#include <ddc/ddc.hpp>
#include <ddc/pdi.hpp>

#include "ddc_alias_inline_functions.hpp"
#include "../poisson/iqnsolver.hpp"
#include "ivlasovsolver.hpp"
#include "predcorr.hpp"
#include "transpose.hpp"

PredCorr::PredCorr(IVlasovSolver const& vlasov_solver, IQNSolver const& poisson_solver)
    : m_vlasov_solver(vlasov_solver)
    , m_poisson_solver(poisson_solver)
{
}

DFieldSpVrVthetaRTheta PredCorr::operator()(
        DFieldSpVrVthetaRTheta const allfdistribu,
        Real const dt,
        int const steps) const
{
    IdxRangeSpRThetaVrVtheta idx_range_5D(get_idx_range(allfdistribu));
    DFieldMemSpRThetaVrVtheta allfdistribu_5D(idx_range_5D);
    auto allfdistribu_host_alloc
            = ddc::create_mirror_view(get_field(allfdistribu_5D));
    host_t<DFieldSpRThetaVrVtheta> allfdistribu_host = get_field(allfdistribu_host_alloc);

    // electrostatic potential and electric field (depending only on x)
    DFieldMemRTheta electrostatic_potential(
            get_idx_range<GridR, GridTheta>(allfdistribu));
    DVectorFieldMemRTheta<X, Y> electric_field(
            get_idx_range<GridR, GridTheta>(allfdistribu));

    host_t<DFieldMemRTheta> electrostatic_potential_host(
            get_idx_range<GridR, GridTheta>(allfdistribu));

    host_t<DVectorFieldMemRTheta<X, Y>> electric_field_host(
            get_idx_range<GridR, GridTheta>(allfdistribu));

    // a 2D memory block of the same size as fdistribu
    DFieldMemSpVrVthetaRTheta allfdistribu_half_t(get_idx_range(allfdistribu));

    int iter = 0;
    for (; iter < steps; ++iter) {
        Real const iter_time = iter * dt;

        // computation of the electrostatic potential at time tn and
        // the associated electric field
        m_poisson_solver(
                get_field(electrostatic_potential),
                get_field(electric_field),
                get_const_field(allfdistribu));

        Kokkos::Profiling::pushRegion("(GSLX) PDIWrite");
        transpose_layout(
                Kokkos::DefaultExecutionSpace(),
                get_field(allfdistribu_5D),
                get_const_field(allfdistribu));
        // copies necessary to PDI
        ddc::parallel_deepcopy(
                allfdistribu_host,
                get_const_field(allfdistribu_5D));
        ddc::parallel_deepcopy(electrostatic_potential_host, electrostatic_potential);
        //ddc::parallel_deepcopy(electric_field_host, electric_field);
        ddc::PdiEvent("iteration")
                .with("iter", iter)
                .with("time_saved", iter_time)
                .with("fdistribu", allfdistribu_host)
                .with("electrostatic_potential", electrostatic_potential_host);
        Kokkos::Profiling::popRegion();
        // copy fdistribu
        ddc::parallel_deepcopy(allfdistribu_half_t, allfdistribu);

        // predictor
        m_vlasov_solver(get_field(allfdistribu_half_t), get_const_field(electric_field), dt / 2);

        // computation of the electrostatic potential at time tn+1/2
        // and the associated electric field
        m_poisson_solver(
                get_field(electrostatic_potential),
                get_field(electric_field),
                get_const_field(allfdistribu_half_t));

        // correction on a dt
        m_vlasov_solver(get_field(allfdistribu), get_const_field(electric_field), dt);
    }

    Real const final_time = iter * dt;
    m_poisson_solver(
            get_field(electrostatic_potential),
            get_field(electric_field),
            get_const_field(allfdistribu));

    Kokkos::Profiling::pushRegion("(GSLX) PDIWrite");
    // transpose layout before final PDI write
    transpose_layout(
            Kokkos::DefaultExecutionSpace(),
            get_field(allfdistribu_5D),
            get_const_field(allfdistribu));
    //copies necessary to PDI
    ddc::parallel_deepcopy(
            allfdistribu_host,
            get_const_field(allfdistribu_5D));
    ddc::parallel_deepcopy(electrostatic_potential_host, electrostatic_potential);
    ddc::PdiEvent("last_iteration")
            .with("iter", iter)
            .with("time_saved", final_time)
            .with("fdistribu", allfdistribu_host)
            .with("electrostatic_potential", electrostatic_potential_host);
    Kokkos::Profiling::popRegion();

        return allfdistribu;
}