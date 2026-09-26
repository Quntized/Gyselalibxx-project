
#include <ddc/ddc.hpp>

#include "maxwellianequilibrium.hpp"

MaxwellianEquilibrium::MaxwellianEquilibrium(
        host_t<FieldMemSp<double>> density_eq,
        host_t<FieldMemSp<double>> temperature_eq,
        host_t<FieldMemSp<double>> mean_velocity_eq)
    : m_density_eq(std::move(density_eq))
    , m_temperature_eq(std::move(temperature_eq))
    , m_mean_velocity_eq(std::move(mean_velocity_eq))
{
}

DFieldSpVrVtheta MaxwellianEquilibrium::operator()(DFieldSpVrVtheta const allfequilibrium) const
{
    IdxRangeSp const gridsp = get_idx_range<Species>(allfequilibrium);
    IdxRangeVrVtheta const gridvrvt = get_idx_range<GridVr, GridVtheta>(allfequilibrium);

    // Initialisation of the maxwellian
    DFieldMemVrVtheta maxwellian_alloc("maxwellian (MaxwellianEquilibrium::operator())", gridvrvt);
    DFieldVrVtheta maxwellian = get_field(maxwellian_alloc);
    ddc::host_for_each(gridsp, [&](IdxSp const isp) {
        compute_maxwellian(
                maxwellian,
                m_density_eq(isp),
                m_temperature_eq(isp),
                m_mean_velocity_eq(isp));

        const std::source_location location = std::source_location::current();
        ddc::parallel_for_each(
                location.function_name(),
                Kokkos::DefaultExecutionSpace(),
                gridvrvt,
                KOKKOS_LAMBDA(IdxVrVtheta const ivrvt) {
                    allfequilibrium(isp, ivrvt) = maxwellian(ivrvt);
                });
    });
    return allfequilibrium;
}


MaxwellianEquilibrium MaxwellianEquilibrium::init_from_input(
        IdxRangeSp idx_range_kinsp,
        PC_tree_t const& yaml_input_file)
{
    host_t<FieldMemSp<double>> l_density_eq(idx_range_kinsp);
    host_t<FieldMemSp<double>> l_temperature_eq(idx_range_kinsp);
    host_t<FieldMemSp<double>> l_mean_velocity_eq(idx_range_kinsp);

    for (IdxSp const isp : idx_range_kinsp) {
        PC_tree_t const conf_isp
                = PCpp_get(yaml_input_file, ".SpeciesInfo[%d]", isp - idx_range_kinsp.front());

        l_density_eq(isp) = PCpp_double(conf_isp, ".density_eq");
        l_temperature_eq(isp) = PCpp_double(conf_isp, ".temperature_eq");
        l_mean_velocity_eq(isp) = PCpp_double(conf_isp, ".mean_velocity_eq");
    }

    return MaxwellianEquilibrium(
            std::move(l_density_eq),
            std::move(l_temperature_eq),
            std::move(l_mean_velocity_eq));
}


/*
 Computing the Maxwellian function as
  fM(vx,vy) = n/(2*PI*T)*exp(-(vx**2+vy**2)/(2*T))
 with n the density and T the temperature and
*/
void MaxwellianEquilibrium::compute_maxwellian(
        DFieldVrVtheta const fMaxwellian,
        double const density,
        double const temperature,
        double const mean_velocity)
{
    double const inv_2pi = 1. / (2. * M_PI * temperature);
    IdxRangeVrVtheta const gridvrvt = get_idx_range<GridVr, GridVtheta>(fMaxwellian);

    const std::source_location location = std::source_location::current();
    ddc::parallel_for_each(
            location.function_name(),
            Kokkos::DefaultExecutionSpace(),
            gridvrvt,
            KOKKOS_LAMBDA(IdxVrVtheta const ivrvt) {
                double const vr = ddc::coordinate(ddc::select<GridVr>(ivrvt));
                double const vtheta = ddc::coordinate(ddc::select<GridVtheta>(ivrvt));
                fMaxwellian(ivrvt) = density * inv_2pi
                                     * Kokkos::exp(
                                             -((vr - mean_velocity) * (vr - mean_velocity)
                                               + (vtheta - mean_velocity) * (vtheta - mean_velocity))
                                             / (2. * temperature));
            });
}