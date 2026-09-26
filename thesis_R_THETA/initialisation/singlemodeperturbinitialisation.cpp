#include <ddc/ddc.hpp>

#include "ddc_alias_inline_functions.hpp"
#include "ddc_helper.hpp"
#include "singlemodeperturbinitialisation.hpp"


SingleModePerturbInitialisation::SingleModePerturbInitialisation(
        DConstFieldSpVrVtheta fequilibrium,
        host_t<IFieldMemSp> init_perturb_mode,
        host_t<DFieldMemSp> init_perturb_amplitude)
    : m_fequilibrium(fequilibrium)
    , m_init_perturb_mode(std::move(init_perturb_mode))
    , m_init_perturb_amplitude(std::move(init_perturb_amplitude))
{
}

DFieldSpVrVthetaRTheta SingleModePerturbInitialisation::operator()(
        DFieldSpVrVthetaRTheta const allfdistribu) const
{
    IdxRangeSp const gridsp = get_idx_range<Species>(allfdistribu);
    IdxRangeRTheta const gridrtheta = get_idx_range<GridR, GridTheta>(allfdistribu);
        IdxRangeVrVtheta const gridvrvtheta = get_idx_range<GridVr, GridVtheta>(allfdistribu);

    // Initialisation of the perturbation
    DFieldMemRTheta perturbation_alloc(
            "perturbation (SingleModePerturbInitialisation::operator())",
            gridrtheta);
    DConstFieldSpVrVtheta fequilibrium_proxy = get_const_field(m_fequilibrium);
    DFieldRTheta perturbation_proxy = get_field(perturbation_alloc);
    ddc::host_for_each(gridsp, [&](IdxSp const isp) {
        perturbation_initialisation(
                perturbation_proxy,
                m_init_perturb_mode(isp),
                m_init_perturb_amplitude(isp));

        // Initialisation of the distribution function --> fill values
        const std::source_location location = std::source_location::current();
        ddc::parallel_for_each(
                location.function_name(),
                Kokkos::DefaultExecutionSpace(),
                                gridrtheta,
                                KOKKOS_LAMBDA(IdxRTheta const irtheta) {
                                        IdxR const ir = ddc::select<GridR>(irtheta);
                                        IdxTheta const itheta = ddc::select<GridTheta>(irtheta);
                                        for (IdxVr const ivr : ddc::select<GridVr>(gridvrvtheta)) {
                                                for (IdxVtheta const ivtheta : ddc::select<GridVtheta>(gridvrvtheta)) {
                                                        double fdistribu_val = fequilibrium_proxy(isp, ivr, ivtheta)
                                                                                                   * (1. + perturbation_proxy(ir, itheta));
                                                        if (fdistribu_val < 1.e-60) {
                                                                fdistribu_val = 1.e-60;
                                                        }
                                                        allfdistribu(isp, ivr, ivtheta, ir, itheta) = fdistribu_val;
                                                }
                    }
                });
    });
    return allfdistribu;
}


SingleModePerturbInitialisation SingleModePerturbInitialisation::init_from_input(
        DConstFieldSpVrVtheta allfequilibrium,
        IdxRangeSp idx_range_kinsp,
        PC_tree_t const& yaml_input_file)
{
    host_t<IFieldMemSp> init_perturb_mode(idx_range_kinsp);
    host_t<DFieldMemSp> init_perturb_amplitude(idx_range_kinsp);

    for (IdxSp const isp : idx_range_kinsp) {
        PC_tree_t const conf_isp
                = PCpp_get(yaml_input_file, ".SpeciesInfo[%d]", isp - idx_range_kinsp.front());

        init_perturb_amplitude(isp) = PCpp_double(conf_isp, ".perturb_amplitude");
        init_perturb_mode(isp) = static_cast<int>(PCpp_int(conf_isp, ".perturb_mode"));
    }

    return SingleModePerturbInitialisation(
            allfequilibrium,
            std::move(init_perturb_mode),
            std::move(init_perturb_amplitude));
}


void SingleModePerturbInitialisation::perturbation_initialisation(
        DFieldRTheta const perturbation,
        int const perturb_mode,
        Real const perturb_amplitude) const
{
        IdxRangeRTheta const gridrtheta = get_idx_range<GridR, GridTheta>(perturbation);
    Real const kr = perturb_mode * 2. * M_PI
                    / ddcHelper::total_interval_length(ddc::select<GridR>(gridrtheta));
    Real const ktheta = perturb_mode * 2. * M_PI
                    / ddcHelper::total_interval_length(ddc::select<GridTheta>(gridrtheta));

    const std::source_location location = std::source_location::current();
    ddc::parallel_for_each(
            location.function_name(),
            Kokkos::DefaultExecutionSpace(),
            gridrtheta,
            KOKKOS_LAMBDA(IdxRTheta const irtheta) {
                IdxR const ir = ddc::select<GridR>(irtheta);
                CoordR const r = ddc::coordinate(ir);
                IdxTheta const itheta = ddc::select<GridTheta>(irtheta);
                CoordTheta const theta = ddc::coordinate(itheta);
                perturbation(ir, itheta)
                        = perturb_amplitude * (Kokkos::cos(kr * r) + Kokkos::cos(ktheta * theta));
            });

}