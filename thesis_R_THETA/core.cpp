#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>

#include <ddc/pdi.hpp>
#include <paraconf.h>
#include <pdi.h>

#include "geometry/geometry_rthetavrvtheta.hpp"
#include "geometry/spline_definitions_r_theta.hpp"
#include "czarny_to_cartesian.hpp"
#include "discrete_poloidal_cs_spline_mapping_builder.hpp"
#include "input.hpp"
#include "output.hpp"
#include "polar_spline_fem_poisson_like_solver.hpp"
#include "polar_foot_finder.hpp"
#include "bsl_advection_vx.hpp"
#include "poisson/qnsolver.hpp"
#include "poisson/chargedensitycalculator.hpp"
#include "pdi_out.yml.hpp"
#include "params.yaml.hpp"
#include "quadrature.hpp"
#include "simpson_quadrature.hpp"
#include "splitvlasovsolver.hpp"
#include "time_integration/predcorr.hpp"
#include "species_info.hpp"
#include "species_init.hpp"
#include "initialisation/maxwellianequilibrium.hpp"
#include "initialisation/singlemodeperturbinitialisation.hpp"
#include "bsl_advection_polar.hpp"
#include "rk4.hpp"

namespace fs = std::filesystem;

using Mapping = CzarnyToCartesian<R, Theta, X, Y>;
using DiscreteMappingBuilder
        = DiscretePoloidalCSSplineMappingBuilder<X, Y, SplineInterpolatorRThetaConst>;
using DiscreteMapping = typename DiscreteMappingBuilder::MappingType;
using PoissonSolver = PolarSplineFEMPoissonLikeSolver<
        GridR,
        GridTheta,
        PolarBSplinesRTheta,
        SplineInterpolatorRTheta,
        typename DiscreteMappingBuilder::MappingType>;

int main(int argc, char** argv)
{
    fs::create_directory("output");
    PC_tree_t conf_gyselalibxx = parse_executable_arguments(argc, argv, params_yaml);
    PC_tree_t conf_pdi = PC_parse_string(PDI_CFG);
    PC_errhandler(PC_NULL_HANDLER);
    PDI_init(conf_pdi);

    Kokkos::ScopeGuard kokkos_scope(argc, argv);
    ddc::ScopeGuard ddc_scope(argc, argv);



    IdxRangeR const mesh_r = init_pseudo_uniform_spline_dependent_idx_range<
            GridR,
            BSplinesR,
            SplineInterpPointsR>(conf_gyselalibxx, "r");
    IdxRangeTheta const mesh_theta = init_pseudo_uniform_spline_dependent_idx_range<
            GridTheta,
            BSplinesTheta,
            SplineInterpPointsTheta>(conf_gyselalibxx, "theta");
    IdxRangeVr const mesh_vr = init_pseudo_uniform_spline_dependent_idx_range<
            GridVr,
            BSplinesVr,
            SplineInterpPointsVr>(conf_gyselalibxx, "vr");
    IdxRangeVtheta const mesh_vtheta = init_pseudo_uniform_spline_dependent_idx_range<
            GridVtheta,
            BSplinesVtheta,
            SplineInterpPointsVtheta>(conf_gyselalibxx, "vtheta");
    IdxRangeSp const idxrange_kinsp = init_species(conf_gyselalibxx);
    IdxRangeRTheta const idxrange_rtheta(mesh_r, mesh_theta);
    IdxRangeVrVtheta const idxrange_vrvtheta(mesh_vr, mesh_vtheta);
    IdxRangeSpVrVtheta const idxrange_spvrvtheta(idxrange_kinsp, idxrange_vrvtheta);
    IdxRangeSpVrVthetaRTheta const idxrange_spvrvthetartheta(idxrange_kinsp,idxrange_vrvtheta,idxrange_rtheta);
    host_t<FieldMemRTheta<CoordRTheta>> coords(idxrange_rtheta);
    ddc::host_for_each(idxrange_rtheta, [&](IdxRTheta const irtheta) {
        coords(irtheta) = ddc::coordinate(irtheta);
    });
    SplineInterpolatorRTheta interpolator(idxrange_rtheta);
    SplineInterpolatorRThetaConst interpolator_const(idxrange_rtheta);
    SplineRThetaBuilder const& builder(interpolator.get_builder());
    double const czarny_epsilon = PCpp_double(conf_gyselalibxx, ".CzarnyMapping.epsilon");
    double const czarny_e = PCpp_double(conf_gyselalibxx, ".CzarnyMapping.e");
    double const major_radius = PCpp_double(conf_gyselalibxx, ".CzarnyMapping.x0");
    double const vertical_offset = PCpp_double(conf_gyselalibxx, ".CzarnyMapping.y0");
    Coord<X, Y> origin_point(major_radius, vertical_offset);
    const Mapping mapping(czarny_epsilon, czarny_e, origin_point);
    DiscreteMappingBuilder const
            discrete_mapping_builder(Kokkos::DefaultExecutionSpace(), mapping, interpolator_const);
    DiscreteMapping const discrete_mapping = discrete_mapping_builder();
    ddc::init_discrete_space<PolarBSplinesRTheta>(discrete_mapping);
    IdxRangeBSRTheta const idx_range_bsplinesRTheta = get_spline_idx_range(builder);
    SplineInterpolatorVr interpolator_vr(mesh_vr);
    SplineInterpolatorVtheta interpolator_vtheta(mesh_vtheta);
    FieldMemSpVrVtheta<double> allfequilibrium(idxrange_spvrvtheta);
    MaxwellianEquilibrium const init_fequilibrium
            = MaxwellianEquilibrium::init_from_input(idxrange_kinsp, conf_gyselalibxx);
    init_fequilibrium(get_field(allfequilibrium));
    DFieldMemSpVrVthetaRTheta allfdistribu(idxrange_spvrvthetartheta);
    SingleModePerturbInitialisation const init
            = SingleModePerturbInitialisation::init_from_input(
                    get_const_field(allfequilibrium),
                    idxrange_kinsp,
                    conf_gyselalibxx);
    init(get_field(allfdistribu));
    double const deltat = PCpp_double(conf_gyselalibxx, ".Algorithm.deltat");
    int const nbiter = static_cast<int>(PCpp_int(conf_gyselalibxx, ".Algorithm.nbiter"));
    int const time_step_diag = static_cast<int>(PCpp_int(conf_gyselalibxx, ".Output.time_step_diag"));
    RK4Builder const time_stepper;
    PolarFootFinder find_feet = make_polar_foot_finder<
            FootFindingSpace::PSEUDO_PHYSICAL,
            AdvectionFieldSpace::PHYSICAL>(
            time_stepper,
            mapping,
            idxrange_spvrvthetartheta,
            interpolator_const);
    BslAdvectionPolar advection_operator(interpolator, find_feet, mapping);
    BslAdvectionVelocity<GeometryVrVthetaRTheta, SplineInterpolatorVr, double> const advection_vr(interpolator_vr);
    BslAdvectionVelocity<GeometryVrVthetaRTheta, SplineInterpolatorVtheta, double> const advection_vtheta(interpolator_vtheta);
    SplitVlasovSolver const vlasov(advection_operator, advection_vr, advection_vtheta);
    FieldMemVrVtheta<double> const quadrature_coeffs(
            quadrature_coeffs_nd<Kokkos::DefaultExecutionSpace, double, GridVr, GridVtheta>(
                    idxrange_vrvtheta,
                    std::
                            bind(simpson_trapezoid_quadrature_coefficients_1d<
                                         Kokkos::DefaultExecutionSpace,
                                         GridVr,
                                         double>,
                                 std::placeholders::_1,
                                 Extremity::FRONT),
                    std::
                            bind(simpson_trapezoid_quadrature_coefficients_1d<
                                         Kokkos::DefaultExecutionSpace,
                                         GridVtheta,
                                         double>,
                                 std::placeholders::_1,
                                 Extremity::FRONT)));
    DFieldMemRTheta coeff_alpha(idxrange_rtheta);
    DFieldMemRTheta coeff_beta(idxrange_rtheta);
    ddc::parallel_fill(coeff_alpha, -1);
    ddc::parallel_fill(coeff_beta, 0);
    PoissonSolver poisson_solver(discrete_mapping, interpolator);
    poisson_solver.update_coefficients(get_const_field(coeff_alpha), get_const_field(coeff_beta));
    ChargeDensityCalculator const rhs(get_const_field(quadrature_coeffs));
    QNSolver poisson(poisson_solver, rhs);
    PredCorr const predcorr(vlasov, poisson);

    // --- expose simulation metadata to PDI ---
    ddc::expose_to_pdi("Nr_spline_cells", ddc::discrete_space<BSplinesR>().ncells());
    ddc::expose_to_pdi("Ntheta_spline_cells", ddc::discrete_space<BSplinesTheta>().ncells());
    ddc::expose_to_pdi("Nvr_spline_cells", ddc::discrete_space<BSplinesVr>().ncells());
    ddc::expose_to_pdi("Nvtheta_spline_cells", ddc::discrete_space<BSplinesVtheta>().ncells());
    expose_mesh_to_pdi("MeshR", mesh_r);
    expose_mesh_to_pdi("MeshTheta", mesh_theta);
    expose_mesh_to_pdi("MeshVr", mesh_vr);
    expose_mesh_to_pdi("MeshVtheta", mesh_vtheta);
    ddc::expose_to_pdi("Nkinspecies", idxrange_kinsp.size());
    ddc::expose_to_pdi(
            "fdistribu_charges",
            ddc::discrete_space<Species>().charges()[idxrange_kinsp]);
    ddc::expose_to_pdi(
            "fdistribu_masses",
            ddc::discrete_space<Species>().masses()[idxrange_kinsp]);
    ddc::expose_to_pdi("deltat", deltat);
    ddc::expose_to_pdi("nbiter", nbiter);
    ddc::expose_to_pdi("time_step_diag", time_step_diag);

    // --- write init state HDF5 ---
    ddc::PdiEvent("initialisation");

    // --- run simulation ---
    std::chrono::steady_clock::time_point const start = std::chrono::steady_clock::now();
    predcorr(get_field(allfdistribu), deltat, nbiter);
    std::chrono::steady_clock::time_point const end = std::chrono::steady_clock::now();
    Real const simulation_time = std::chrono::duration<Real>(end - start).count();
    std::cout << "Simulation time: " << simulation_time << "s\n";

    PDI_finalize();

    PC_tree_destroy(&conf_pdi);
    PC_tree_destroy(&conf_gyselalibxx);
    return EXIT_SUCCESS;
}