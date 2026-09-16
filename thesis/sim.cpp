#include "Geometry_thesis.hpp"
#include "spline_.hpp"

namespace {
using DiscreteMappingBuilder
        = DiscretePoloidalCSSplineMappingBuilder<X, Y, SplineInterpolatorRThetaConst>;
using PoissonSolver = PolarSplineFEMPoissonLikeSolver<
        GridR,
        GridTheta,
        PolarBSplinesRTheta,
        SplineInterpolatorRTheta,
        typename DiscreteMappingBuilder::MappingType>;
using Mapping = CzarnyToCartesian<R, Theta, X, Y>;
namespace fs = std::filesystem;

}

int main(int argc, char** argv)
{
    fs::create_directory("output");

    PC_tree_t conf_gyselalibxx = parse_executable_arguments(argc, argv, params_yaml);
    PC_tree_t conf_pdi = PC_parse_string(PDI_CFG);
    PC_errhandler(PC_NULL_HANDLER);
    PDI_init(conf_pdi);

    Kokkos::ScopeGuard kokkos_scope(argc, argv);
    ddc::ScopeGuard ddc_scope(argc, argv);

    std::chrono::time_point<std::chrono::system_clock> start_simulation;
    std::chrono::time_point<std::chrono::system_clock> end_simulation;

    start_simulation = std::chrono::system_clock::now();

    IdxRangeR const mesh_r = init_pseudo_uniform_spline_dependent_idx_range<
            GridR,
            BSplinesR,
            SplineInterpPointsR>(conf_gyselalibxx, "r");
    IdxRangeTheta const mesh_theta = init_pseudo_uniform_spline_dependent_idx_range<
            GridTheta,
            BSplinesTheta,
            SplineInterpPointsTheta>(conf_gyselalibxx, "theta");
    double const dt(PCpp_double(conf_gyselalibxx, ".Time.delta_t"));
    double const final_T(PCpp_double(conf_gyselalibxx, ".Time.final_T"));

    IdxRangeRTheta const mesh_rtheta(mesh_r, mesh_theta);

    host_t<FieldMemRTheta<CoordRTheta>> coords(mesh_rtheta);
    ddc::host_for_each(mesh_rtheta, [&](IdxRTheta const irtheta) {
        coords(irtheta) = ddc::coordinate(irtheta);
    });
    SplineInterpolatorRTheta interpolator(mesh_rtheta);
    SplineInterpolatorRThetaConst interpolator_const(mesh_rtheta);
    SplineRThetaBuilder const& builder(interpolator.get_builder());
    double const major_radius = PCpp_double(conf_gyselalibxx, ".CzarnyMapping.x0");
    double const vertical_offset = PCpp_double(conf_gyselalibxx, ".CzarnyMapping.y0");
    Coord<X, Y> origin_point(major_radius, vertical_offset);
    double const czarny_epsilon = PCpp_double(conf_gyselalibxx, ".CzarnyMapping.epsilon");
    double const czarny_e = PCpp_double(conf_gyselalibxx, ".CzarnyMapping.e");
    const Mapping mapping(czarny_epsilon, czarny_e, origin_point);
    SplineRThetaEvaluatorNullBound const& evaluator(interpolator.get_evaluator());

    DiscreteMappingBuilder const
            discrete_mapping_builder(Kokkos::DefaultExecutionSpace(), mapping, interpolator_const);
    DiscretePoloidalCSSplineMapping const discrete_mapping = discrete_mapping_builder();
    ddc::init_discrete_space<PolarBSplinesRTheta>(discrete_mapping);
    IdxRangeBSRTheta idx_range_bsplinesRTheta = get_spline_idx_range(builder);
    RK4Builder const time_stepper;

    PolarFootFinder find_feet = make_polar_foot_finder<
            FootFindingSpace::PSEUDO_PHYSICAL,
            AdvectionFieldSpace::
                    PHYSICAL>(time_stepper, mapping, mesh_rtheta, interpolator);
    BslAdvectionPolar advection_operator(interpolator, find_feet, mapping);
    DFieldMemRTheta coeff_alpha_alloc(mesh_rtheta); 
    DFieldMemRTheta coeff_beta_alloc(mesh_rtheta);
    DFieldRTheta coeff_alpha = get_field(coeff_alpha_alloc); 
    DFieldRTheta coeff_beta = get_field(coeff_beta_alloc);
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            mesh_rtheta,
            KOKKOS_LAMBDA(IdxRTheta const irtheta) {
                coeff_alpha(irtheta) = Kokkos::exp(
                        -Kokkos::tanh((ddc::coordinate(ddc::select<GridR>(irtheta)) - 0.7) / 0.05));
                coeff_beta(irtheta) = 1.0 / coeff_alpha(irtheta);
            });

    PoissonSolver poisson_solver(discrete_mapping, interpolator);
    poisson_solver.update_coefficients(get_const_field(coeff_alpha), get_const_field(coeff_beta));
    BslPredCorrRTheta predcorr_operator(
            mapping,
            advection_operator,
            interpolator,
            poisson_solver);
    end_simulation = std::chrono::system_clock::now();
    int const iter_nb = final_T * int(1 / dt);
    // --- save simulation data
    ddc::expose_to_pdi("r_size", ddc::discrete_space<BSplinesR>().ncells());
    ddc::expose_to_pdi("theta_size", ddc::discrete_space<BSplinesTheta>().ncells());

    expose_mesh_to_pdi("r_coords", mesh_r);
    expose_mesh_to_pdi("theta_coords", mesh_theta);

    ddc::expose_to_pdi("delta_t", dt);
    ddc::expose_to_pdi("final_T", final_T);
    ddc::expose_to_pdi("time_step_diag", PCpp_int(conf_gyselalibxx, ".Output.time_step_diag"));

    host_t<FieldMemRTheta<CoordX>> coords_x(mesh_rtheta);
    host_t<FieldMemRTheta<CoordY>> coords_y(mesh_rtheta);
    host_t<DFieldMemRTheta> jacobian(mesh_rtheta);
    ddc::host_for_each(mesh_rtheta, [&](IdxRTheta const irtheta) {
        CoordXY coords_xy = mapping(ddc::coordinate(irtheta));
        coords_x(irtheta) = ddc::select<X>(coords_xy);
        coords_y(irtheta) = ddc::select<Y>(coords_xy);
        jacobian(irtheta) = mapping.jacobian(ddc::coordinate(irtheta));
    });
    host_t<DFieldMemRTheta> rho_alloc_host(mesh_rtheta);
    host_t<DFieldMemRTheta> rho_eq_alloc_host(mesh_rtheta);

    // Initialise rho and rho equilibrium ****************************
    // 2D density in physical (x, y) space via the Czarny mapping:
    //   rho_eq  = Gaussian centered at the O-point (magnetic axis)
    //   rho_0   = rho_eq * (1 + eps * cos(l * theta))
    double const sigma_rho = PCpp_double(conf_gyselalibxx, ".Perturbation.sigma");
    double const eps_rho   = PCpp_double(conf_gyselalibxx, ".Perturbation.eps");
    int const l_mode       = PCpp_int(conf_gyselalibxx, ".Perturbation.l_mode");
    CzarnyDensitySolution<Mapping> exact_rho(mapping, sigma_rho, eps_rho, l_mode);

    ddc::host_for_each(mesh_rtheta, [&](IdxRTheta const irtheta) {
        rho_alloc_host(irtheta) = exact_rho.initialisation(coords(irtheta));
        rho_eq_alloc_host(irtheta) = exact_rho.equilibrium(coords(irtheta));
    });
    auto rho_eq_alloc = ddc::create_mirror_view_and_copy(
            Kokkos::DefaultExecutionSpace(),
            get_field(rho_eq_alloc_host));

    DFieldMemRTheta phi_eq_alloc(mesh_rtheta);
    host_t<DFieldMemRTheta> phi_eq_alloc_host(mesh_rtheta);
    Spline2DMem rho_coef_eq_alloc(idx_range_bsplinesRTheta);
    builder(get_field(rho_coef_eq_alloc), get_const_field(rho_eq_alloc));
    PoissonLikeRHSFunction
            poisson_rhs_eq(get_const_field(rho_coef_eq_alloc), interpolator.get_evaluator());
    poisson_solver(get_field(phi_eq_alloc), poisson_rhs_eq);
    ddc::parallel_deepcopy(phi_eq_alloc_host, phi_eq_alloc);
    ddc::PdiEvent("initialisation")
            .with("x_coords", coords_x)
            .with("y_coords", coords_y)
            .with("jacobian", jacobian)
            .with("density_eq", rho_eq_alloc_host)
            .with("electrical_potential_eq", phi_eq_alloc_host);
    predcorr_operator(get_field(rho_alloc_host), dt, iter_nb);

    end_simulation = std::chrono::system_clock::now();
    display_time_difference("Simulation time: ", start_simulation, end_simulation);

    PC_tree_destroy(&conf_pdi);
    PDI_finalize();
    PC_tree_destroy(&conf_gyselalibxx);

    return EXIT_SUCCESS;
}