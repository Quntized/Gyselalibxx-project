#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <ddc/ddc.hpp>
#include <ddc/pdi.hpp>

#include <paraconf.h>
#include <pdi.h>

#include "bsl_advection_polar.hpp"
#include "bsl_predcorr.hpp"
#include "bsl_predcorr_second_order_explicit.hpp"
#include "bsl_predcorr_second_order_implicit.hpp"
#include "cartesian_to_circular.hpp"
#include "circular_to_cartesian.hpp"
#include "crank_nicolson.hpp"
#include "ddc_alias_inline_functions.hpp"
#include "czarny_density_solution.hpp"
#include "czarny_to_cartesian.hpp"
#include "discrete_poloidal_cs_spline_mapping.hpp"
#include "discrete_poloidal_cs_spline_mapping_builder.hpp"
#include "euler.hpp"
#include "geometry_r_theta.hpp"
#include "input.hpp"
#include "l_norm_tools.hpp"
#include "output.hpp"
#include "paraconfpp.hpp"
#include "params.yaml.hpp"
#include "pdi_out.yml.hpp"
#include "poisson_like_rhs_function.hpp"
#include "polar_foot_finder.hpp"
#include "polar_spline_fem_poisson_like_solver.hpp"
#include "quadrature.hpp"
#include "rk3.hpp"
#include "rk4.hpp"
#include "simulation_utils_tools.hpp"
#include "spline_definitions_r_theta.hpp"
#include "spline_quadrature.hpp"
#include "trapezoid_quadrature.hpp"


namespace fs = std::filesystem;
using CircularToCartMapping = CircularToCartesian<R, Theta, X, Y>;
using CartToCircularMapping = CartesianToCircular<X, Y, R, Theta>;
using DiscreteMappingBuilder = DiscretePoloidalCSSplineMappingBuilder<X, Y, SplineInterpolatorRThetaConst>;
