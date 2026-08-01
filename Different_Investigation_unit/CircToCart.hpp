#include <cmath>
#include <random>

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include <gtest/gtest.h>

#include "circular_to_cartesian.hpp"
#include "discrete_poloidal_cs_spline_mapping.hpp"
#include "discrete_poloidal_cs_spline_mapping_builder.hpp"
#include "polar_bsplines.hpp"
#include "test_utils.hpp"
#include "view.hpp"

namespace Test{
struct R_cov;
struct Theta_cov;
struct R{
    static bool constexpr PERIODIC = false;
    static bool constexpr IS_COVARIANT = false;
    static bool constexpr IS_CONTRAVARIANT = true;
    using Dual = R_cov;
};
struct Theta{
    static bool constexpr PERIODIC = false;
    static bool constexpr IS_COVARIANT = false;
    static bool constexpr IS_CONTRAVARIANT = true;
    using Dual = Theta_cov;
};
struct Theta_cov{
    static bool constexpr PERIODIC = true;
    static bool constexpr IS_COVARIANT = true;
    static bool constexpr IS_CONTRAVARIANT = false;
    using Dual = Theta;
};
struct X{
    static bool constexpr IS_CONTRAVARIANT = true;
    static bool constexpr IS_COVARIANT = true;
    using Dual = X;
};
struct Y{
    static bool constexpr IS_CONTRAVARIANT = true;
    static bool IS_COVARIANT = true;
    using Dual = Y;
};
struct BSplinesR : ddc::NonUniformBSplines<R,3>{};
struct BSplinesTheta : ddc::NonUniformBSplines<Theta, 3>{};
using GrevillePointsR = ddc::GrevilleInterpolationPoints<BSplinesR, ddc::SplineBuilderClosure::GREVILLE,ddc::SplineBuilderClosure::GREVILLE>;
using GrevillePointsTheta =  ddc::GrevilleInterpolationPoints<BSplinesTheta,ddc::SplineBuilderClosure::PERIODIC,ddc::GrevilleInterpolationPoints::PERIODIC>;
struct GridR : GrevillePointsR::interpolation_discrete_dimension_type
{};
struct GridTheta : GrevillePointsTheta::interpolation_discrete_dimension_type
{};
struct BSplines : PolarBSplines<BSplinesR, BSplinesTheta, 1>
TEST(Basic_structure, Polar_splines){
    using IdxStepR = IdxStep<GridR>;
    using IdxSterTheta = IdxStep<GridTheta>;
    using PolarCoord = Coord<R,Theta>;
    using CircToCartesian = CircularToCartesian<R, Theta, X, Y>
    using SplinesRThetaBuilder = ddc::SplineBuilder2D<
        Kokkos::DefaultHostExecutionSpace,
        Kokkos::DefaultHostExecutionSpace::memory_space,
        BSplinesR,
        BSplinesTheta,
        GridR,
        GridTheta,
        ddc::SplineBuilderClosure::GREVILLE,
        ddc::SplineBuilderClosure::GREVILLE,
    >
}
}