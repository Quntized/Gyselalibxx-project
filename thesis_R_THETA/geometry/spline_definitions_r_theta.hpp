#pragma once
#include <ddc/kernels/splines.hpp>

#include "geometry_rthetavrvtheta.hpp"
#include "spline_interpolation.hpp"


int constexpr BSDegreeR = 3;
int constexpr BSDegreeTheta = 3;

bool constexpr BsplineOnUniformCellsR = false;
bool constexpr BsplineOnUniformCellsTheta = false;
bool constexpr BsplineOnUniformCellsVr = false;
bool constexpr BsplineOnUniformCellsVtheta = false;

struct BSplinesR
    : std::conditional_t<
              BsplineOnUniformCellsR,
              ddc::UniformBSplines<R, BSDegreeR>,
              ddc::NonUniformBSplines<R, BSDegreeR>>
{
};

struct BSplinesTheta
    : std::conditional_t<
              BsplineOnUniformCellsTheta,
              ddc::UniformBSplines<Theta, BSDegreeTheta>,
              ddc::NonUniformBSplines<Theta, BSDegreeTheta>>
{
};
struct BSplinesVr
    : std::conditional_t<
              BsplineOnUniformCellsVr,
              ddc::UniformBSplines<Vr, BSDegreeR>,
              ddc::NonUniformBSplines<Vr, BSDegreeR>>
{
};
struct BSplinesVtheta
    : std::conditional_t<
              BsplineOnUniformCellsVtheta,
              ddc::UniformBSplines<Vtheta, BSDegreeTheta>,
              ddc::NonUniformBSplines<Vtheta, BSDegreeTheta>>
{
};
struct PolarBSplinesRTheta : PolarBSplines<BSplinesR, BSplinesTheta, 1>
{
        enum : int { continuity = 1 };
};
ddc::SplineBuilderClosure constexpr SplineRClosure = ddc::SplineBuilderClosure::GREVILLE;
ddc::SplineBuilderClosure constexpr SplineThetaClosure = ddc::SplineBuilderClosure::PERIODIC;
ddc::SplineBuilderClosure constexpr SplineVrClosure
        = ddc::SplineBuilderClosure::HOMOGENEOUS_HERMITE;
ddc::SplineBuilderClosure constexpr SplineVthetaClosure
        = ddc::SplineBuilderClosure::HOMOGENEOUS_HERMITE;

using SplineInterpPointsR
        = ddc::GrevilleInterpolationPoints<BSplinesR, SplineRClosure, SplineRClosure>;
using SplineInterpPointsTheta
        = ddc::GrevilleInterpolationPoints<BSplinesTheta, SplineThetaClosure, SplineThetaClosure>;
using SplineInterpPointsVr
        = ddc::GrevilleInterpolationPoints<BSplinesVr, SplineVrClosure, SplineVrClosure>;
using SplineInterpPointsVtheta
        = ddc::GrevilleInterpolationPoints<BSplinesVtheta, SplineVthetaClosure, SplineVthetaClosure>;

using SplineRThetaEvaluatorNullBound_host = ddc::SplineEvaluator2D<
        Kokkos::DefaultHostExecutionSpace,
        Kokkos::HostSpace,
        BSplinesR,
        BSplinesTheta,
        GridR,
        GridTheta,
        ddc::NullExtrapolationRule, // boundary at r=0
        ddc::NullExtrapolationRule, // boundary at rmax
        ddc::PeriodicExtrapolationRule<Theta>,
        ddc::PeriodicExtrapolationRule<Theta>>;
using SplineInterpolatorRThetaConst = SplineInterpolator<
        Kokkos::DefaultExecutionSpace,
        IdxRange<BSplinesR, BSplinesTheta>,
        IdxRangeRTheta,
        ExtrapolationRule::Constant_Constant, // radial extrapolation
        ExtrapolationRule::Periodic, // poloidal extrapolation
        SplineBoundaryClosures<
                SplineRClosure, // boundary at r=0
                SplineRClosure>, // boundary at rmax
        SplineBoundaryClosures<SplineThetaClosure, SplineThetaClosure>>;

using SplineInterpolatorRTheta = SplineInterpolator<
        Kokkos::DefaultExecutionSpace,
        IdxRange<BSplinesR, BSplinesTheta>,
        IdxRangeRTheta,
        ExtrapolationRule::Null_Null,
        ExtrapolationRule::Periodic,
        SplineBoundaryClosures<SplineRClosure, SplineRClosure>,
        SplineBoundaryClosures<SplineThetaClosure, SplineThetaClosure>>;
using SplineRThetaBuilder = typename SplineInterpolatorRTheta::BuilderType;
using SplineRThetaEvaluatorNullBound = typename SplineInterpolatorRTheta::EvaluatorType;


using SplineInterpolatorVr = SplineInterpolator<
        Kokkos::DefaultExecutionSpace,
        IdxRange<BSplinesVr>,
        IdxRange<GridVr>,
        ExtrapolationRule::Constant_Constant,
        SplineBoundaryClosures<SplineVrClosure, SplineVrClosure>>;
using SplineInterpolatorVtheta = SplineInterpolator<
        Kokkos::DefaultExecutionSpace,
        IdxRange<BSplinesVtheta>,
        IdxRange<GridVtheta>,
        ExtrapolationRule::Constant_Constant,
        SplineBoundaryClosures<SplineVthetaClosure, SplineVthetaClosure>>;

using IdxRangeBSr = IdxRange<BSplinesR>;
using IdxRangeBSRTheta = IdxRange<BSplinesR, BSplinesTheta>;
using IdxRangeBSVtheta = IdxRange<BSplinesVtheta>;
using IdxRangeBSVrVtheta = IdxRange<BSplinesVr, BSplinesVtheta>;

template <class ElementType>
using BSConstFieldrtheta = Field<ElementType const, IdxRangeBSRTheta>;
using DBSConstFieldrtheta = BSConstFieldrtheta<double>;