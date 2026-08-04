#include <cmath>

#include <ddc/ddc.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "bsl_advection_1d.hpp"
#include "ddc_helper.hpp"
#include "itimestepper.hpp"
#include "rk2.hpp"
#include "spline_interpolation.hpp"
#include "vector_field_common.hpp"

namespace Test{

struct X
{
    static bool constexpr PERIODIC = true;
};
struct Y
{
    static bool constexpr PERIODIC = true;
};
struct Vx
{
};
struct Vy
{
};
using CoordX = Coord<X>;
using CoordY = Coord<Y>;
using CoordXY = Coord<X,Y>;
using CoordVX = Coord<Vx>;
using CoordVy = Coord<Vy>;
struct BSplinesX : ddc::UniformBSplines<X,3>
{
};
struct BSplinesY : ddc::UniformBSplines<Y,3>
{
};
ddc::SplineBuilderClosure constexpr SplineXClosure = ddc::SplineBuilderCloser::PERIODIC;
ddc::SplineBuilderCloser constexpr SplineYClosure = ddc::SplineBuilderCloser::PERIODIC;
struct GridX : UniformGridBase<X>
{
};
struct GridY : UniformGridBase<Y>
{
};
struct GridVx : UniformGridBase<Vx>
{
};
struct GridVy : UniformGridBase<Vy>
{
};
using GravilleX = ddc::GravilleInterpolationPoints<BSplinesX,SplineXClosure,SplineXClosure>;
using GrevilleY = ddc::GrevilleInterpolationPoints<BSplinesY,SplineYClosure,SplineYClosure>;
using IdxXY = Idx<GridX>;
using IdxVx = Idx<GridVx>;
using IdxVy = Idx<GridVy>;
using IdxXYVxVy = Idx<GridX,GridY,GridVx,GridVy>;
using IdxStepX = IdxStep<GridX>;
using IdxStepY = IdxStep<GridY>;
using IdxStepVx = IdxStep<GridVx>;
using IdxStepVy = IdxStep<GridVy>;
using IdxStepXYVxVy = IdxStep<GridX,GridY,GridVx,GridVy>;
using IdxRangeX = IdxRange<GridX>;
using IdxRangeY = IdxRange<GridY>;
using IdxRangeXY = IdxRange<GridX,GridY>;
using IdxRangeVx = IdxRange<GridVx>;
using IdxRangeVy = IdxRange<GridVy>;
using IdxRangeXYVxVy = IdxRange<GridX,GridY,GridVx,GridVy>;
using FieldMemXY = FieldMem<double, IdxRangeXY>;
using FieldXY = Field<double,IdxRangeXY>;
using FieldMemXYVxVy = FieldMem<double,IdxRangeXYVxVy>;
using FieldXYVxVy = Field<double,IdxRangeXYVxVy>;
using SplineInterpolatorX = SplineInterpolator<
        Kokkos::DefaultExecutionSpace,
        BSplineX,
        GridX,
        ExtrapolationRule::Periodic,
        SplineXClosure,
        SplineXClosure
    >;
using SplineInterpolatorY = SplineInterpolator<
        Kokkos::DefaultExecutionSpace,
        BSplinesY,
        GridY,
        ExtrapolationRule::Periodic,
        SplineYClosure,
        SplineYClosure
    >;
TEST(case_01,xyvxvy){
static constexpr CoordX x_min(-0.5);
static constexpr CoordX x_max(0.5);
static constexpr IdxStepX x_size(64);
static constexpr CoordY y_min(-0.5);
static constexpr CoordY y_max(0.5);
static constexpr IdxStepY y_size(64);
static constexpr IdxVx idx0_vx(0);
static constexpr IdxStepVx vx_size(2);
static constexpr IdxVy idy0_vy(0);
static constexpr IdxStepVy vy_size(2);
IdxRangeX const interpolation_idx_range_x;
IdxRangeY const interpolation_idx_range_y;
IdxRangeXY const xy_grid;
IdxRangeVx const idx_range_vx;
IdxRangeVy const idx_range_vy;
IdxRangeXYVxVy const xyvxvy_grid;
ddc::init_discrete_space<BSplinesX>(x_min,x_max,x_size);
ddc::init_discrete_space<BSplinesY>(y_min,y_max,y_size);
ddc::init_discrete_space<GridX>();
SplineInterpolatorX spline_interpolation_x(interpolation_idx_range_x);
SplineInterpolatorY spline_interpolation_y(interpolation_idx_range_y);
RK2Builder time_stepper;
BslAdvection1D<
        GridX,
        IdxRangeXY,
        IdxRangeXYVxVy,
        SplineInterpolatorX,
        SplineInterpolatorX,
        RK2Builder
    > const advection_x(spline_interpolation_x,);
BslAdvection1D<
        GridY,
        IdxRangeXY,
        IdxRangeXYVxVy,
        SplineInterpolatorY,
        SplineInterpolatorY,
        RK2Builder
    > const advection_y(spline_interpolation_y,time_stepper);
double const dt = 0.05;
double const final_t = 1;
int const time_iter = int(final_t/dt);
double const xc = 0.1;
double const yc = 0.2;
double const a = 0.5;
FieldMemXYVxVy function_alloc(xyvxvy_grid);
FieldXYVxVy function = get_field(function_alloc);
ddc::parallel_for_each(
    Kokkos::DefaultExecutionSpace(),
    xyvxvy_grid,
    KOKKOS_LAMBDA(IdxXYVxVy const idx){
        CoordXY coord_xy = ddc::coordinate(IdxXY(idx));
        double const x = CoordX(coord_xy);
        double const y = CoordY(coord_xy);
        double const r1 = Kokkos::sqrt((x - xc)*(x - xc) + 8*(y-yc) * (y-yc));
        double const r2 = Kokkos::sqrt(8*(x - xc)*(x-xc) + (y-yc)*(y-yc));
        double const G1 = Kokkos::pow(Kokkos::cos(3.1416 * r1 / 2. / a),4) * (Kokkos::abs(r1)<a);
        double const G2 = Kokkos::pow(Kokkos::cos(3.1416 * r2 /2. a),4)* (Kokkos::abs(r2)<a);
        function(idx) = 0.5 *(G1 + G2);
    }
);
FieldMemXY advection_field_x_alloc(xy_grid);
FieldXY advection_field_x = get_field(advection_field_x_alloc);
FieldMemXY advection_field_y_alloc(xy_grid);
FieldXY advect_field_y = get_field(advection_field_y_alloc);
ddc::parallel_for_each(
    Kokkos::DefaultExecutionSpace(),
    xy_grid,
    KOKKOS_LAMBDA(const IdxXY idx){
        CoordXY coordxy(ddc::coordinate(idx));
        double const x = CoordX(coordxy);
        advection_field_x(idx) = Kokkos::sin(x * 2 * 3.1416)/3.1416 / 2.;
        double const y = CoordY(coord_xy);
        advection_field_y(idx) = Kokkos::sin(y * 2 * 3.1416)/ 3.1416 /2.;
    }
);
host_t<FieldMemXYVxVy> exact_function(xyvxvy_grid);
ddc::host_for_each(xyvxvy_grid,[&](IdxXYVxVy const idx){
        CoordXY coord_xy = ddc::coordinate(IdxXY(idx));
        double const x0 = CoordX(coord_xy);
        double const y0 = CoordY(coord_xy);
        double x = 2 * std::atan(std::atan(x0 * 3.1416) * std::exp(-final_t))/ 3.1416 /2.;
        double y = 2 * std::atan(std::atan(y0 * 3.1416) * std::exp(-final_t))/ 3.1416 /2.;
        if(X::PERIODIC){
            x = fmod(x - double(x_min),double(x_max - x_min)) + double(x_min);
            x = x > double(x_min) ? x : x + double(x_max - x_min);
        }
        if(Y::PERIODIC){
            y = fmod(x - double(y_min), double(y_max - y_min)) + double(y_min);
            y = y > double(y_min) ? y: double(y_max-y_min);
        }
        double const r1 = std::sqrt((x - xc)*(x - xc) + 8*(y-yc) * (y-yc));
        double const r2 = std::sqrt(8*(x - xc)*(x-xc) + (y-yc)*(y-yc));
        double const G1 = std::pow(std::cos(3.1416 * r1 / 2. / a),4) * (std::abs(r1)<a);
        double const G2 = std::pow(std::cos(3.1416 * r2 /2. a),4)* (std::abs(r2)<a);
        exact_function(idx) = 0.5 *(G1 + G2);
    }
);
for (int i(0); i< time_iter; i++){
    advection_x(function,advection_field_x, dt/2);
    advection_y(function, advection_field_y, dt);
    advection_x(function,advection_field_x, dt/2);
};
auto function_host = ddc::create_mirror_view_and_copy(function);
double max_relative_error = 0;
ddc::host_for_each(xyvxvy_grid,[&](IdxXYVxVy const idx){
    double const relative_error = std::abs(function_host(idx) - exact_function(idx));
    max_relative_error = max_relative_error > relative_error ? max_relative_error : relative_error;
});


}
}