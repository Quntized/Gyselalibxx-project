#pragma once
#include <ddc/ddc.hpp>

#include "ddc_alias_inline_functions.hpp"
#include "ddc_aliases.hpp"
#include "ddc_helper.hpp"
#include "polar_bsplines.hpp"
#include "vector_field.hpp"
#include "vector_field_mem.hpp"
#include "species_info.hpp"
#include "vector_index_tools.hpp"

struct R_cov;
struct Theta_cov;
struct R
{
    static bool constexpr PERIODIC = false;
    static bool constexpr IS_COVARIANT = false;
    static bool constexpr IS_CONTRAVARIANT = true;
    using Dual = R_cov;
};
struct Theta
{
    static bool constexpr PERIODIC = true;
    static bool constexpr IS_COVARIANT = false;
    static bool constexpr IS_CONTRAVARIANT = true;
    using Dual = Theta_cov;
};
struct R_cov
{
    static bool constexpr PERIODIC = false;
    static bool constexpr IS_COVARIANT = true;
    static bool constexpr IS_CONTRAVARIANT = false;
    using Dual = R;
};
struct Theta_cov
{
    static bool constexpr PERIODIC = true;
    static bool constexpr IS_COVARIANT = true;
    static bool constexpr IS_CONTRAVARIANT = false;
    using Dual = Theta;
};
struct Vr
{
    static bool constexpr PERIODIC = false;
};
struct Vtheta
{
    static bool constexpr PERIODIC = false;
};
using Real = double;
using CoordR = Coord<R>;
using CoordTheta = Coord<Theta>;
using CoordRTheta = Coord<R, Theta>;

using CoordVr = Coord<Vr>;
using CoordVtheta = Coord<Vtheta>;
struct GridR : NonUniformGridBase<R>
{
};
struct GridTheta : NonUniformGridBase<Theta>
{
};
struct GridVr : NonUniformGridBase<Vr>
{
};
struct GridVtheta : NonUniformGridBase<Vtheta>
{
};
using IdxR = Idx<GridR>;
using IdxTheta = Idx<GridTheta>;
using IdxRTheta = Idx<GridR, GridTheta>;
using IdxVr = Idx<GridVr>;
using IdxVtheta = Idx<GridVtheta>;
using IdxVrVtheta = Idx<GridVr, GridVtheta>;
using IdxSpVrVthetaRTheta = Idx<Species, GridVr, GridVtheta, GridR, GridTheta>;
using IdxRThetaVrVtheta = Idx<GridR, GridTheta, GridVr, GridVtheta>;

using IdxStepR = IdxStep<GridR>;
using IdxStepTheta = IdxStep<GridTheta>;
using IdxStepRTheta = IdxStep<GridR, GridTheta>;
using IdxStepVr = IdxStep<GridVr>;
using IdxStepVtheta = IdxStep<GridVtheta>;
using IdxStepVrVtheta = IdxStep<GridVr, GridVtheta>;
using IdxStepRThetaVrVtheta = IdxStep<GridR, GridTheta, GridVr, GridVtheta>;

using IdxRangeR = IdxRange<GridR>;
using IdxRangeTheta = IdxRange<GridTheta>;
using IdxRangeRTheta = IdxRange<GridR, GridTheta>;
using IdxRangeVr = IdxRange<GridVr>;
using IdxRangeVtheta = IdxRange<GridVtheta>;
using IdxRangeVrVtheta = IdxRange<GridVr, GridVtheta>;
using IdxRangeSpVrVtheta = IdxRange<Species, GridVr, GridVtheta>;
using IdxRangeSpRThetaVrVtheta = IdxRange<Species, GridR, GridTheta, GridVr, GridVtheta>;
using IdxRangeRThetaVrVtheta = IdxRange<GridR, GridTheta, GridVr, GridVtheta>;
using IdxRangeSpVrVthetaRTheta = IdxRange<Species, GridVr, GridVtheta, GridR, GridTheta>;

template <class ElementType>
using FieldMemR = FieldMem<ElementType, IdxRangeR>;

template <class ElementType>
using FieldMemTheta = FieldMem<ElementType, IdxRangeTheta>;

template <class ElementType>
using FieldMemRTheta = FieldMem<ElementType, IdxRangeRTheta>;

template <class ElementType>
using FieldMemVr = FieldMem<ElementType, IdxRangeVr>;

template <class ElementType>
using FieldMemVtheta = FieldMem<ElementType, IdxRangeVtheta>;

template <class ElementType>
using FieldMemVrVtheta = FieldMem<ElementType, IdxRangeVrVtheta>;

template <class ElementType>
using FieldMemSpVrVtheta = FieldMem<ElementType, IdxRangeSpVrVtheta>;

template <class ElementType>
using FieldMemRThetaVrVtheta = FieldMem<ElementType, IdxRangeRThetaVrVtheta>;

template <class ElementType>
using FieldMemSpVrVthetaRTheta = FieldMem<ElementType, IdxRangeSpVrVthetaRTheta>;

template <class ElementType>
using FieldMemSpRThetaVrVtheta = FieldMem<ElementType, IdxRangeSpRThetaVrVtheta>;


using DFieldMemR = FieldMemR<double>;
using DFieldMemTheta = FieldMemTheta<double>;
using DFieldMemRTheta = FieldMemRTheta<double>;
using DFieldMemVr = FieldMemVr<double>;
using DFieldMemVtheta = FieldMemVtheta<double>;
using DFieldMemVrVtheta = FieldMemVrVtheta<double>;
using DFieldMemRThetaVrVtheta = FieldMemRThetaVrVtheta<double>;
using DFieldMemSpVrVtheta = FieldMemSpVrVtheta<double>;
using DFieldMemSpVrVthetaRTheta = FieldMemSpVrVthetaRTheta<double>;
using DFieldMemSpRThetaVrVtheta = FieldMemSpRThetaVrVtheta<double>;


template <class ElementType>
using FieldR = Field<ElementType, IdxRangeR>;

template <class ElementType>
using FieldTheta = Field<ElementType, IdxRangeTheta>;


template <class ElementType>
using FieldRTheta = Field<ElementType, IdxRangeRTheta>;

template <class ElementType>
using FieldVr = Field<ElementType, IdxRangeVr>;

template <class ElementType>
using FieldVtheta = Field<ElementType, IdxRangeVtheta>;

template <class ElementType>
using FieldVrVtheta = Field<ElementType, IdxRangeVrVtheta>;

template <class ElementType>
using FieldRThetaVrVtheta = Field<ElementType, IdxRangeRThetaVrVtheta>;

template <class ElementType>
using FieldSpVrVtheta = Field<ElementType, IdxRangeSpVrVtheta>;

template <class ElementType>
using FieldSpVrVthetaRTheta = Field<ElementType, IdxRangeSpVrVthetaRTheta>;

template <class ElementType>
using FieldSpRThetaVrVtheta = Field<ElementType, IdxRangeSpRThetaVrVtheta>;

using DFieldR = FieldR<double>;
using DFieldTheta = FieldTheta<double>;
using DFieldRTheta = FieldRTheta<double>;
using DFieldVr = FieldVr<double>;
using DFieldVtheta = FieldVtheta<double>;
using DFieldVrVtheta = FieldVrVtheta<double>;
using DFieldRThetaVrVtheta = FieldRThetaVrVtheta<double>;
using DFieldSpVrVtheta = FieldSpVrVtheta<double>;
using DFieldSpVrVthetaRTheta = FieldSpVrVthetaRTheta<double>;
using DFieldSpRThetaVrVtheta = FieldSpRThetaVrVtheta<double>;

template <class ElementType>
using ConstFieldR = ConstField<ElementType, IdxRangeR>;

template <class ElementType>
using ConstFieldTheta = ConstField<ElementType, IdxRangeTheta>;

template <class ElementType>
using ConstFieldRTheta = ConstField<ElementType, IdxRangeRTheta>;

template <class ElementType>
using ConstFieldVr = ConstField<ElementType, IdxRangeVr>;

template <class ElementType>
using ConstFieldVtheta = ConstField<ElementType, IdxRangeVtheta>;

template <class ElementType>
using ConstFieldVrVtheta = ConstField<ElementType, IdxRangeVrVtheta>;

template <class ElementType>
using ConstFieldRThetaVrVtheta = ConstField<ElementType, IdxRangeRThetaVrVtheta>;

template <class ElementType>
using ConstFieldSpVrVtheta = ConstField<ElementType, IdxRangeSpVrVtheta>;

template <class ElementType>
using ConstFieldSpVrVthetaRTheta = ConstField<ElementType, IdxRangeSpVrVthetaRTheta>;

template <class ElementType>
using ConstFieldSpRThetaVrVtheta = ConstField<ElementType, IdxRangeSpRThetaVrVtheta>;

using DConstFieldR = ConstFieldR<double>;
using DConstFieldTheta = ConstFieldTheta<double>;
using DConstFieldRTheta = ConstFieldRTheta<double>;
using DConstFieldVr = ConstFieldVr<double>;
using DConstFieldVtheta = ConstFieldVtheta<double>;
using DConstFieldVrVtheta = ConstFieldVrVtheta<double>;
using DConstFieldRThetaVrVtheta = ConstFieldRThetaVrVtheta<double>;
using DConstFieldSpVrVtheta = ConstFieldSpVrVtheta<double>;
using DConstFieldSpVrVthetaRTheta = ConstFieldSpVrVthetaRTheta<double>;
using DConstFieldSpRThetaVrVtheta = ConstFieldSpRThetaVrVtheta<double>;

template <class Dim1, class Dim2>
using DVectorFieldMemRTheta = VectorFieldMem<double, IdxRangeRTheta, VectorIndexSet<Dim1, Dim2>>;

template <class Dim1, class Dim2>
using DVectorFieldRTheta = VectorField<double, IdxRangeRTheta, VectorIndexSet<Dim1, Dim2>>;

template <class Dim1, class Dim2>
using DVectorConstFieldRTheta
    = VectorConstField<double, IdxRangeRTheta, VectorIndexSet<Dim1, Dim2>>;

struct X
{
    static bool constexpr PERIODIC = false;
    static bool constexpr IS_COVARIANT = true;
    static bool constexpr IS_CONTRAVARIANT = true;
    using Dual = X;
};

struct Y
{
    static bool constexpr PERIODIC = false;
    static bool constexpr IS_COVARIANT = true;
    static bool constexpr IS_CONTRAVARIANT = true;
    using Dual = Y;
};
struct Vx
{
    static bool constexpr PERIODIC = false;
};
struct Vy
{
    static bool constexpr PERIODIC = false;
};

using CoordX = Coord<X>;
using CoordY = Coord<Y>;
using CoordXY = Coord<X, Y>;

using CoordVx = Coord<Vx>;
using CoordVy = Coord<Vy>;
using CoordVxVy = Coord<Vx, Vy>;
using CoordXYVxVy = Coord<X, Y, Vx, Vy>;

class GeometryRThetaVrVtheta
{
public:
    template <class T>
    using velocity_dim_for = std::conditional_t<
            std::is_same_v<T, GridR>,
            GridVr,
            std::conditional_t<std::is_same_v<T, GridTheta>, GridVtheta, void>>;
    using IdxRangeSpatial = IdxRangeRTheta;
    using IdxRangeVelocity = IdxRangeVrVtheta;
    using IdxRangeFdistribu = IdxRangeSpRThetaVrVtheta;
};
class GeometryVrVthetaRTheta
{
public:
    template <class T>
    using velocity_dim_for = std::conditional_t<
            std::is_same_v<T, GridR>,
            GridVr,
            std::conditional_t<std::is_same_v<T, GridTheta>, GridVtheta, void>>;
    using IdxRangeSpatial = IdxRangeRTheta;
    using IdxRangeVelocity = IdxRangeVrVtheta;
    using IdxRangeFdistribu = IdxRangeSpVrVthetaRTheta;
};