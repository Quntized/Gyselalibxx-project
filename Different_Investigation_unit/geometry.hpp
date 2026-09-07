#pragma once 
#include <ddc/ddc.hpp>
#include "ddc_alias_inline_functions.hpp"
#include "ddc_aliases.hpp"
#include "ddc_helper.hpp"
#include "polar_bsplines.hpp"
#include "vector_field.hpp"
#include "vector_field_mem.hpp"
#include "vector_index_tools.hpp"

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
    static bool constexpr PERIODIC = true;
    static bool constexpr IS_COVARIANT= false;
    static bool constexpr IS_CONTRAVARIANT = true;
    using Dual = Theta_cov;
};
struct R_cov{
    static bool constexpr PERIODIC = false;
    static bool constexpr IS_COVARIANT = true;
    static bool constexpr IS_CONTRAVARIANT = false;
    using Dual = R;
};
struct Theta_cov{
    static bool constexpr PERIODIC = true;
    static bool constexpr IS_COVARIANT = true;
    static bool constexpr IS_CONTRAVARIANT= false;
    using Dual = Theta;
};
struct Vr{
    static bool constexpr PERIODIC=false;
};
struct Vtheta{
    static bool constexpr PERIODIC = true;
};
using CoordR = Coord<R>;
using CoordRTheta = Coord<R, Theta>;
using Coordtheta = Coord<Theta>;
using CoordVr = Coord<Vr>;
using CoordVtheta = Coord<Vtheta>;
using CoordVrVtheta = Coord<Vr, Vtheta>;
struct GridR : NonUniformGridBase<R> {};
struct GridTheta : NonUniformGridBase<Theta> {};
using IdxR = Idx<GridR>;
using IdxTheta = Idx<GridTheta>;
using IdxRTheta = Idx<GridR,GridTheta>;
using IdxStepR = IdxStep<GridR>;
using IdxStepTheta = IdxStep<GridTheta>;
using IdxStepRTheta = IdxStep<GridR,GridTheta>;
using IdxRangeR = IdxRange<GridR>;
using IdxRangeTheta = IdxRange<GridTheta>;
using IdxRangeRTheta = IdxRange<GridR,GridTheta>;
using FieldMemR = FieldMem<double, IdxRangeR>;
using FieldMemTheta = FieldMem<double,IdxRangeTheta>;
using FieldMemRTheta = FieldMem<double,IdxRangeRTheta>;
using FieldR = Field<double,IdxRangeR>;
using FieldTheta = Field<double,IdxRangeTheta>;
using FieldRTheta = Field<double,IdxRangeRTheta>;
using ConstFieldR = ConstField<double,IdxRangeR>;
using ConstFieldTheta = ConstField<double,IdxRangeTheta>;
using ConstFieldRTheta = ConstField<double,IdxRangeRTheta>;

}