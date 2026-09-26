
#pragma once

#include "geometry_rthetavrvtheta.hpp"
#include "iadvectionvx.hpp"
#include "iadvectionx.hpp"
#include "ivlasovsolver.hpp"

/**
 * @brief A class that solves a Vlasov equation using Strang's splitting.
 *
 * The Vlasov equation is split between four advection equations 
 * along the X, Y, Vx and Vy directions. The splitting involves solving 
 * the advections in the X, Y, and Vx directions first on a time interval
 * of length dt/2, then the Vy-direction advection on a time dt, and
 * finally the X, Y, and Vx directions again in reverse order on dt/2.
 */
template <class AdvectionPolar, class AdvectionVr, class AdvectionVtheta>
class SplitVlasovSolver : public IVlasovSolver
{
    /// Advection operator in the r-theta direction
    AdvectionPolar const& m_advec_rtheta;
    /// Advection operator in the vr direction
    AdvectionVr const& m_advec_vr;
    /// Advection operator in the vtheta direction
    AdvectionVtheta const& m_advec_vtheta;

public:
    /**
     * @brief Creates an instance of the split vlasov solver class.
     * @param[in] advec_rtheta An advection operator along the r theta direction.
     * @param[in] advec_vr An advection operator along the vr direction.
     * @param[in] advec_vtheta An advection operator along the vtheta direction.
     */
    SplitVlasovSolver(
            AdvectionPolar const& advec_rtheta,
            AdvectionVr const& advec_vr,
            AdvectionVtheta const& advec_vtheta)
        : m_advec_rtheta(advec_rtheta)
        , m_advec_vr(advec_vr)
        , m_advec_vtheta(advec_vtheta)
    {
    }

    ~SplitVlasovSolver() override = default;

    /**
     * @brief Solves a Vlasov equation on a timestep dt.
     *
     * @param[in, out] allfdistribu On input : the initial value of the distribution function.
     *                              On output : the value of the distribution function after solving 
     *                              the Vlasov equation.
     * @param[in] electric_field The electric field computed at all spatial positions.
     * @param[in] dt The timestep. 
     *
     * @return The distribution function after solving the Vlasov equation.
     */
    DFieldSpVrVthetaRTheta operator()(
            DFieldSpVrVthetaRTheta allfdistribu,
            DVectorConstFieldRTheta<X, Y> electric_field,
            Real dt) const override
    {
        using FullAdvectionField = VectorFieldMem<
            double,
            IdxRangeSpVrVthetaRTheta,
            VectorIndexSet<X, Y>>;
        FullAdvectionField advection_field_alloc(get_idx_range(allfdistribu));
        auto advection_field = get_field(advection_field_alloc);
        ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            get_idx_range(allfdistribu),
            KOKKOS_LAMBDA(IdxSpVrVthetaRTheta const idx) {
                IdxRTheta const spatial_idx(idx);
                ddcHelper::get<X>(advection_field)(idx)
                    = ddcHelper::get<X>(electric_field)(spatial_idx);
                ddcHelper::get<Y>(advection_field)(idx)
                    = ddcHelper::get<Y>(electric_field)(spatial_idx);
            });
        m_advec_rtheta(allfdistribu, get_const_field(advection_field), dt / 2);
        m_advec_vr(allfdistribu, ddcHelper::get<X>(electric_field), dt / 2);
        m_advec_vtheta(allfdistribu, ddcHelper::get<Y>(electric_field), dt);
        m_advec_vr(allfdistribu, ddcHelper::get<X>(electric_field), dt / 2);
        m_advec_vtheta(allfdistribu, ddcHelper::get<Y>(electric_field), dt / 2);
        m_advec_rtheta(allfdistribu, get_const_field(advection_field), dt / 2);
        return allfdistribu;
    }
};