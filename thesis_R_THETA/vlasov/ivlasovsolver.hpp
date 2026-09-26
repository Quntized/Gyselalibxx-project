
#pragma once

#include "geometry_rthetavrvtheta.hpp"

/**
 * @brief An abstract class for solving a Vlasov equation.
 */
class IVlasovSolver
{
public:
    virtual ~IVlasovSolver() = default;

    /**
     * @brief Solves a Vlasov equation on a timestep dt.
     *
     * @param[in, out] allfdistribu On input : the initial value of the distribution function.
     *                              On output : the value of the distribution function after solving 
     *                              the Vlasov equation.
     * @param[in] efield The electric field computed at all spatial positions.
     * @param[in] dt The timestep. 
     *
     * @return The distribution function after solving the Vlasov equation.
     */
    virtual DFieldSpVrVthetaRTheta operator()(
            DFieldSpVrVthetaRTheta allfdistribu,
            DVectorConstFieldRTheta<X, Y> efield,
            Real dt) const = 0;
};