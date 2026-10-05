#pragma once

#include "geometry_rthetavrvtheta.hpp"

class IQNSolver
{
public:
    virtual ~IQNSolver() = default;

    virtual void operator()(
            DFieldRTheta electrostatic_potential,
            DVectorFieldRTheta<X, Y> electric_field,
            DConstFieldSpVrVthetaRTheta allfdistribu) const = 0;
};