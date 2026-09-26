#pragma once

#include "geometry_rthetavrvtheta.hpp"

class IEquilibrium
{
public:
    virtual ~IEquilibrium() = default;

    virtual DFieldSpVrVtheta operator()(DFieldSpVrVtheta allfequilibrium) const = 0;
};