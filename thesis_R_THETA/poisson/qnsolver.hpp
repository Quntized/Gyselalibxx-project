// SPDX-License-Identifier: MIT

#pragma once
#include "chargedensitycalculator.hpp"
#include "ddc_aliases.hpp"
#include "iqnsolver.hpp"
#include "polar_spline_fem_poisson_like_solver.hpp"
#include "discrete_poloidal_cs_spline_mapping_builder.hpp"
#include "spline_definitions_r_theta.hpp"
#include "geometry_rthetavrvtheta.hpp"

/**
 * @brief An operator which solves the Quasi-Neutrality equation using a fast
 * Fourier transform.
 *
 * An operator which solves the Quasi-Neutrality equation:
 * @f$ - \frac{d^2 \phi}{dx^2} = \rho @f$
 * using a fast Fourier transform on a periodic index range.
 * This operator only works for equidistant points.
 *
 * The electric field, @f$ \frac{d \phi}{dx} @f$ is calculated using
 * a spline interpolation implemented in ElectricField.
 */
class QNSolver : public IQNSolver
{
public:

    using DiscreteMappingBuilder
        = DiscretePoloidalCSSplineMappingBuilder<X, Y, SplineInterpolatorRThetaConst>;
    using PoissonSolver = PolarSplineFEMPoissonLikeSolver<
        GridR,
        GridTheta,
        PolarBSplinesRTheta,
        SplineInterpolatorRTheta,
        typename DiscreteMappingBuilder::MappingType>;
    PoissonSolver const& m_solve_poisson;
    IChargeDensityCalculator const& m_compute_rho;

public:
    /**
     * Construct the QNSolver operator.
     *
     * @param solve_poisson The operator which solves the Poisson solver.
     * @param compute_rho The operator which calculates the charge density, the right hand side of the equation.
     */
    QNSolver(PoissonSolver const& solve_poisson, IChargeDensityCalculator const& compute_rho);

    ~QNSolver() override = default;

    /**
     * The operator which solves the equation using the method described by the class.
     *
     * @param[out] electrostatic_potential The electrostatic potential, the result of the poisson solver.
     * @param[out] electric_field The electric field, the gradient of the electrostatic potential.
     * @param[in] allfdistribu The distribution function.
     */
    void operator()(
            DFieldRTheta electrostatic_potential,
            DVectorFieldRTheta<X, Y> electric_field,
            DConstFieldSpVrVthetaRTheta allfdistribu) const override;
};