// SPDX-License-Identifier: MIT
#pragma once
#include <cmath>

#include <ddc/ddc.hpp>

#include "ddc_aliases.hpp"

/**
 * @brief A 2D density solution defined in the physical (x, y) plane
 * via a Czarny (or any) mapping from logical (r, theta) coordinates.
 *
 * The equilibrium density is a Gaussian centered at the O-point
 * (magnetic axis) of the mapping:
 *
 * @f$ \rho_{\mathrm{eq}}(r, \theta) = \exp\left(-\frac{(x - x_O)^2 + (y - y_O)^2}
 *     {2\sigma^2}\right) @f$
 *
 * The initial perturbed density adds an azimuthal mode:
 *
 * @f$ \rho_0(r, \theta) = \rho_{\mathrm{eq}}(r, \theta) \cdot
 *     (1 + \varepsilon \cos(l\theta)) @f$
 *
 * where @f$ (x,y) = \mathcal{F}(r, \theta) @f$ is the coordinate mapping
 * and @f$ (x_O, y_O) @f$ is the O-point of the mapping.
 *
 * @tparam Mapping
 *      A class describing a mapping from curvilinear (r, theta) to
 *      Cartesian (x, y) coordinates. Must provide operator() and o_point().
 */
template <class Mapping>
class CzarnyDensitySolution
{
public:
    /// The type of the first physical coordinate.
    using X = typename Mapping::cartesian_tag_x;
    /// The type of the second physical coordinate.
    using Y = typename Mapping::cartesian_tag_y;
    /// The type of the first logical coordinate.
    using R = typename Mapping::curvilinear_tag_r;
    /// The type of the second logical coordinate.
    using Theta = typename Mapping::curvilinear_tag_theta;

private:
    Mapping const& m_mapping;

    /**
     * @brief Width of the Gaussian equilibrium profile.
     */
    double m_sigma;

    /**
     * @brief Amplitude of the azimuthal perturbation.
     */
    double m_eps;

    /**
     * @brief Azimuthal mode number of the perturbation.
     */
    int m_l;

public:
    /**
     * @brief Instantiate a CzarnyDensitySolution.
     *
     * @param[in] mapping
     *      The coordinate mapping from logical (r, theta) to physical (x, y).
     * @param[in] sigma
     *      The width @f$ \sigma @f$ of the Gaussian equilibrium profile.
     * @param[in] eps
     *      The amplitude @f$ \varepsilon @f$ of the azimuthal perturbation.
     * @param[in] l
     *      The azimuthal mode number @f$ l @f$ of the perturbation.
     */
    CzarnyDensitySolution(
            Mapping const& mapping,
            double sigma,
            double eps,
            int l)
        : m_mapping(mapping)
        , m_sigma(sigma)
        , m_eps(eps)
        , m_l(l)
    {
    }

    /**
     * @brief Get the initial perturbed density.
     *
     * The initial condition is given by
     *
     * @f$ \rho_0(r, \theta) = \exp\left(-\frac{(x - x_O)^2 + (y - y_O)^2}
     *     {2\sigma^2}\right) \cdot (1 + \varepsilon \cos(l\theta)) @f$
     *
     * where @f$ (x, y) = \mathcal{F}(r, \theta) @f$ and
     * @f$ (x_O, y_O) @f$ is the O-point of the mapping.
     *
     * @param[in] coord
     *      The coordinate @f$ (r, \theta) @f$ where we evaluate the density.
     *
     * @return The value of the initial perturbed density at the given coordinate.
     */
    double initialisation(Coord<R, Theta> const& coord) const
    {
        Coord<X, Y> const coord_xy = m_mapping(coord);
        Coord<X, Y> const o_pt = m_mapping.o_point();
        double const dx = ddc::get<X>(coord_xy) - ddc::get<X>(o_pt);
        double const dy = ddc::get<Y>(coord_xy) - ddc::get<Y>(o_pt);
        double const theta = ddc::get<Theta>(coord);

        double const profile = std::exp(-(dx * dx + dy * dy) / (2.0 * m_sigma * m_sigma));
        return profile * (1.0 + m_eps * std::cos(m_l * theta));
    }

    /**
     * @brief Get the equilibrium density (no perturbation).
     *
     * The equilibrium is given by
     *
     * @f$ \rho_{\mathrm{eq}}(r, \theta) = \exp\left(-\frac{(x - x_O)^2 + (y - y_O)^2}
     *     {2\sigma^2}\right) @f$
     *
     * @param[in] coord
     *      The coordinate @f$ (r, \theta) @f$ where we evaluate the equilibrium.
     *
     * @return The value of the equilibrium density at the given coordinate.
     */
    double equilibrium(Coord<R, Theta> const& coord) const
    {
        Coord<X, Y> const coord_xy = m_mapping(coord);
        Coord<X, Y> const o_pt = m_mapping.o_point();
        double const dx = ddc::get<X>(coord_xy) - ddc::get<X>(o_pt);
        double const dy = ddc::get<Y>(coord_xy) - ddc::get<Y>(o_pt);

        return std::exp(-(dx * dx + dy * dy) / (2.0 * m_sigma * m_sigma));
    }
};
