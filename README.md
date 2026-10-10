This work has been taken from [Gyselalibxx](https://github.com/gyselax/gyselalibxx) along with [Gyselalibxx-mini-app](https://github.com/gyselax/gysela-mini-app_io). 
This work has been dedicated to my Thesis. At the moment this repository contains only geometry, and a basic main file. 

> [!WARNING]
> **Work In Progress (WIP)**


## List of contributions that i made in gyselalibxx library:

Here I listed some of my contribution to Gyselalibxx, the reason is, for me to track my contribution and it impacts. This will help me to motivate to engage with the community.
1) [Fix incorrect memory_space type alias in DerivField](https://github.com/gyselax/gyselalibxx/pull/553)
2) [Fix Incorrect memory_space type_alias in DerivFieldMem](https://github.com/gyselax/gyselalibxx/pull/559)
3) [Fix docstring errors and operator+ signature in TensorCommon](https://github.com/gyselax/gyselalibxx/pull/563)
4) [exec_space replacement instead of ExecSpace()](https://github.com/gyselax/gyselalibxx/pull/586)
5) [correct column index assertion in Matrix_Corner_Block and Matrix_Periodic_Banded](https://github.com/gyselax/gyselalibxx/pull/567)
6) [add missing enable_tensor_type for CartesianLeviCivitaTensor and fix Levi-Civita size()](https://github.com/gyselax/gyselalibxx/pull/579/changes)
7) [A typo in toolchain's README.md](https://github.com/gyselax/gyselalibxx/pull/624)
8) [Missing template parameter in SplineInterpolator constructor initialization on get_extrapolation](https://github.com/gyselax/gyselalibxx/pull/732)
9) [Fix LaTeX rendering and add (xc, yc) offset in adv README test cases](https://github.com/gyselax/gyselalibxx/pull/735)





=======
### Background

## 1. Electrostatic Plasma and  Boltzmann equation:
A Boltzmann equation refers to an advection equation in phase space with sources. In the simplified 1D geometry in space and velocity it has the general form:

$$
\partial_\tau f + v \partial_x f + \partial_x \phi \partial_v f = S(f)
$$
where f is distribution function and x, v are space and velocity respectively. The Boltzmann equation is differential equation that describes the time-evolving distribution function of a collisional plasma .For a 1D electrostatic plasma on a short timescale, we can neglect the magnetic field term.
$$
\partial_x^2 \phi = \int_{-\infty}^{+\infty} f \mathrm{d}v - 1
$$

Now, the distribution can be written using equilibrium and perturbation form, such as:
$$
f = f_\eq + f_\eq(v) * f_\
$$

The distribution function can be decomposed into velocity dependent equilibrium and time-space-velocity perturbation:
$$
f(x, v, t) = f_{eq}(v) + \tilde{f}(x, v, t), ........(1)
$$
at t=0, equation 1 can be written down as:
$$
f(x, v, t=0) = f_{eq}(v) + \tilde{f}(x, v, t=0)
$$
The initial perturbation is not an independent field. It can be shown as:
$$
\tilde{f}(x, v, t=0) = f_{eq}(v) A \cos(k_x x)
$$
so, after substitution:
$$
f(x, v, t=0) = f_{eq}(v) + f_{eq}(v) A \cos(k_x x)
$$
and 
$$
f(x, v, t=0) = f_{eq}(v) \left[ 1 + A \cos(k_x x) \right]
$$
## 2.  Kinetic Instabilities

# Bump-on-tail instability

The Bump-on-tail instability develops when the equilibrium distribution function presents a positive slope with respect to velocity in the tail. It can occur when a low density plasma beam of finite mean velocity interacts with the bulk plasma, at rest. Suppose, the high-energy electrons are injected into the thermalized plasma, which which results in a non-maxwellian distribution with a **"bump"(High-energy-tail)**  in velocity space. 
We consider, an equilibrium made of 2 maxwellian $$ f_/eq = f_\1 + f_\2 $$. The bulk plasma particles f1 at rest, of density $$ n_\1 = (1 - /eps) $$ and temperature unity, and beam f2 of constant velocit v0, of small density $$ n_\2 = \eps $$ and temperature $$T_\0$$ with $$ \eps $$ a small positive parameter, ranging from $$ 0 <= \eps <= 1$$ 
$$
f_1 = \frac{1 - \varepsilon}{\sqrt{2\pi}} \exp\left\{ \frac{-v^2}{2} \right\}
$$

$$
f_2 = \frac{\varepsilon}{\sqrt{2\pi T_0}} \exp\left\{ \frac{-(v - v_0)^2}{2T_0} \right\}
$$
This bump introduces a positive slope of ∂ f / ∂ v > 0. Which enables **wave-particle resonance** and **the growth of Langmuir waves** . This transfers the energy from particles to wave, results in a growth of the wave perturbation and is oscillated at the nonlinear stage.


## 3. Optimal control

