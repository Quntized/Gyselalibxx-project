This work has been taken from [Gyselalibxx](https://github.com/gyselax/gyselalibxx) along with [Gyselalibxx-mini-app](https://github.com/gyselax/gysela-mini-app_io). 
This work has been dedicated to my Thesis. 

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

# Two-stream Instability
The two-stream instability is another fundamental kinetic instability driven by ** counter-propagating electron beams ** in plasma. Both electron and ion have the initial distribution function which is the maxwellian. When two beams of charged particles interpenetrate, they are highly susceptible to microscopic fluctuations.

Now, particles moving initially slightly slower than the wave, the velocity will get closer to the phase velocity, so that it will keep interacting with a long time. On top of that, they will get kinetic energy as
m(v0 + δv ) /2 > mv0 /2
Conversely, those which are decelerated will move further away from the resonance, hence interacting weakly with the wave. In summary, and because energy transfer is due to resonant interactions between
waves and particles, most of the energy transfer will go from the wave to these slow particles.
Performing the same analysis for particles moves slightly faster would lead to the opposite conclusion. Most of the energy goes from those "fast" particles to the wave. 
Then, to decide whether the wave ultimately gains or looses energy, one has to count the number of particles of each category.If the equilibrium distribution function is a Maxwellian,
then there are more particles slower than the wave than faster. As a result, the net balance will
be in favor of the particles, and the wave will loose energy. This corresponds to Landau damping.
Conversely, if the equilibrium distribution function presents a bump on the tail characterizing an
inversion of population at the phase velocity, then more energy will be transfered to the wave.
Such a regime leads to the so-called bump-on-tail instability.
The distribution function is then:
$$f_0(v) = \frac{n_0}{2\sqrt{2\pi v_{th}^2}} \left[ \exp \left( -\frac{(v - v_0)^2}{2v_{th}^2} \right) + \exp \left( -\frac{(v + v_0)^2}{2v_{th}^2} \right) \right]$$

Each is given in gyselalibxx as a allfequilibrium, which contains both the electron and ion distribution function.
# Kinetic source
The source term given in boltzman equation. The first we will consider is kinetic source. Where it allows to inject density and energy independently.
$$ S_{\mathrm{sc}}(x, v_a) = s_{\mathrm{k}} \frac{\mathcal{M}_{\mathrm{sc}}(x)}{\int_0^{L_x} \mathcal{M}_{\mathrm{sc}}(x) \, dx} S_{\mathrm{v}}(v_a) $$

Here, M defines the spatial extent of the source, most of the time it has a hyperbolic tangent expression. And S_v can be written as:

$$ S_{\mathrm{v}}(v) = \left\{ s_0 \left( \frac{3}{2} - \frac{v^2}{2T_{\mathrm{sc}}} \right) + s_2 \left( -\frac{1}{2} + \frac{v^2}{2T_{\mathrm{sc}}} \right) \right\} \frac{1}{\sqrt{2\pi T_{\mathrm{sc}}}} e^{-\frac{v^2}{2T_{\mathrm{sc}}}} $$

The Lx is simulation box length. The T_sc is constant source temperature. The s0, s2 and sk these are the inputs to the code that allows to define the properties of the source. 

# Krook Source Adaptive
The Bhatnagar-Gross-Krook operators, which can act as sources or sinks of particles, momentum and energy. They can be written as:
$$ \mathcal{S}(f_a) = -\nu_a\mathcal{M} (f_a - g). $$

The M here is the same as kinetic source which is to active the intend region, provided by the simulation time. The target distribution function g, which is a maxwellian characterized by a constant density n_g and temperature T_g . The normalized expression is:

$$g(v) = \frac{n_g}{\sqrt{2\pi T_g}} \exp \left( -\frac{v^2}{2T_g} \right).$$

The nu_a coefficient sets the magnitude of the operator. It can be wither constant, or depend on the time ans space variables.

However, in this case, the coefficient is treated as, for electron it can adapt at each time step and spatial position and for ion it assumes to be constant.

## 3. Optimal control
 
Both the kinetic source, krook source adaptive are having great oppurtunity to control the instabilities. However, the prime intent here is, instead of slow simulation which will takes time to control the instabilities. The Transformer model and Reinforcement Learning are present great oppurtunity to run simulation at a faster, and for later, can be controll the instabilities efficiently, which can adapt in physical the problematic plasma instabilities cases.

The NS-Transformer model will learn the function mapping for future time step. We will use that phenomena to predict future time steps. This can be seen by RL (e.g. DDPG) and can alter the action for better rewards.

In this specific case, as plasma waves fluctuations get bigger when energy is transfer from particles to waves, we have taken to minimize this:

$$ J := \int_{0}^{L} |\nabla \phi|^2 dx$$

NS-Transformer will take state in encoder and control in decoder, and can adapt the changing nature of data which are the adaptation of mean and variance. 

Here, specifically, we have taken mean_ne, mean_ni, mean_Te, mean_Ti,$$ \int E^2 dx $$ as state and s0, s2, sk, nu_a, n_g, T_g as the control. 
The RL will try to minimise the J to learn the optimal policy to minimse fluctuations which will driven by instabilities.






The results and poster will soon be present. Currently, I am running the simulation in my University Lab.

Thank you!!
