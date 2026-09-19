#pragma once

constexpr char const* const params_yaml = R"PDI_CFG(SplineMesh:
  r_ncells: 32
  r_min: 0.0
  r_max: 1.0
  theta_ncells: 64
  theta_min: 0.0
  theta_max: 6.283185307179586
  vr_ncells: 32
  vr_min: -6.0
  vr_max: 6.0
  vtheta_ncells: 32
  vtheta_min: -6.0
  vtheta_max: 6.0

CzarnyMapping:
  epsilon: 0.3
  e: 1.4
  x0: 6.1
  y0: 0.3

Algorithm:
  deltat: 0.01
  nbiter: 10

Output:
  time_step_diag: 1

SpeciesInfo:
  - name: electron
    charge: -1.0
    mass: 0.0005
    density_eq: 1.0
    temperature_eq: 1.0
    mean_velocity_eq: 0.0
    perturb_mode: 1
    perturb_amplitude: 0.01
  - name: ion
    charge: 1.0
    mass: 1.0
    density_eq: 1.0
    temperature_eq: 1.0
    mean_velocity_eq: 0.0
    perturb_mode: 1
    perturb_amplitude: 0.01
)PDI_CFG";