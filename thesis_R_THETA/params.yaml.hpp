#pragma once

constexpr char const* const params_yaml = R"PDI_CFG(SplineMesh:
  r_ncells: 32
  r_min: 0.0
  r_max: 1.0
  theta_ncells: 64
  theta_min: 0.0
  theta_max: 6.283185307179586
  vpar_ncells: 32
  vpar_min: -6.0
  vpar_max: 6.0
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

SpeciesInfo:
  - name: electron
    charge: -1.0
    mass: 0.0005
    density_eq: 1.0
    temperature_eq: 1.0
    mean_velocity_eq: 0.0
  - name: ion
    charge: 1.0
    mass: 1.0
    density_eq: 1.0
    temperature_eq: 1.0
    mean_velocity_eq: 0.0
)PDI_CFG";