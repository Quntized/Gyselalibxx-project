#pragma once

constexpr char const* const PDI_CFG = R"PDI_CFG(
metadata:
  iter: int
  time_saved: double
  time_step_diag: int
  Nr_spline_cells: int
  Ntheta_spline_cells: int
  Nvr_spline_cells: int
  Nvtheta_spline_cells: int
  Nkinspecies: int
  deltat: double
  nbiter: int

  MeshR_extents: {type: array, subtype: int64, size: 1}
  MeshR:
    type: array
    subtype: double
    size: ['$MeshR_extents[0]']
  MeshTheta_extents: {type: array, subtype: int64, size: 1}
  MeshTheta:
    type: array
    subtype: double
    size: ['$MeshTheta_extents[0]']
  MeshVr_extents: {type: array, subtype: int64, size: 1}
  MeshVr:
    type: array
    subtype: double
    size: ['$MeshVr_extents[0]']
  MeshVtheta_extents: {type: array, subtype: int64, size: 1}
  MeshVtheta:
    type: array
    subtype: double
    size: ['$MeshVtheta_extents[0]']

  fdistribu_charges_extents: {type: array, subtype: int64, size: 1}
  fdistribu_charges:
    type: array
    subtype: double
    size: ['$fdistribu_charges_extents[0]']
  fdistribu_masses_extents: {type: array, subtype: int64, size: 1}
  fdistribu_masses:
    type: array
    subtype: double
    size: ['$fdistribu_masses_extents[0]']

data:
  fdistribu_extents: {type: array, subtype: int64, size: 5}
  fdistribu:
    type: array
    subtype: double
    size: ['$fdistribu_extents[0]', '$fdistribu_extents[1]', '$fdistribu_extents[2]', '$fdistribu_extents[3]', '$fdistribu_extents[4]']
  electrostatic_potential_extents: {type: array, subtype: int64, size: 2}
  electrostatic_potential:
    type: array
    subtype: double
    size: ['$electrostatic_potential_extents[0]', '$electrostatic_potential_extents[1]']

plugins:
  decl_hdf5:
    - file: 'output/GYSELALIBXX_initstate.h5'
      on_event: [initialisation]
      collision_policy: replace_and_warn
      write: [Nr_spline_cells, Ntheta_spline_cells, Nvr_spline_cells, Nvtheta_spline_cells, Nkinspecies, MeshR, MeshTheta, MeshVr, MeshVtheta, fdistribu_charges, fdistribu_masses, deltat, nbiter, time_step_diag]
    - file: 'output/GYSELALIBXX_${iter:05}.h5'
      on_event: [iteration, last_iteration]
      when: '${iter} % ${time_step_diag} = 0'
      collision_policy: replace_and_warn
      write: [time_saved, fdistribu, electrostatic_potential]
)PDI_CFG";