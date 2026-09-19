#pragma once

constexpr char const* const PDI_CFG = R"PDI_CFG(
metadata:
  iter: int
  time_saved: double

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


)PDI_CFG";