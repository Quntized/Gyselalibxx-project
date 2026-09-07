#include <ddc/ddc.hpp>

#include <paraconf.h>
#include <pdi.h>
#include "geometry.hpp"
#include <iostream>
#include "paraconfpp.hpp"
#include "input.hpp"

using std::cout;
using std::endl;
namespace fs = std::filesystem;

struct ConfigHandles {
  PC_tree_t conf_gyselax;
  PC_tree_t conf_pdi;
};

void display_help(std::string const &exe) {
  std::cerr << "usage: " << exe << " <config_file.yaml> <pdi_config.yaml>"<< endl;
  std::exit(EXIT_FAILURE);
}


IdxRangeSp init_species_from_yaml(PC_tree_t conf_gyselax) {
  int const nb_species = PCpp_len(conf_gyselax, ".SpeciesInfo");
  IdxRangeSp idx_range_sp(IdxSp(0), IdxStepSp(nb_species));

  host_t<DFieldMemSp> kinetic_charges(idx_range_sp);
  host_t<DFieldMemSp> kinetic_masses(idx_range_sp);
  for (int i = 0; i < nb_species; ++i) {
    std::string const base = ".SpeciesInfo[" + std::to_string(i) + "]";
    kinetic_charges(IdxStepSp(i)) =
        PCpp_double(conf_gyselax, (base + ".charge").c_str());
    kinetic_masses(IdxStepSp(i)) =
        PCpp_double(conf_gyselax, (base + ".mass").c_str());
  }

  ddc::init_discrete_space<Species>(std::move(kinetic_charges),
                                    std::move(kinetic_masses));
  return idx_range_sp;
}
ConfigHandles parse_config_files(int argc, char **argv) {
  ConfigHandles configs{};
  std::string exe = argv[0];
  if (argc > 2) {
    fs::path gysela_config_yml = argv[1];
    if (gysela_config_yml.extension() != ".yaml" &&
        gysela_config_yml.extension() != ".yml") {
      std::cerr << "Expected a .yaml file for the config_file.yaml. Received : "
                << gysela_config_yml << endl;
      display_help(exe);
    }
    configs.conf_gyselax = PC_parse_path(gysela_config_yml.c_str());
    fs::path pdi_config_yml = argv[2];
    if (pdi_config_yml.extension() != ".yaml" &&
        pdi_config_yml.extension() != ".yml") {
      std::cerr << "Expected a .yaml file for the pdi_config.yaml. Received : "
                << pdi_config_yml << endl;
      display_help(exe);
    }
    configs.conf_pdi = PC_parse_path(pdi_config_yml.c_str());
  } else {
    display_help(exe);
  }
  PC_errhandler(PC_NULL_HANDLER);
  return configs;
}

IdxRangeSpTor3DV2D initialise_mesh(PC_tree_t conf_gyselax) {
  IdxRangeSp const idx_range_sp = init_species_from_yaml(conf_gyselax);
  IdxRange<GridTor1> const idx_range_tor1 =
    init_spline_dependent_idx_range<GridTor1, BSplinesTor1,
                                    SplineInterpPointsTor1>(conf_gyselax,
                                                              "Tor1");
    IdxRange<GridTor2> const idx_range_tor2 =
        init_spline_dependent_idx_range<GridTor2, BSplinesTor2,
                                        SplineInterpPointsTor2>(conf_gyselax,
                                                              "Tor2");
    IdxRange<GridTor3> const idx_range_tor3 =
        init_spline_dependent_idx_range<GridTor3, BSplinesTor3,
                                        SplineInterpPointsTor3>(conf_gyselax,
                                                              "Tor3");
    IdxRange<GridVpar> const idx_range_vpar =
        init_spline_dependent_idx_range<GridVpar, BSplinesVpar,
                                        SplineInterpPointsVpar>(conf_gyselax,
                                                              "Vpar");
    IdxRange<GridMu> const idx_range_mu =
        init_spline_dependent_idx_range<GridMu, BSplinesMu, SplineInterpPointsMu>(
        conf_gyselax, "Mu");

    cout << "Grid sizes:" << endl;
    cout << "  tor1: " << idx_range_tor1.size() << endl;
    cout << "  tor2: " << idx_range_tor2.size() << endl;
    cout << "  tor3: " << idx_range_tor3.size() << endl;
    cout << "  vpar: " << idx_range_vpar.size() << endl;
    cout << "  mu: " << idx_range_mu.size() << endl;

    return IdxRangeSpTor3DV2D(idx_range_sp, idx_range_tor1, idx_range_tor2,
                            idx_range_tor3, idx_range_vpar, idx_range_mu);
}
void init_distribution_fun(DFieldSpGrid allfdistribu,
                           IdxRangeSpTor3DV2D const &mesh,
                           PC_tree_t conf_gyselax) {
  // Read species-specific parameters from YAML
  IdxRangeSp const idx_range_sp(mesh);

  int const nb_species = idx_range_sp.size();
  host_t<DFieldMem<IdxRangeSp>> density_host(idx_range_sp);
  host_t<DFieldMem<IdxRangeSp>> temperature_host(idx_range_sp);
  host_t<DFieldMem<IdxRangeSp>> mean_velocity_host(idx_range_sp);

  for (int i = 0; i < nb_species; ++i) {
    std::string const base = ".SpeciesInfo[" + std::to_string(i) + "]";
    density_host(IdxSp(i)) =
        PCpp_double(conf_gyselax, (base + ".N_maxw").c_str());
    temperature_host(IdxSp(i)) =
        PCpp_double(conf_gyselax, (base + ".T_maxw").c_str());
    mean_velocity_host(IdxSp(i)) =
        PCpp_double(conf_gyselax, (base + ".Upar_maxw").c_str());
  }

  // Copy to device
  DFieldMem<IdxRangeSp> density_alloc(idx_range_sp);
  DFieldMem<IdxRangeSp> temperature_alloc(idx_range_sp);
  DFieldMem<IdxRangeSp> mean_velocity_alloc(idx_range_sp);
  DField<IdxRangeSp> density = get_field(density_alloc);
  DField<IdxRangeSp> temperature = get_field(temperature_alloc);
  DField<IdxRangeSp> mean_velocity = get_field(mean_velocity_alloc);
  ddc::parallel_deepcopy(density, density_host);
  ddc::parallel_deepcopy(temperature, temperature_host);
  ddc::parallel_deepcopy(mean_velocity, mean_velocity_host);

  // Compute Maxwellian distribution: fM(vpar,mu) =
  // (2*PI*T)**(-1.5)*n*exp(-energy) with energy = 0.5*((vpar-Upar)**2-mu) /T
  ddc::parallel_for_each(
      Kokkos::DefaultExecutionSpace(), mesh,
      KOKKOS_LAMBDA(IdxSpTor3DV2D const ispgrid) {
        IdxSp const isp(ispgrid);
        double const density_loc = density(isp);
        double const temperature_loc = temperature(isp);
        double const mean_velocity_loc = mean_velocity(isp);
        double const vpar = ddc::coordinate(ddc::select<GridVpar>(ispgrid));
        double const mu = ddc::coordinate(ddc::select<GridMu>(ispgrid));
        double const inv_2piT = 1. / (2. * M_PI * temperature_loc);
        double const coeff_maxw = Kokkos::pow(inv_2piT, 1.5);
        double const energy =
            0.5 *
            ((vpar - mean_velocity_loc) * (vpar - mean_velocity_loc) + mu) /
            temperature_loc;
        allfdistribu(ispgrid) = coeff_maxw * density_loc * Kokkos::exp(-energy);
      });
}
