#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <ddc/pdi.hpp>
#include <ddc/ddc.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <ddc/ddc.hpp>
#include <ddc/pdi.hpp>

#include <paraconf.h>
#include <pdi.h>


#include "core.hpp"
#include "ddc_alias_inline_functions.hpp"
#include "fluid_moments.hpp"
#include "geometry.hpp"
#include "input.hpp"
#include "mesh_builder.hpp"
#include "mpitransposealltoall.hpp"
#include "output.hpp"
#include "paraconfpp.hpp"
#include "pdi_helper.hpp"
#include "quadrature.hpp"
#include "species_init.hpp"
#include "transpose.hpp"
#include "input.hpp"
#include "trapezoid_quadrature.hpp"
#include "spline_interpolation.hpp"
#include "params.yaml.hpp"

int main(int argc, char **argv)
{
    fs::create_directory("output");
    Kokkos::ScopeGuard scope(argc, argv);
    ddc::ScopeGuard ddc_scope(argc,argv);
    PC_tree_t conf_gyselax = parse_executable_arguments(argc, argv, params_yaml);
    PC_tree_t conf_pdi = PC_parse_string(PDI_CFG);
    PC_errhandler(PC_NULL_HANDLER);
    PDI_init(conf_pdi);
    IdxRangeSpTor3DV2D const global_mesh = initialise_mesh(conf_gyselax);
    //IdxRangeSp const idx_range_kinsp = init_species(conf_gyselalibxx);
    IdxRangeSp const idx_range_kinsp(global_mesh);
    double const dt(PCpp_double(conf_gyselax, ".Time.delta_t"));
    double const final_T(PCpp_double(conf_gyselax, ".Time.final_T"));
    DFieldMemSpGrid f_memory(global_mesh);
    DFieldSpGrid f = get_field(f_memory);
    init_distribution_fun(
        f,
        global_mesh,
        conf_gyselax);
    ddc::parallel_for_each(
        Kokkos::DefaultExecutionSpace(),
        global_mesh,
        KOKKOS_LAMBDA(IdxSpTor3DV2D const idx) {
            double const value = f(idx);

            if (!isfinite(value) || value < 0.0) {
                f(idx) = 0.0;
            }
        });
}