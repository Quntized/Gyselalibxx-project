add_executable(gys
    gys.cpp
)
find_package(DDC REQUIRED COMPONENTS pdi)
find_package(PDI REQUIRED)
find_package(paraconf REQUIRED COMPONENTS C)

target_link_libraries(gys
    PUBLIC
        gyselalibxx_project
        DDC::core
        DDC::pdi
        PDI::PDI_C
        paraconf::paraconf
)