include(GoogleTest)
add_executable(ANAL analysis.cpp)
target_link_libraries(ANAL PRIVATE GTest::gtest gslx::advection gslx::timestepper gslx::utils)
