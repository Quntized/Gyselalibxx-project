include(GoogleTest)
add_executable(analysis analysis.cpp)
target_link_libraries(ANAL PRIVATE GTest::gtest)