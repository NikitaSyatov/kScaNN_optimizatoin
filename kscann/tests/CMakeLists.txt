# Unit test (simple, no external framework)
add_executable(test_ivf_pq test_ivf_pq.cpp)
target_link_libraries(test_ivf_pq kscann_lib)
target_compile_options(test_ivf_pq PRIVATE -O2)
add_test(NAME IVF_PQ_Test COMMAND test_ivf_pq)

# Benchmark (requires dataset path as argument)
add_executable(bench_glove bench_glove.cpp)
target_link_libraries(bench_glove kscann_lib)
target_compile_options(bench_glove PRIVATE -O3)

add_executable(bench_mnist bench_mnist.cpp)
target_link_libraries(bench_mnist kscann_lib)
target_compile_options(bench_mnist PRIVATE -O3)

add_executable(bench_gist bench_gist.cpp)
target_link_libraries(bench_gist kscann_lib)
target_compile_options(bench_gist PRIVATE -O3)