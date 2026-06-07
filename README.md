# ScaNN_optimizatoin
Course work repository


bazel build -c opt --features=thin_lto --copt=-mavx --copt=-mfma --cxxopt="-std=c++17" \
--copt=-fsized-deallocation --copt=-w --action_env=CC=/usr/bin/clang-17 \
--action_env=CXX=/usr/bin/clang++-17 \
//:build_pip_pkg
