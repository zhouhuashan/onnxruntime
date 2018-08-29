Adding a new op
===============

## A new op can be written and registered with Lotus in the following 3 ways:
### 1. Using a dynamic shared library (Linux only)
1. First write the implementation of the op and schema (if required) and
assemble them in a shared library. 
See [this](https://aiinfra.visualstudio.com/_git/Lotus?path=%2Flotus%2Ftest%2Fcustom_op_shared_lib&version=GBmaster)
for an example.

Example of creating a shared lib using g++:
```g++ -std=c++14 -shared test_custom_op.cc -o test_custom_op.so -fPIC
-I. -Iinclude/lotus -L. -llotus_runtime -DONNX_ML -DONNX_NAMESPACE=onnx```

*Note*: Lotus is compiled with gcc4. On gcc version >=5: gcc uses the new C++
ABI since version 5. 
If you compile your op library with gcc>=5, add -D_GLIBCXX_USE_CXX11_ABI=0 to
the command line to make the library compatible with the older abi.

2. Register the shared lib with Lotus. 
See [this](https://aiinfra.visualstudio.com/_git/Lotus?path=%2Flotus%2Ftest%2Fshared_lib%2Ftest_inference.cc&version=GBmaster) for an example.

### 2. Using RegisterCustomRegistry API
1. Implement your kernel and schema (if required) using the OpKernel and OpSchema APIs
   (headers are in the include folder).
2. Create a CustomRegistry object and register your kernel and schema with this
   registry.
3. Register the custom registry with Lotus using RegisterCustomRegistry API.

See
[this](https://aiinfra.visualstudio.com/_git/Lotus?path=%2Flotus%2Ftest%2Fframework%2Flocal_kernel_registry_test.cc&version=GBmaster&line=363&lineStyle=plain&lineEnd=364&lineStartColumn=1&lineEndColumn=1)
as an example.

### 3. Contributing the op to Lotus
This is mostly meant for Microsoft internal partners only.
See [this](https://aiinfra.visualstudio.com/_git/Lotus?path=%2Flotus%2Fcontrib_ops&version=GBmaster) for an example.
