onnx_test_runner [options...] <data_root>
Options:
        -m TEST_MODE: TEST_MODE could be 'node' or 'model'. Default: 'node'.
        -p PLANNER_TYPE: PLANNER_TYPE could be 'seq' or 'simple'. Default: 'simple'.
        -h: help

How to run node tests:
1. Install onnx. Onnx's version must > 1.1.0. Strictly greater than!
2. execute:
       backend-test-tools generate-data -o <some_empty_folder>
    backend-test-tools is a tool under C:\Python35\Scripts (If your python was installed to C:\Python35)
3. compile onnx_test_runner and run
    onnx_test_runner -m node <test_data_dir>

How to run model tests:
1. Download onnx models from their model zoo
2. compile onnx_test_runner and run
   onnx_test_runner -m model <test_data_dir>

