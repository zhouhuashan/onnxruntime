onnx_test_runner [options...] <data_root>
Options:
        -m TEST_MODE: TEST_MODE could be 'node' or 'model'. Default: 'node'.
        -p PLANNER_TYPE: PLANNER_TYPE could be 'seq' or 'simple'. Default: 'simple'.
        -h: help

How to run node tests:
1. Install onnx. Onnx's version must > 1.1.0. Strictly greater than!
2. execute:
       backend-test-tools generate-data -o <some_empty_folder>
   e.g. 
       backend-test-tools generate-data -o C:\testdata
    backend-test-tools is a tool under C:\Python35\Scripts (If your python was installed to C:\Python35)
3. compile onnx_test_runner and run
      onnx_test_runner -m node <test_data_dir>
	e.g.
	  onnx_test_runner -m node C:\testdata\node

How to run model tests:
1. Download test data from VSTS drop
   1) Download drop app from https://aiinfra.artifacts.visualstudio.com/_apis/drop/client/exe
      Unzip the downloaded file and add lib/net45 dir to your PATH
   2) Download the test data by using this command:
      drop get -s https://aiinfra.artifacts.visualstudio.com/DefaultCollection -n Lotus/testdata/onnx/model/1 -d C:\testdata
	  You may change C:\testdata to any directory in your disk.
   Full document: https://www.1eswiki.com/wiki/VSTS_Drop

2. compile onnx_test_runner and run
   onnx_test_runner -m model <test_data_dir>
   e.g.
     onnx_test_runner -m model C:\os\onnx\onnx\backend\test\data\pytorch-converted
	 onnx_test_runner -m model C:\testdata
