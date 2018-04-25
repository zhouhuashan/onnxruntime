#!/usr/bin/env python3

import argparse
import glob
import logging
import os
import subprocess
import sys
import fileinput
import shutil

logging.basicConfig(format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.DEBUG)
log = logging.getLogger("Build")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Lotus CI build driver.")
    parser.add_argument("--build_dir", required=True, help="Path to the build directory.")
    parser.add_argument("--config", nargs="+", default=["Debug"],
                        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
                        help="Configuration(s) to build.")
    parser.add_argument("--cudnn_home", help="Path to CUDNN home.")
    parser.add_argument("--cmake_extra_defines", nargs="+",
                        help="Extra definitions to pass to CMake during build system generation. " +
                             "These are just CMake -D options without the leading -D.")
    parser.add_argument("--cmake_path", default="cmake", help="Path to the CMake program.")
    parser.add_argument("--ctest_path", default="ctest", help="Path to the CTest program.")
    parser.add_argument("--update", action='store_true', help="Update makefiles.")
    parser.add_argument("--enable_onnx_tests", action='store_true',
                        help="When running the Update phase, enable running ONNX tests in the generated makefiles.")
    parser.add_argument("--install_onnx", action='store_true',
                        help="Install onnx after building Lotus, for running tests")
    parser.add_argument("--build", action='store_true', help="Build.")
    parser.add_argument("--parallel", action='store_true', help="Use parallel build.")
    parser.add_argument("--test", action='store_true', help="Run unit tests.")
    parser.add_argument("--use_cuda", action='store_true', help="Enable Cuda.")

    return parser.parse_args()

def is_windows():
    return sys.platform.startswith("win")

def get_config_build_dir(build_dir, config):
    # build directory per configuration
    return os.path.join(build_dir, config)

def run_subprocess(args, cwd=None):
    log.debug("Running subprocess: \n%s", args)
    subprocess.run(args, cwd=cwd, check=True)

def generate_build_tree(cmake_path, source_dir, build_dir, cudnn_home, configs, cmake_extra_defines, enable_onnx_tests, use_cuda):
    log.info("Generating CMake build tree")
    cmake_dir = os.path.join(source_dir, "cmake")
    cmake_args = [cmake_path, cmake_dir,
                 "-Dlotus_RUN_ONNX_TESTS=" + ("ON" if enable_onnx_tests else "OFF"),
                 "-Dlotus_GENERATE_TEST_REPORTS=ON",
                 "-Dlotus_USE_CUDA=" + ("ON" if use_cuda else "OFF"),
                 "-Dlotus_CUDNN_HOME=" + (cudnn_home if use_cuda else "")  
                 ]

    cmake_args += ["-D{}".format(define) for define in cmake_extra_defines]

    if is_windows():
        cmake_args += ["-A", "x64", "-DCMAKE_GENERATOR='Visual Studio 15 2017'"]

    for config in configs:
        config_build_dir = get_config_build_dir(build_dir, config)
        os.makedirs(config_build_dir, exist_ok=True)

        run_subprocess(cmake_args  + ["-DCMAKE_BUILD_TYPE={}".format(config)], cwd=config_build_dir)

def build_targets(cmake_path, build_dir, configs, parallel):
    for config in configs:
        log.info("Building targets for %s configuration", config)
        build_dir = get_config_build_dir(build_dir, config)
        cmd_args = [cmake_path,
                       "--build", build_dir,
                       "--config", config,
                       ]

        build_tool_args = []
        if parallel:
            if is_windows():
                build_tool_args += ["/maxcpucount:4"]
            else:
                build_tool_args += ["-j4"]

        if (build_tool_args):
            cmd_args += [ "--" ]
            cmd_args += build_tool_args

        run_subprocess(cmd_args)


def install_onnx(build_dir, source_dir, configs, cmake_path, onnx_test_data_dir):
    "Install ONNX and create test data."
    dep_packages = ['typing_extensions','typing','six','protobuf','setuptools', 'numpy', 'pytest_runner']
    run_subprocess([sys.executable, '-m', 'pip', 'install', '--trusted-host', 'files.pythonhosted.org', '--upgrade'] + dep_packages)

    pb_config = None
    release_build_dir = None
    pb_src_dir = None

    if 'Release' in configs:
        pb_config = 'Release'
        release_build_dir = get_config_build_dir(build_dir, pb_config)
        pb_src_dir = os.path.join(release_build_dir, 'protobuf\src\protobuf')
    elif 'RelWithDebInfo' in configs:
        pb_config = 'RelWithDebInfo'
        release_build_dir = get_config_build_dir(build_dir, pb_config)
        pb_src_dir = os.path.join(release_build_dir, 'protobuf\src\protobuf')
    else:
        # make a protobuf release build
        pb_config = 'Release'
        release_build_dir = get_config_build_dir(build_dir, pb_config)
        pb_build_dir = os.path.join(release_build_dir, 'protobuf\src\protobuf')
        os.makedirs(pb_build_dir, exist_ok=True)
        pb_src_dir = os.path.join(get_config_build_dir(build_dir, 'Debug'), 'protobuf', 'src', 'protobuf')
        # clean up old config
        pb_cmake_cache = os.path.join(pb_src_dir, 'CMakeCache.txt')
        if (os.path.isfile(pb_cmake_cache)):
            os.remove(pb_cmake_cache)

        shutil.rmtree(os.path.join(pb_src_dir, 'CMakeFiles'), ignore_errors=True)
        run_subprocess(
            [cmake_path, os.path.join(pb_src_dir, 'cmake'), '-Dprotobuf_MSVC_STATIC_RUNTIME:BOOL=OFF', '-T', 'host=x64',
             '-G', 'Visual Studio 15 2017 Win64', '-Dprotobuf_BUILD_TESTS=OFF'],
            cwd=pb_build_dir)
        run_subprocess([cmake_path,
                        "--build", pb_build_dir,
                        "--config", pb_config])

    # Add protoc to PATH
    pb_bin_path = os.path.join(release_build_dir, 'protobuf\src\protobuf', pb_config)
    print('Add %s to PATH' % pb_bin_path)
    os.environ["PATH"] += os.pathsep + pb_bin_path
    # Add cmake to PATH
    os.environ["PATH"] += os.pathsep + os.path.dirname(cmake_path)
    os.environ["LIB"] += os.pathsep + pb_bin_path
    pb_inc_dir = os.path.join(pb_src_dir, 'src')
    print('Add %s to INCLUDE' % pb_inc_dir)
    os.environ["INCLUDE"] += os.pathsep + pb_inc_dir
    onnx_src = os.path.join(source_dir, 'external', 'onnx')

    # patch onnx source code
    with fileinput.input(os.path.join(onnx_src, 'CMakeLists.txt'), inplace=1) as f:
        for line in f:
            print(line.replace("onnx_cpp2py_export PRIVATE /MT", "onnx_cpp2py_export PRIVATE /MD").rstrip('\r\n'))

    with fileinput.input(os.path.join(onnx_src, 'setup.py'), inplace=1) as f:
        for line in f:
            line = line.replace('DONNX_USE_MSVC_STATIC_RUNTIME=ON', 'DONNX_USE_MSVC_STATIC_RUNTIME=OFF').rstrip('\r\n')
            line = line.replace('os.curdir]', 'os.curdir,\'--config\',\'Release\']')
            print(line)

    run_subprocess([sys.executable, 'setup.py', 'install'], cwd=onnx_src)
    run_subprocess([sys.executable, '-m', 'onnx.backend.test.cmd_tools', 'generate-data', '-o', onnx_test_data_dir])


def run_tests(ctest_path, build_dir, configs, onnx_test_data_dir):
    for config in configs:
        log.info("Running tests for %s configuration", config)
        cwd = get_config_build_dir(build_dir, config)
        run_subprocess([ctest_path, "--build-config", config, "--verbose"],
                       cwd=cwd)

        # If test data dir exists and looks valid, assume we've run with --install_onnx
        # and run the onnx tests.
        # TODO: Add additional arg if this isn't a good default behaviour and user should explicitly enable
        if os.path.isdir(onnx_test_data_dir):
            node_dir = os.path.join(onnx_test_data_dir, 'node')
            if (os.path.isdir(node_dir)):
                run_subprocess([os.path.join(cwd,config,'onnx_test_runner'), node_dir],cwd=cwd)

def main():
    args = parse_arguments()

    cmake_path = args.cmake_path
    cmake_extra_defines = args.cmake_extra_defines if args.cmake_extra_defines else []

    # if there was no explicit argument saying what to do, default to update, build and test.
    if (args.update == False and args.build == False and args.test == False):
        log.debug("Defaulting to running update, build and test.")
        args.update = True
        args.build = True
        args.test = True

    ctest_path = args.ctest_path
    build_dir = args.build_dir
    cudnn_home = args.cudnn_home
    script_dir = os.path.realpath(os.path.dirname(__file__))
    source_dir = os.path.join(script_dir, "..", "..")

    # directory that ONNX test data is created in if this script is run with --install_onnx
    # If the directory exists and looks valid we assume ONNX is installed and run tests
    # using that data.
    onnx_test_data_dir = os.path.join(build_dir, 'test_data')

    os.makedirs(build_dir, exist_ok=True)

    configs = set(args.config)

    log.info("Build started")

    if (args.update):
        generate_build_tree(cmake_path, source_dir, build_dir, cudnn_home, configs, cmake_extra_defines, args.enable_onnx_tests, args.use_cuda)

    if (args.build):
        build_targets(cmake_path, build_dir, configs, args.parallel)

    if (args.install_onnx and sys.platform.startswith('win')):
        #try to install onnx from this source tree
        install_onnx(build_dir, source_dir, configs, cmake_path, onnx_test_data_dir)

    if (args.test):
        run_tests(ctest_path, build_dir, configs, onnx_test_data_dir)

    log.info("Build complete")

if __name__ == "__main__":
    sys.exit(main())
