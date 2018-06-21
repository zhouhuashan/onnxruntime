#!/usr/bin/env python3

import argparse
import fileinput
import glob
import logging
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys

logging.basicConfig(format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.DEBUG)
log = logging.getLogger("Build")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Lotus CI build driver.",
                                     usage='''
Default behavior is --update --build --test.

The Update phase will update git submodules, and run cmake to generate makefiles.
The Build phase will build all projects.
The Test phase will run all unit tests, and optionally the ONNX tests.

Use the individual flags to only run the specified stages.
                                     ''')
    # Main arguments
    parser.add_argument("--build_dir", required=True, help="Path to the build directory.")
    parser.add_argument("--config", nargs="+", default=["Debug"],
                        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
                        help="Configuration(s) to build.")
    parser.add_argument("--update", action='store_true', help="Update makefiles.")
    parser.add_argument("--build", action='store_true', help="Build.")
    parser.add_argument("--parallel", action='store_true', help='''Use parallel build.
    The build setup doesn't get all dependencies right, so --parallel only works if you're just rebuilding Lotus code. 
    If you've done an update that fetched external dependencies you have to build without --parallel the first time. 
    Once that's done, run with "--build --parallel --test" to just build in parallel and run tests.''')
    parser.add_argument("--test", action='store_true', help="Run unit tests.")

    # enable ONNX tests
    parser.add_argument("--enable_onnx_tests", action='store_true',
                        help='''When running the Update phase, enable running ONNX tests in the generated makefiles.
                        When running the Test phase, run onnx_test_running against available test data directories.''')
    parser.add_argument("--pb_home", help="Path to protobuf installation")
    # CUDA related
    parser.add_argument("--cudnn_home", help="Path to CUDNN home.")
    parser.add_argument("--cuda_home", help="Path to CUDA home.")
    parser.add_argument("--use_cuda", action='store_true', help="Enable Cuda.")

    # Python bindings
    parser.add_argument("--enable_pybind", action='store_true', help="Enable Python Bindings.")
    parser.add_argument("--build_wheel", action='store_true', help="Build Python Wheel. ")

    # Build options
    parser.add_argument("--cmake_extra_defines", nargs="+",
                        help="Extra definitions to pass to CMake during build system generation. " +
                             "These are just CMake -D options without the leading -D.")
    parser.add_argument("--x86", action='store_true',
                        help="Create x86 makefiles. Requires --update and no existing cache CMake setup. Delete CMakeCache.txt if needed")

    # Arguments needed by CI
    parser.add_argument("--cmake_path", default="cmake", help="Path to the CMake program.")
    parser.add_argument("--ctest_path", default="ctest", help="Path to the CTest program.")
    parser.add_argument("--skip_submodule_sync", action='store_true', help="Don't do a 'git submodule update'. Makes the Update phase faster.")

    parser.add_argument("--install_onnx", action='store_true',
                        help="Install ONNX. This also creates Lotus ONNX test data in the build directory.")
    parser.add_argument("--use_jemalloc", action='store_true', help="use jemalloc")
    parser.add_argument("--use_openblas", action='store_true', help="Build with OpenBLAS.")
    return parser.parse_args()

def is_windows():
    return sys.platform.startswith("win")

def is_ubuntu_1604():
    return platform.linux_distribution()[0] == 'Ubuntu' and platform.linux_distribution()[1] == '16.04'

def get_config_build_dir(build_dir, config):
    # build directory per configuration
    return os.path.join(build_dir, config)

def run_subprocess(args, cwd=None):
    log.debug("Running subprocess: \n%s", args)
    subprocess.run(args, cwd=cwd, check=True)

def update_submodules(source_dir):
    run_subprocess(["git", "submodule", "update", "--init", "--recursive"], cwd=source_dir)

def install_ubuntu_deps(args):
    try:
        run_subprocess(['add-apt-repository', 'ppa:deadsnakes/ppa'])
    except Exception as e:
        log.error("Could not install ubuntu dependency packages {}".format(str(e)))
        sys.exit(-1)

def install_python_deps():
    dep_packages = ['setuptools', 'wheel', 'numpy']
    run_subprocess([sys.executable, '-m', 'pip', 'install', '--trusted-host', 'files.pythonhosted.org'] + dep_packages)

def generate_build_tree(cmake_path, source_dir, build_dir, cuda_home, cudnn_home, pb_home, configs, cmake_extra_defines, args, cmake_extra_args):
    log.info("Generating CMake build tree")
    cmake_dir = os.path.join(source_dir, "cmake")
    nvml_stub_path = ""
    if args.use_cuda and not is_windows():
        nvml_stub_path = cuda_home + "/lib64/stubs"
    cmake_args = [cmake_path, cmake_dir,
                 "-Dlotus_RUN_ONNX_TESTS=" + ("ON" if args.enable_onnx_tests else "OFF"),
                 "-Dlotus_GENERATE_TEST_REPORTS=ON",
                 "-DPYTHON_EXECUTABLE=" + sys.executable,
                 "-Dlotus_USE_CUDA=" + ("ON" if args.use_cuda else "OFF"),
                 "-Dlotus_CUDNN_HOME=" + (cudnn_home if args.use_cuda else ""),  
                 "-Dlotus_USE_JEMALLOC=" + ("ON" if args.use_jemalloc else "OFF"),
                 "-Dlotus_ENABLE_PYTHON=" + ("ON" if args.enable_pybind else "OFF"),
                 "-Dlotus_USE_EIGEN_FOR_BLAS=" + ("OFF" if args.use_openblas else "ON"),
                 "-Dlotus_USE_OPENBLAS=" + ("ON" if args.use_openblas else "OFF"),
                 "-DCUDA_CUDA_LIBRARY=" + nvml_stub_path
                 ]
    if pb_home:
        cmake_args += ["-DONNX_CUSTOM_PROTOC_EXECUTABLE=" + os.path.join(pb_home,'bin','protoc'), '-Dlotus_USE_PREBUILT_PB=ON']
    cmake_args += ["-D{}".format(define) for define in cmake_extra_defines]

    if is_windows():
        cmake_args += cmake_extra_args

    for config in configs:
        config_build_dir = get_config_build_dir(build_dir, config)
        os.makedirs(config_build_dir, exist_ok=True)

        run_subprocess(cmake_args  + ["-DCMAKE_BUILD_TYPE={}".format(config)], cwd=config_build_dir)

def build_targets(cmake_path, build_dir, configs, parallel, GPU_BUILD):
    for config in configs:
        log.info("Building targets for %s configuration", config)
        build_dir2 = get_config_build_dir(build_dir, config)
        cmd_args = [cmake_path,
                       "--build", build_dir2,
                       "--config", config,
                       ]

        build_tool_args = []
        if parallel:
            num_cores = str(multiprocessing.cpu_count())
            if is_windows():
                build_tool_args += ["/maxcpucount:" + num_cores]
            elif not GPU_BUILD:
                build_tool_args += ["-j" + num_cores]

        if (build_tool_args):
            cmd_args += [ "--" ]
            cmd_args += build_tool_args

        run_subprocess(cmd_args)

def install_onnx_linux(build_dir, source_dir, configs, cmake_path, lotus_onnx_test_data_dir, cmake_extra_args):
    "Install ONNX and create test data."
    #dep_packages = ['typing_extensions','typing','six','protobuf','setuptools', 'numpy', 'pytest_runner']
    #TODO: replace it with conda install
    #run_subprocess([sys.executable, '-m', 'pip', 'install', '--upgrade'] + dep_packages)
    pb_config = None
    pb_src_dir = os.path.join(source_dir, 'cmake', 'external', 'protobuf')
    if 'Release' in configs:
        pb_config = 'Release'
        release_build_dir = get_config_build_dir(build_dir, pb_config)
    elif 'RelWithDebInfo' in configs:
        pb_config = 'RelWithDebInfo'
        release_build_dir = get_config_build_dir(build_dir, pb_config)
    else:
        # make a protobuf release build
        pb_config = 'Release'
        release_build_dir = get_config_build_dir(build_dir, pb_config)
        pb_build_dir = os.path.join(release_build_dir, 'external', 'protobuf','cmake')
        os.makedirs(pb_build_dir, exist_ok=True)
        run_subprocess([cmake_path, os.path.join(pb_src_dir, 'cmake'), '-Dprotobuf_BUILD_TESTS=OFF','-DCMAKE_POSITION_INDEPENDENT_CODE=ON'] + cmake_extra_args,
                    cwd=pb_build_dir)
    pb_build_path = os.path.join(release_build_dir, 'external', 'protobuf','cmake')
    install_dir = os.path.join(release_build_dir, 'install')
    run_subprocess(['make','install','DESTDIR=%s' % install_dir], cwd=pb_build_path)
    os.environ["PATH"] = os.path.join(install_dir, 'usr/local/bin') + os.pathsep + os.environ["PATH"]
    onnx_src = os.path.join(source_dir, 'cmake', 'external', 'onnx')
    run_subprocess([sys.executable, 'setup.py', 'install','--user'], cwd=onnx_src)
    run_subprocess([sys.executable, '-m', 'onnx.backend.test.cmd_tools', 'generate-data', '-o', lotus_onnx_test_data_dir])


def install_onnx_win(build_dir, source_dir, configs, cmake_path, lotus_onnx_test_data_dir, cmake_extra_args):
    "Install ONNX and create test data."
    dep_packages = ['typing_extensions','typing','six','protobuf','setuptools', 'numpy', 'pytest_runner']
    run_subprocess([sys.executable, '-m', 'pip', 'install', '--trusted-host', 'files.pythonhosted.org', '--upgrade'] + dep_packages)

    pb_config = None
    release_build_dir = None
    pb_bin_path = None
    pb_src_dir = os.path.join(source_dir, 'cmake', 'external', 'protobuf')
    if 'Release' in configs:
        pb_config = 'Release'
        release_build_dir = get_config_build_dir(build_dir, pb_config)
        pb_bin_path = os.path.join(release_build_dir, 'external', 'protobuf','cmake', pb_config)
    elif 'RelWithDebInfo' in configs:
        pb_config = 'RelWithDebInfo'
        release_build_dir = get_config_build_dir(build_dir, pb_config)
        pb_bin_path = os.path.join(release_build_dir, 'external', 'protobuf','cmake', pb_config)
    else:
        # make a protobuf release build
        pb_config = 'Release'
        release_build_dir = get_config_build_dir(build_dir, pb_config)
        pb_build_dir = os.path.join(release_build_dir, 'protobuf', 'src', 'protobuf')
        os.makedirs(pb_build_dir, exist_ok=True)
        run_subprocess(
            [cmake_path, os.path.join(pb_src_dir, 'cmake'), '-Dprotobuf_MSVC_STATIC_RUNTIME:BOOL=OFF', '-Dprotobuf_BUILD_TESTS=OFF'] + cmake_extra_args,
            cwd=pb_build_dir)
        run_subprocess([cmake_path,
                        "--build", pb_build_dir,
                        "--config", pb_config])
        pb_bin_path = os.path.join(pb_build_dir, pb_config)

    # Add protoc to PATH
    print('Adding %s to PATH' % pb_bin_path)
    os.environ["PATH"] += os.pathsep + pb_bin_path
    # Add cmake to PATH
    cmake_path_dir = os.path.dirname(cmake_path)
    print('Adding %s to PATH' % cmake_path_dir)
    os.environ["PATH"] += os.pathsep + cmake_path_dir
    os.environ["LIB"] += os.pathsep + pb_bin_path
    pb_inc_dir = os.path.join(pb_src_dir, 'src')
    print('Add %s to INCLUDE' % pb_inc_dir)
    os.environ["INCLUDE"] += os.pathsep + pb_inc_dir
    onnx_src = os.path.join(source_dir, 'cmake', 'external', 'onnx')

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
    run_subprocess([sys.executable, '-m', 'onnx.backend.test.cmd_tools', 'generate-data', '-o', lotus_onnx_test_data_dir])

def add_dir_if_exists(dir, dir_list):
    if (os.path.isdir(dir)):
        dir_list.append(dir)

def set_cuda_dir(cuda_home):
    if (is_windows()):
        cuda_bin_path = os.path.join(cuda_home, 'bin')
        os.environ["CUDA_PATH"] = cuda_home
        os.environ["CUDA_BIN_PATH"] = cuda_bin_path
        os.environ["CUDA_PATH_V9_0"] = cuda_home 
        os.environ["CUDA_TOOLKIT_ROOT_DIR"] = cuda_home
        os.environ["PATH"] += os.pathsep + cuda_bin_path


def run_lotus_tests(ctest_path, build_dir, configs, enable_python_tests):
    for config in configs:
        log.info("Running tests for %s configuration", config)
        cwd = get_config_build_dir(build_dir, config)
        run_subprocess([ctest_path, "--build-config", config, "--verbose"],
                       cwd=cwd)

        if enable_python_tests:
            if is_windows():
                cwd = os.path.join(cwd, config)
            run_subprocess([sys.executable, 'lotus_test_python.py'], cwd=cwd)


def run_onnx_tests(build_dir, configs, lotus_onnx_test_data_dir, onnx_test_data_dir):

    for config in configs:
        test_data_dirs = []
        # test data created by running with --install_onnx
        add_dir_if_exists(os.path.join(lotus_onnx_test_data_dir, 'node'), test_data_dirs)
        cwd = get_config_build_dir(build_dir, config)
        if is_windows():
           exe = os.path.join(cwd, config, 'onnx_test_runner')
        else:
           exe = os.path.join(cwd, 'onnx_test_runner')
        if test_data_dirs:
            run_subprocess([exe] + test_data_dirs, cwd=cwd)

def build_python_wheel(source_dir, build_dir, configs):
    for config in configs:
        cwd = get_config_build_dir(build_dir, config)
        if is_windows():
            cwd = os.path.join(cwd, config)
        run_subprocess([sys.executable, os.path.join(source_dir, 'setup.py'), 'bdist_wheel'], cwd=cwd)

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

    if args.build_wheel:
        args.enable_pybind = True

    ctest_path = args.ctest_path
    build_dir = args.build_dir
    cudnn_home = args.cudnn_home
    cuda_home = args.cuda_home
    script_dir = os.path.realpath(os.path.dirname(__file__))
    source_dir = os.path.normpath(os.path.join(script_dir, "..", ".."))
    # directory from ONNX submodule with ONNX test data
    onnx_test_data_dir = os.path.join(source_dir, "external", "onnx", "onnx", "backend", "test", "data")

    # directory that Lotus ONNX test data is created in if this script is run with --install_onnx
    # If the directory exists and looks valid we assume ONNX is installed and run tests
    # using that data.
    lotus_onnx_test_data_dir = os.path.join(build_dir, 'test_data')

    os.makedirs(build_dir, exist_ok=True)

    configs = set(args.config)

    GPU_BUILD = False
    if (cuda_home):
        set_cuda_dir(cuda_home)
        GPU_BUILD = True

    log.info("Build started")

    cmake_extra_args = []
    if(is_windows()):
      if (args.x86):
        cmake_extra_args = ['-A','Win32','-G', 'Visual Studio 15 2017']
      else:
        cmake_extra_args = ['-A','x64','-T', 'host=x64', '-G', 'Visual Studio 15 2017']

    #Add python to PATH. Please remove this after https://github.com/onnx/onnx/issues/1080 is fixed (@chasun)
    os.environ["PATH"] += os.pathsep + os.path.dirname(sys.executable)

    if (args.update):
        if is_ubuntu_1604():
            install_ubuntu_deps(args)
        if (args.enable_pybind and is_windows()):
            install_python_deps()
        if (not args.skip_submodule_sync):
            update_submodules(source_dir)

        generate_build_tree(cmake_path, source_dir, build_dir, cuda_home, cudnn_home, args.pb_home, configs, cmake_extra_defines,
                            args, cmake_extra_args)

    if (args.build):
        build_targets(cmake_path, build_dir, configs, args.parallel, GPU_BUILD)

    if (args.install_onnx):
        #try to install onnx from this source tree
        if(is_windows()):
          install_onnx_win(build_dir, source_dir, configs, cmake_path, lotus_onnx_test_data_dir, cmake_extra_args)
        else:
          install_onnx_linux(build_dir, source_dir, configs, cmake_path, lotus_onnx_test_data_dir, cmake_extra_args)


    if (args.test):
        run_lotus_tests(ctest_path, build_dir, configs, args.enable_pybind)

    # run the onnx tests if requested explicitly.
    # it could be done implicitly by user installing onnx but currently the tests fail so that doesn't work for CI
    if (args.enable_onnx_tests or args.install_onnx):
        run_onnx_tests(build_dir, configs, lotus_onnx_test_data_dir, onnx_test_data_dir)

    if args.build_wheel:
        build_python_wheel(source_dir, build_dir, configs)

    log.info("Build complete")

if __name__ == "__main__":
    sys.exit(main())
