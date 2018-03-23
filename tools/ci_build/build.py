#!/usr/bin/env python3

import argparse
import glob
import logging
import os
import subprocess
import sys

logging.basicConfig(format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.DEBUG)
log = logging.getLogger("Build")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Lotus CI build driver.")
    parser.add_argument("--build_dir", required=True, help="Path to the build directory.")
    parser.add_argument("--config", nargs="+", default=["Debug"],
                        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
                        help="Configuration(s) to build.")
    parser.add_argument("--cmake_extra_defines", nargs="+",
                        help="Extra definitions to pass to CMake during build system generation. " +
                             "These are just CMake -D options without the leading -D.")
    parser.add_argument("--cmake_path", default="cmake", help="Path to the CMake program.")
    parser.add_argument("--ctest_path", default="ctest", help="Path to the CTest program.")

    return parser.parse_args()

def is_windows():
    return sys.platform.startswith("win")

def get_config_build_dir(build_dir, config):
    if is_windows():
        # assume Visual Studio generator, single build directory
        return build_dir
    else:
        # build directory per configuration
        return os.path.join(build_dir, config)

def run_subprocess(args, cwd=None):
    log.debug("Running subprocess: \n%s", args)
    subprocess.run(args, cwd=cwd, check=True)

def generate_build_tree(cmake_path, source_dir, build_dir, configs,
                        cmake_extra_defines):
    log.info("Generating CMake build tree")
    cmake_dir = os.path.join(source_dir, "cmake")
    cmake_args = [cmake_path, cmake_dir,
                 "-Dlotus_RUN_ONNX_TESTS=OFF",
                 "-Dlotus_GENERATE_TEST_REPORTS=ON",
                 ]
    cmake_args += ["-D{}".format(define) for define in cmake_extra_defines]

    if is_windows():
        run_subprocess(cmake_args + ["-A", "x64"], cwd=build_dir)
    else:
        for config in configs:
            config_build_dir = get_config_build_dir(build_dir, config)
            os.makedirs(config_build_dir, exist_ok=True)
            run_subprocess(cmake_args + ["-DCMAKE_BUILD_TYPE={}".format(config)],
                           cwd=config_build_dir)

def build_targets(cmake_path, build_dir, configs):
    for config in configs:
        log.info("Building targets for %s configuration", config)
        run_subprocess([cmake_path,
                       "--build", get_config_build_dir(build_dir, config),
                       "--config", config,
                       ])

def run_tests(ctest_path, build_dir, configs):
    for config in configs:
        log.info("Running tests for %s configuration", config)
        run_subprocess([ctest_path, "--build-config", config, "--verbose"],
                       cwd=get_config_build_dir(build_dir, config))

def main():
    args = parse_arguments()

    cmake_path = args.cmake_path
    cmake_extra_defines = \
        args.cmake_extra_defines if args.cmake_extra_defines else []
    ctest_path = args.ctest_path
    build_dir = args.build_dir
    script_dir = os.path.realpath(os.path.dirname(__file__))
    source_dir = os.path.join(script_dir, "..", "..")

    os.makedirs(build_dir, exist_ok=True)

    configs = set(args.config)

    log.info("Build started")

    generate_build_tree(cmake_path, source_dir, build_dir, configs,
                        cmake_extra_defines)
    build_targets(cmake_path, build_dir, configs)
    run_tests(ctest_path, build_dir, configs)

    log.info("Build complete")

if __name__ == "__main__":
    sys.exit(main())
