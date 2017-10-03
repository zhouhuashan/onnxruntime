import argparse
import glob
import logging
import os
import subprocess
import sys

logging.basicConfig(format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.DEBUG)
log = logging.getLogger("Build")

def parse_arguments():
    parser = argparse.ArgumentParser(description="LotusIR CI build driver.")
    parser.add_argument("--build_dir", required=True, help="Path to the build directory.")
    parser.add_argument("--output_dir", required=True, help="Path to the output directory.")
    parser.add_argument("--config", nargs="+", default=["Debug"],
                        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
                        help="Configuration(s) to build.")
    parser.add_argument("--cmake_path", default="cmake", help="Path to the CMake program.")

    return parser.parse_args()

def run_subprocess(args, cwd=None):
    log.debug("Running subprocess: \n%s", args)
    subprocess.run(args, cwd=cwd, check=True)

def generate_build_tree(cmake_path, source_dir, build_dir):
    log.info("Generating CMake build tree")
    cmake_dir = os.path.join(source_dir, "cmake")
    run_subprocess([cmake_path, cmake_dir,
                   "-G", "Visual Studio 14 Win64",
                   "-DlotusIR_RUN_ONNX_TESTS=ON"],
                   cwd=build_dir)

def build_targets(cmake_path, build_dir, configs):
    targets = [
        "ALL_BUILD",
    ]

    for config in configs:
        for target in targets:
            log.info("Building target %s in configuration %s", target, config)
            run_subprocess([cmake_path,
                           "--build", build_dir,
                           "--target", target,
                           "--config", config,
                           ])

def run_unit_tests(build_dir, output_dir, configs):
    ut_basename = "LotusIR_UT"
    for config in configs:
        ut_file = os.path.join(build_dir, config, ut_basename)
        log.info("Running unit test program: %s", ut_file)
        output_file = os.path.join(output_dir, "test", "{}.{}.results.xml".format(ut_basename, config))
        run_subprocess([ut_file, "--gtest_output=xml:%s" % output_file],
                       cwd=os.path.join(build_dir, config))

def main():
    args = parse_arguments()

    cmake_path = args.cmake_path
    build_dir = args.build_dir
    output_dir = args.output_dir
    script_dir = os.path.realpath(os.path.dirname(__file__))
    source_dir = os.path.join(script_dir, "..", "..")

    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    configs = set(args.config)

    log.info("Build started")

    generate_build_tree(cmake_path, source_dir, build_dir)
    build_targets(cmake_path, build_dir, configs)
    run_unit_tests(build_dir, output_dir, configs)

    log.info("Build complete")

if __name__ == "__main__":
    sys.exit(main())
