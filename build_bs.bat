:: Copyright (c) Microsoft Corporation. All rights reserved.
:: Licensed under the MIT License.

@echo off
rem Requires a python 3.6 or higher install to be available in your PATH
nuget restore -PackagesDirectory nuget_root
python %~dp0\tools\ci_build\build.py --use_brainslice --brain_slice_package_path %~dp0\nuget_root --enable_msinternal --build_dir %~dp0\build\Windows %*