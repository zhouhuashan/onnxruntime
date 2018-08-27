@echo off
set "%1%" == "help" goto help:
set PYTHONPATH=%~dp0\build\Windows\Release\Release
set "%1%" == "build" goto build_release:
if exist %PYTHONPATH%\Lotus\__init__.py goto build_doc:

:build_release:
python %~dp0\tools\ci_build\build.py --build_dir %~dp0\build\Windows %* --config Release --build_wheel

:build_doc:
python -c "from sphinx import build_main;build_main(['-j2','-v','-T','-b','html','-d','build/docs/doctrees','docs/python','build/docs/python'])"
goto end:

:help:
@echo Builds the documentation.
@echo It requires modules sphinx, sphinx-gallery, sphinx_rtd_theme, recommonmark.
@echo It also requires onnx, protobuf, onnxmltools, onnx_coreml, lightgbm to run all the examples.
@echo 
@echo help        print the help
@echo build       build Lotus even if already build

:end:
