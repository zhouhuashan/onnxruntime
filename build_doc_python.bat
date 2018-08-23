@echo off
python %~dp0\tools\ci_build\build.py --build_dir %~dp0\build\Windows %* --config Release --enable_pybind --build_wheel
if not exist %~dp0build\env_doc_py virtualenv %~dp0build\env_doc_py --system-site-packages
set PATH=%~dp0build\env_doc_py\Scripts;%PATH%
if exist %~dp0build\env_doc_py\Scripts\pip3.7.exe pip install --force-reinstall %~dp0build\Windows\Release\Release\dist\lotus-0.1.4-cp37-cp37m-win_amd64.whl
if exist %~dp0build\env_doc_py\Scripts\pip3.6.exe pip install --force-reinstall %~dp0build\Windows\Release\Release\dist\lotus-0.1.4-cp37-cp36m-win_amd64.whl
python -c "from sphinx import build_main;build_main(['-j2','-v','-T','-b','html','-d','build/docs/doctrees','docs_python','build/docs/python'])"
