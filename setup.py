"""setup script for Lotus package
"""
from setuptools import setup, find_packages
from os import path
import platform

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    bdist_wheel = None

data = []
if platform.system() == 'Linux':
  pybind_module = path.join('python', 'lotus_pybind11_state.so')
  mkl_dll = path.join('python', 'libmkldnn.so.0')
else:
  pybind_module = path.join('python', 'lotus_pybind11_state.pyd')
  mkl_dll = path.join('python', 'mkldnn.dll')

data.append(pybind_module)
if path.isfile(path.join('lotus', mkl_dll)):
  data.append(mkl_dll)

setup(
    name='lotus',
    version='0.1.3',
    description='Lotus Runtime Python bindings',
    long_description="Python bindings for Lotus Runtime.",
    author='Lotus team',
    author_email='LotusTeam@microsoft.com',
    cmdclass={'bdist_wheel': bdist_wheel},
    packages=['lotus', 'lotus.python', 'lotus.python.tools'],
    package_data= {
        'lotus': data,
    },
    entry_points= {
        'console_scripts': [
            'lotus_test = lotus.python.tools.lotus_test:main',
        ]
    },
)
