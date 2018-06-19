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

if platform.system() == 'Linux':
  pybind_module = path.join('python', 'lotus_pybind11_state.so')
else:
  pybind_module = path.join('python', 'lotus_pybind11_state.pyd')

setup(
    name='lotus',
    version='0.1.1',
    description='Lotus Runtime Python bindings',
    long_description="Python bindings for Lotus Runtime.",
    author='Lotus team',
    author_email='LotusTeam@microsoft.com',
    cmdclass={'bdist_wheel': bdist_wheel},
    packages=['lotus', 'lotus.python', 'lotus.python.tools'],
    package_data= {
        'lotus': [pybind_module],
    },
    entry_points= {
        'console_scripts': [
            'lotus_test = lotus.python.tools.lotus_test:main',
        ]
    },
)
