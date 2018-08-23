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
  libs = ['lotus_pybind11_state.so', 'libmkldnn.so.0', 'libmklml_intel.so', 'libiomp5.so']
else:
  libs = ['lotus_pybind11_state.pyd', 'mkldnn.dll', 'mklml.dll', 'libiomp5md.dll']

data = [path.join('python', x) for x in libs if path.isfile(path.join('lotus', 'python', x))]

examples_names = ["mul_1.pb"]
examples = [path.join('python', 'datasets', x) for x in examples_names]

setup(
    name='lotus',
    version='0.1.4',
    description='Lotus Runtime Python bindings',
    long_description="Python bindings for Lotus Runtime.",
    author='Lotus team',
    author_email='LotusTeam@microsoft.com',
    cmdclass={'bdist_wheel': bdist_wheel},
    packages=['lotus', 'lotus.python', 'lotus.python.tools', 'lotus.python.datasets'],
    package_data= {
        'lotus': data + examples,
    },
    entry_points= {
        'console_scripts': [
            'lotus_test = lotus.python.tools.lotus_test:main',
        ]
    },
)
