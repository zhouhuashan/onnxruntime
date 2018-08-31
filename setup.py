#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from setuptools import setup, find_packages
from os import path, getcwd
import platform

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    bdist_wheel = None

# Additional binaries
if platform.system() == 'Linux':
  libs = ['lotus_pybind11_state.so', 'libmkldnn.so.0', 'libmklml_intel.so', 'libiomp5.so']
else:
  libs = ['lotus_pybind11_state.pyd', 'mkldnn.dll', 'mklml.dll', 'libiomp5md.dll']

data = [path.join('python', x) for x in libs if path.isfile(path.join('lotus', 'python', x))]

# Additional examples
examples_names = ["mul_1.pb"]
examples = [path.join('python', 'datasets', x) for x in examples_names]

# Description
README = path.join(getcwd(), "README.rst")
with open(README) as f:
    long_description = f.read()

# Setup
setup(
    name='lotus',
    version='0.1.4',
    description='Lotus Runtime Python bindings',
    long_description=long_description,
    author='Microsoft Corporation',
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
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License'],
    )    

