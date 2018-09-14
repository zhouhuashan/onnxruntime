# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.

import os
import sys
import shutil
# Check these extensions were installed.
import sphinx_gallery.gen_gallery
# The package should be installed in a virtual environment.
import onnxruntime


# -- Project information -----------------------------------------------------

project = 'ONNX Runtime'
copyright = '2018, Microsoft'
author = 'Microsoft'
version = '0.1'
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.imgmath',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    "sphinx.ext.autodoc",
    "sphinx_gallery.gen_gallery",
    'sphinx.ext.autodoc',
    "docfx_yaml.md_outputter",
    "docfx_yaml.extension",
]

templates_path = ['_templates']

source_parsers = {
   '.md': 'recommonmark.parser.CommonMarkParser',
}

source_suffix = ['.rst', '.md']

master_doc = 'main'
language = "en"
exclude_patterns = []
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

html_static_path = ['_static']
# html_sidebars = {}

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'https://docs.python.org/': None}

# -- Options for Sphinx Gallery ----------------------------------------------

sphinx_gallery_conf = {
     'examples_dirs': 'examples',
     'gallery_dirs': 'auto_examples',
}

# -- Setup actions -----------------------------------------------------------

def setup(app):
    # Placeholder to initialize the folder before
    # generating the documentation.
    
    # copy third-party license
    root = os.path.abspath(os.path.dirname(__file__))
    source = os.path.join(root, "..", "..", "cmake", "external")
    third = ['googletest', 'protobuf', 'onnx', 'tvm']
    for th in third:
        lic = os.path.join(source, th, "LICENSE")
        if not os.path.exists(lic):
            raise FileNotFoundError(lic)
        dst = th + "_LICENSE"
        print("Copy license of {0}".format(th))
        shutil.copy(lic, os.path.join(root, dst))
    return app
