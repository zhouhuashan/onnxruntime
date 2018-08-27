# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.

import os
import sys
# Check these extensions were installed.
import recommonmark
import sphinx_gallery.gen_gallery
# The package should be installed in a virtual environment.
import lotus


# -- Project information -----------------------------------------------------

project = 'Lotus'
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
]

templates_path = ['_templates']

source_parsers = {
   '.md': 'recommonmark.parser.CommonMarkParser',
}

source_suffix = ['.rst', '.md']

master_doc = 'index'
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
    # We copy all *md* files in this directory.
    this = os.path.dirname(__file__)
    docs = os.path.join(this, "..")
    for md in filter(lambda n: n.endswith('.md'), os.listdir(docs)):
        with open(os.path.join(docs, md), "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open(os.path.join(this, md), "w", encoding="utf-8") as f:
            start = 1 if "####" in lines[0] else 0
            f.write("".join(lines[start:]))
    for sub in ['media']:
        src = os.path.join(docs, sub)
        dst = os.path.join(this, sub)
        if not os.path.exists(dst):
            os.mkdir(dst)
        for name in filter(lambda n: ".png" in n, os.listdir(src)):
            with open(os.path.join(src, name), "rb") as f:
                content = f.read()
            with open(os.path.join(dst, name), "wb") as f:
                f.write(content)
