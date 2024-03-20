# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------
import sys
from datetime import datetime
from pathlib import Path
import os
pathCurrent=os.path.abspath(Path(__file__))
parent_directory = os.path.dirname(pathCurrent)
parent_directory = os.path.dirname(parent_directory)
parent_directory = os.path.dirname(parent_directory)

sys.path.insert(0, str(parent_directory))

# -- Project information
import TrajAtlas
project = 'TrajAtlas'
copyright = '2024, Litian Han'
author = 'Litian Han'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'recommonmark',
    "sphinx_markdown_tables",
    "sphinxcontrib.bibtex",
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# bibliography
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "alpha"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}

# -- Options for EPUB output
epub_show_urls = 'footnote'

source_parsers = {
    '.md': 'recommonmark.parser.CommonMarkParser',
}

source_suffix = ['.rst', '.md']

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css",
    "css/override.css",
]

html_show_sphinx = False
html_show_sourcelink = False
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "img/logo.png",
    "dark_logo": "img/logo.png",
    "light_css_variables": {
        "color-brand-primary": "#003262",
        "color-brand-content": "#003262",
        "admonition-font-size": "var(--font-size-normal)",
        "admonition-title-font-size": "var(--font-size-normal)",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/GilbertHan1011/TrajAtlas",
            "html": "",
            "class": "fab fa-github",
        },
    ],
}