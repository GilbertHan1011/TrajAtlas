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
sys.path.insert(0, os.path.join(os.path.dirname(pathCurrent), "_ext"))
#sys.path.insert(0, str(Path(__file__).parent / "_ext"))

# -- Project information
import TrajAtlas
project = 'TrajAtlas'
copyright = '2024, Litian Han'
author = 'Litian Han'

release = '0.1'
version = '1.0.0'

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
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "myst_nb",
    "sphinx_autodoc_typehints",
    "sphinx_tippy",
    "sphinx_design",
    "_typed_returns",
]



intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
    "pygam": ("https://pygam.readthedocs.io/en/latest/", None),
    "pygpcca": ("https://pygpcca.readthedocs.io/en/latest/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "anndata": ("https://anndata.readthedocs.io/en/latest/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/latest/", None),
    "scvelo": ("https://scvelo.readthedocs.io/en/latest/", None),
    "squidpy": ("https://squidpy.readthedocs.io/en/latest/", None),
    "moscot": ("https://moscot.readthedocs.io/en/latest/", None),
    "ot": ("https://pythonot.github.io/", None),
}
#intersphinx_disabled_domains = ['std']
master_doc = "index"
pygments_style = "tango"
pygments_dark_style = "monokai"

nitpicky = True

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

# myst
nb_execution_mode = "off"
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
]
myst_heading_anchors = 2

# hover
tippy_anchor_parent_selector = "div.content"
tippy_enable_mathjax = True
# no need because of sphinxcontrib-bibtex
tippy_enable_doitips = False

# autodoc + napoleon
autosummary_generate = True
autodoc_member_order = "alphabetical"
autodoc_typehints = "description"
autodoc_mock_imports = ["moscot"]
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# spelling
spelling_lang = "en_US"
spelling_warning = True
spelling_word_list_filename = "spelling_wordlist.txt"
spelling_add_pypi_package_names = True
spelling_exclude_patterns = ["references.rst"]
# see: https://pyenchant.github.io/pyenchant/api/enchant.tokenize.html
spelling_filters = [
    "enchant.tokenize.URLFilter",
    "enchant.tokenize.EmailFilter",
    "enchant.tokenize.MentionFilter",
]

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