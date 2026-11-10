"""Sphinx configuration for the pyNamo-EGT documentation."""

import pynamo_egt

project = "pyNamo-EGT"
author = "Slimane Dridi and Benjamin Giraudon"
copyright = "2026, Slimane Dridi and Benjamin Giraudon"
release = pynamo_egt.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "generated/*.rst.bak"]

autosummary_generate = True
templates_path = ["_templates"]
autodoc_default_options = {"members": True, "show-inheritance": True}
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

html_theme = "pydata_sphinx_theme"
html_title = "pyNamo-EGT"
html_theme_options = {
    "github_url": "https://github.com/pynamo-egt/pynamo",
    "navigation_with_keys": False,
}
