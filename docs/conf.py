"""Sphinx configuration for the pyNamo-EGT documentation."""

import shutil
from pathlib import Path

import pynamo_egt

DOCS = Path(__file__).resolve().parent
# The tutorial stays at the repository root for the Binder and Colab links;
# copy it into the source tree so MyST-NB executes and renders it.
shutil.copyfile(DOCS.parent / "tutorial.ipynb", DOCS / "tutorial.ipynb")

project = "pyNamo-EGT"
author = "Slimane Dridi and Benjamin Giraudon"
copyright = "2026, Slimane Dridi and Benjamin Giraudon"
release = pynamo_egt.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
exclude_patterns = ["_build", "generated/*.rst.bak"]

autosummary_generate = True
templates_path = ["_templates"]
autodoc_default_options = {"members": True, "show-inheritance": True}
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 600
nb_execution_show_tb = True
nb_merge_streams = True
# ipympl figures have no saved widget state in a static page; show their PNG.
nb_mime_priority_overrides = [("html", "application/vnd.jupyter.widget-view+json", None)]

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
