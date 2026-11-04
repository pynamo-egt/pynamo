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
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
exclude_patterns = [
    "_build",
    # sphinx-gallery inputs and its downloadable notebooks are not pages.
    "examples",
    "auto_examples/*.ipynb",
    "**/sg_execution_times.rst",
]

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

sphinx_gallery_conf = {
    "examples_dirs": "examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"/plot_",
    "within_subsection_order": "FileNameSortKey",
    "download_all_examples": False,
    "remove_config_comments": True,
}

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
