"""Configuration for sphinx."""

# -- Path setup --------------------------------------------------------------

import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath("../../"))
sys.path.append(os.path.abspath("./_ext"))
sys.setrecursionlimit(1500)


# -- Project information -----------------------------------------------------

project = "aim2dat"
copyright = "2024, aim2dat developers"  # noqa: A001
author = "aim2dat developers"
language = "en"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx_immaterial",
    "sphinx_design",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.extlinks",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "sphinxcontrib.contentui",
    "autodoc_aiida_process",
]
source_suffix = [".rst", ".md", ".html"]
exclude_patterns = ["_*"]

# -- autoapi configuration ---------------------------------------------------

autoapi_dirs = ["../../aim2dat"]
# autoapi_root = "python_api"
autoapi_template_dir = "_templates/autoapi"
autoapi_type = "python"
autoapi_add_toctree_entry = False
autoapi_keep_files = True
autoapi_options = [
    "members",
    "inherited-members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_ignore = [
    "*base_*",
    "*backend*",
]
autoapi_member_order = "groupwise"
nonpublic_subpackages = [
    "aim2dat.aiida_workflows.chargemol",
    "aim2dat.aiida_workflows.cp2k",
    "aim2dat.aiida_workflows.critic2",
    "aim2dat.aiida_workflows.enumlib",
    "aim2dat.ext_interfaces",
    "aim2dat.utils.data",
]
nonpublic_submodules = [
    "aim2dat.function_analysis.discretization",
    "aim2dat.function_analysis.fingerprint",
    "aim2dat.function_analysis.function_comparison",
    "aim2dat.io.auxiliary_functions",
    "aim2dat.io.cp2k.bands_dos",
    "aim2dat.io.cp2k.restart",
    "aim2dat.io.cp2k.stdout",
    "aim2dat.plots.band_structure_dos",
    "aim2dat.plots.partial_charges",
    "aim2dat.plots.partial_rdf",
    "aim2dat.plots.phase",
    "aim2dat.plots.planar_fields",
    "aim2dat.plots.simple_plot",
    "aim2dat.plots.spectroscopy",
    "aim2dat.plots.surface",
    "aim2dat.plots.thermal_properties",
    "aim2dat.strct.brillouin_zone_2d",
    "aim2dat.strct.mixin",
    "aim2dat.strct.strct_comparison",
    "aim2dat.strct.strct_coordination",
    "aim2dat.strct.strct_fragmentation",
    "aim2dat.strct.strct_manipulation",
    "aim2dat.strct.strct_misc",
    "aim2dat.strct.strct_planes",
    "aim2dat.strct.strct_point_groups",
    "aim2dat.strct.strct_prdf",
    "aim2dat.strct.strct_space_groups",
    "aim2dat.strct.strct_super_cell",
    "aim2dat.strct.strct_validation",
    "aim2dat.strct.strct_warren_cowley_order_parameters",
    "aim2dat.strct.strct",
    "aim2dat.strct.structure_collection",
    "aim2dat.strct.structure_importer",
    "aim2dat.strct.structure_operations",
    "aim2dat.strct.surface_utils",
    "aim2dat.strct.surface",
    "aim2dat.strct.ext_analysis.dscribe_descriptors",
    "aim2dat.strct.ext_analysis.ffprint_order_p",
    "aim2dat.strct.ext_analysis.fragmentation",
    "aim2dat.strct.ext_analysis.graphs",
    "aim2dat.strct.ext_analysis.planes",
    "aim2dat.strct.ext_analysis.prdf",
    "aim2dat.strct.ext_analysis.warren_cowley_order_parameters",
    "aim2dat.strct.ext_manipulation.add_functional_group",
    "aim2dat.strct.ext_manipulation.add_structure",
    "aim2dat.strct.ext_manipulation.rotate_structure",
    "aim2dat.strct.ext_manipulation.translate_structure",
    "aim2dat.strct.ext_manipulation.utils",
    "aim2dat.strct.stability",
    "aim2dat.utils.print",
]


# autoapi jinja adjustments:
def contains(seq, item):
    """Contains function."""
    return item in seq


def prepare_jinja_env(jinja_env) -> None:
    """Jinja envs."""
    jinja_env.tests["contains"] = contains


autoapi_prepare_jinja_env = prepare_jinja_env


# -- extlinks configuration --------------------------------------------------

extlinks = {
    "doi": ("https://dx.doi.org/%s", "doi:%s"),
}

# -- Napoleon configuration --------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_use_keyword = False

# -- nbsphinx configuration ---------------------------------------------------

nbsphinx_execute = "always"
nbsphinx_allow_errors = False

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_immaterial"
html_theme_options = {
    "font": False,
    "icon": {
        "repo": "fontawesome/brands/github",
    },
    # Set the color and the accent color
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "green",
            "accent": "blue",
            "toggle": {
                "icon": "material/weather-night",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "green",
            "accent": "blue",
            "toggle": {
                "icon": "material/weather-sunny",
                "name": "Switch to light mode",
            },
        },
    ],
    "globaltoc_collapse": False,
    "toc_title_is_page_title": False,
    "features": [
        "navigation.tabs",
        "navigation.tabs.sticky",
        "navigation.top",
        "navigation.tracking",
        "toc.follow",
        "toc.sticky",
        "content.tabs.link",
        "announce.dismiss",
    ],
    "repo_url": "https://github.com/aim2dat/aim2dat",
    "repo_name": "aim2dat",
}


def skip_members(app, what, name, obj, skip, options):
    """Skip members function."""
    if what == "module":
        if name in nonpublic_submodules:
            skip = True
    elif what == "package":
        if name in nonpublic_subpackages:
            skip = True
    return skip


def setup(sphinx):
    """Set up sphinx."""
    sphinx.connect("autoapi-skip-member", skip_members)
