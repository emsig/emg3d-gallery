import time
import warnings
from emg3d import __version__
from sphinx_gallery.sorting import ExplicitOrder, ExampleTitleSortKey

# ==== 1. Extensions  ====

# Load extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.intersphinx',
    'numpydoc',
    'sphinx_gallery.gen_gallery',
]

# Numpydoc settings
numpydoc_show_class_members = False
# numfig = True
# numfig_format = {'figure': 'Figure %s:'}

# Todo settings
todo_include_todos = True

# Sphinx gallery configuration
sphinx_gallery_conf = {
    'examples_dirs': '../examples',
    'gallery_dirs': 'gallery',
    'subsection_order': ExplicitOrder([
        '../examples/tutorials',
        '../examples/comparisons',
        '../examples/magnetics',
        '../examples/time_domain',
        '../examples/tools',
        '../examples/reproducing',
        '../examples/interactions',
        ]),
    'capture_repr': ('_repr_html_', '__repr__'),
    # Patter to search for example files
    "filename_pattern": r"\.py",
    # Sort gallery example by file name instead of number of lines (default)
    "within_subsection_order": ExampleTitleSortKey,
    # Remove the settings (e.g., sphinx_gallery_thumbnail_number)
    'remove_config_comments': True,
    # Show memory
    'show_memory': True,
    # Custom first notebook cell
    'first_notebook_cell': '%matplotlib notebook',
    # PyVista scraper
    'image_scrapers': ('matplotlib', 'pyvista', ),
}

# https://github.com/sphinx-gallery/sphinx-gallery/pull/521/files
# Remove matplotlib agg warnings from generated doc when using plt.show
warnings.filterwarnings("ignore", category=UserWarning,
                        message='Matplotlib is currently using agg, which is a'
                                ' non-GUI backend, so cannot show the figure.')

# Intersphinx configuration
intersphinx_mapping = {
    "numpy": ("https://docs.scipy.org/doc/numpy", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "discretize": ("https://discretize.simpeg.xyz/en/master", None),
    "empymod": ("https://empymod.readthedocs.io/en/stable", None),
    "emg3d": ("https://emg3d.readthedocs.io/en/stable", None),
}

# ==== 2. General Settings ====
description = 'A multigrid solver for 3D electromagnetic diffusion.'

# The templates path.
templates_path = ['_templates']

# The suffix(es) of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'emg3d-gallery'
copyright = u'2018-{}, The emg3d Developers.'.format(time.strftime("%Y"))
author = 'The emg3d Developers'

# |version| and |today| tags (|release|-tag is not used).
version = __version__
release = __version__
today_fmt = '%d %B %Y'

# List of patterns to ignore, relative to source directory.
exclude_patterns = ['_build', ]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'friendly'

# ==== 3. HTML settings ====
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'both',
}
html_static_path = ['_static']
html_logo = '_static/logo-emg3d-cut.svg'
html_favicon = '_static/favicon.ico'
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'searchbox.html',
    ]
}

html_context = {
    'menu_links_name': 'Links',
    'menu_links': [
        ('<i class="fa fa-link fa-fw"></i> Website',
         'https://emsig.github.io'),
        ('<i class="fa fa-github fa-fw"></i> Source Code',
         'https://github.com/emsig/emg3d-gallery'),
    ],
}

htmlhelp_basename = 'emg3d-gallery-doc'


# -- CSS fixes --
def setup(app):
    app.add_css_file("style.css")
