"""
1. GemPy-I: *Simple Fault Model*
================================

This example uses `GemPy <https://www.gempy.org>`_ to create a geological model
as input to emg3d, utilizing `discretize <http://discretize.simpeg.xyz>`_.
Having it in discretize allows us also to plot it with `PyVista
<https://github.com/pyvista>`_.

The starting point is the *simple_fault_model* as used in `Chapter 1.1
<https://docs.gempy.org/tutorials/ch1_fundamentals/ch1_1_basics.html>`_ of the
GemPy documentation. It is a nice, made-up model of a folded structure with a
fault. Here we slightly modify it (convert it into a shallow marine setting),
and create a resisistivity model out of the lithological model.

The result is what is referred to in other examples as model `GemPy-I`, a
synthetic, shallow-marine resistivity model consisting of a folded structure
with a fault. It is one of a few models created to be used in other examples.

**Note:** The original model (*simple_fault_model*) hosted on
https://github.com/cgre-aachen/gempy_data is released under the `LGPL-3.0
License <https://www.gnu.org/licenses/lgpl-3.0.en.html>`_.

"""
import emg3d
import pyvista
import numpy as np
import gempy as gempy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use('bmh')
# sphinx_gallery_thumbnail_number = 3

return  # will break but create the title # TODO Not Updated Yet


###############################################################################
# Get and initiate the *simple_fault_model*
# -----------------------------------------
#
# **Changes made to the original model** (differences between the files
# `simple_fault_model_*.csv` and `simple_fault_model_*_geophy.csv`): Changed
# the stratigraphic unit names, and moved the model 2 km down.
#
# Instead of reading a csv-file we could also initiate an empty instance and
# then add points and orientations after that by, e.g., providing numpy arrays.

# Initiate a model
geo_model = gempy.create_model('GemPy-I')

# Location of data files.
data_url = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/'
data_url += 'master/data/input_data/tut_chapter1/'

# Importing the data from CSV-files and setting extent and resolution
# This is a regular grid, mainly for plotting purposes
gempy.init_data(
    geo_model,
    [0, 2000., 0, 2000., -2000, 40.], [50, 50, 51],
    path_o=data_url+"simple_fault_model_orientations_geophy.csv",
    path_i=data_url+"simple_fault_model_points_geophy.csv",
)


###############################################################################
# Initiate the stratigraphies and faults, and add an air layer
# ------------------------------------------------------------

# Add an air-layer: Horizontal layer at z=0m
geo_model.add_surfaces('air')
geo_model.add_surface_points(0, 0, 0, 'air')
geo_model.add_surface_points(0, 0, 0, 'air')
geo_model.add_orientations(0, 0, 0, 'air', [0, 0, 1])

# Add a Series for the air layer; this series will not be cut by the fault
geo_model.add_series('Air_Series')
geo_model.modify_order_series(2, 'Air_Series')
gempy.map_series_to_surfaces(geo_model, {'Air_Series': 'air'})

# Map the different series
gempy.map_series_to_surfaces(
    geo_model,
    {
        "Fault_Series": 'fault',
        "Air_Series": ('air'),
        "Strat_Series": ('seawater', 'overburden', 'target',
                         'underburden', 'basement')
    },
    remove_unused_series=True
)

geo_model.rename_series({'Main_Fault': 'Fault_Series'})

# Set which series the fault series is cutting
geo_model.set_is_fault('Fault_Series')
geo_model.faults.faults_relations_df


###############################################################################
# Compute the model with GemPy
# ----------------------------

# Set the interpolator.
gempy.set_interpolator(
    geo_model,
    compile_theano=True,
    theano_optimizer='fast_compile',
    verbose=[]
)

# Compute it.
sol = gempy.compute_model(geo_model, compute_mesh=True)

# Plot lithologies (colour-code corresponds to lithologies)
_ = gempy.plot_2d(geo_model, cell_number=25, direction='y', show_data=True)


###############################################################################
# Get id's for a discretize mesh
# ------------------------------
#
# We could define the resistivities before, but currently it is difficult for
# GemPy to interpolate for something like resistivities with a very wide range
# of values (several orders of magnitudes). So we can simply map it here to the
# ``id`` (Note: GemPy does not do interpolation for cells which lie in
# different stratigraphies, so the id is always in integer).

# First we create a detailed discretize-mesh to store the resistivity model and
# use it in other examples as well.
hxy = np.ones(100)*100
hz = np.ones(100)*25
grid = emg3d.TensorMesh([hxy, hxy, hz], x0=(-4000, -4000, -2400))
grid

# Get the solution at cell centers of our grid.
sol = gempy.compute_model(geo_model, at=grid.gridCC)

# Show the surfaces.
geo_model.surfaces


###############################################################################
# Replace id's by resistivities
# -----------------------------

# Now, we convert the id's to resistivities
res = sol.custom[0][0, :grid.n_cells]

res[res == 1] = 1e8  # air
# id=2 is the fault
res[np.round(res) == 3] = 0.3  # sea water
res[np.round(res) == 4] = 1.0  # overburden
res[np.round(res) == 5] = 50   # resistive layer
res[np.round(res) == 6] = 1.5  # underburden
res[np.round(res) == 7] = 200  # resistive basement

# Create an emg3d-model.
model = emg3d.Model(grid, property_x=res, mapping='Resistivity')

###############################################################################
# Plot the model with PyVista
# ---------------------------

dataset = grid.toVTK({'res': np.log10(res)})

# Create the rendering scene and add a grid axes
p = pyvista.Plotter(notebook=True)
p.show_grid(location='outer')

# Add spatially referenced data to the scene
dparams = {'rng': np.log10([0.3, 500]), 'cmap': 'viridis', 'show_edges': False}
xyz = (1500, 500, -1500)
p.add_mesh(dataset.slice('x', xyz), name='x-slice', **dparams)
p.add_mesh(dataset.slice('y', xyz), name='y-slice', **dparams)

# Add a layer as 3D
p.add_mesh(dataset.threshold([1.69, 1.7]), name='vol', **dparams)

# Show the scene!
p.camera_position = [(-10000, 25000, 4000), (1000, 1000, -1000), (0, 0, 1)]
p.show()


###############################################################################
# Plot the model with discretize
# ------------------------------
grid.plot_3d_slicer(
    model.property_x, zslice=-1000,
    pcolor_opts={'cmap': 'viridis', 'norm': LogNorm(vmin=0.3, vmax=500)})


###############################################################################
# Create CSEM survey and corresponding computational grid/model
# -------------------------------------------------------------

# Source location and frequency
src = [1000, 1000, -500, 0, 0]  # x-directed el-source at (1000, 1000, -500)
freq = 1.0                      # Frequency

# Get computation domain as a function of frequency (resp., skin depth)
hx_min, xdomain = emg3d.meshes.get_domain(
        x0=src[0], freq=freq, limits=[0, 2000], min_width=[5, 100])
hz_min, zdomain = emg3d.meshes.get_domain(
        freq=freq, limits=[-2000, 0], min_width=[5, 20], fact_pos=40)

# Create stretched grid
nx = 2**6
hx = emg3d.meshes.get_stretched_h(hx_min, xdomain, nx, src[0])
hy = emg3d.meshes.get_stretched_h(hx_min, xdomain, nx, src[1])
hz = emg3d.meshes.get_stretched_h(hz_min, zdomain, nx*2, x0=src[2], x1=0)
comp_grid = emg3d.TensorMesh(
        [hx, hy, hz], x0=(xdomain[0], xdomain[0], zdomain[0]))

comp_grid

###############################################################################
# Compute the electric field
# --------------------------

# Get the computational model
comp_model = model.interpolate2grid(grid, comp_grid)

# Get the source field
sfield = emg3d.get_source_field(comp_grid, src, freq, 0)

# Compute the efield
efield = emg3d.solve(comp_grid, comp_model, sfield, sslsolver=True, verb=4)

###############################################################################

comp_grid.plot_3d_slicer(
    efield.fx.ravel('F'), zslice=-1000, zlim=(-2000, 50),
    view='abs', v_type='Ex',
    pcolor_opts={'cmap': 'viridis', 'norm': LogNorm(vmin=1e-13, vmax=1e-8)})

###############################################################################
# Store the grid and the model for use in other examples.

# emg3d.save('../data/models/GemPy-I.h5', model=model, mesh=grid)

###############################################################################

emg3d.Report([gempy, pyvista, 'pandas'])
