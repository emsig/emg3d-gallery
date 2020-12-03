"""
2. GemPy-II: *Perth Basin*
==========================

This example is mainly about building a deep marine resistivity model that can
be used in other examples. There is not a lot of explanation. For more details
regarding the integration of `GemPy` and `emg3d` see the
:ref:`sphx_glr_gallery_interactions_gempy-i.py`, and make sure to consult the
many useful information over at `GemPy <https://www.gempy.org>`_.

The model is based on the `Perth Basin Model
<https://docs.gempy.org/examples/real/Perth_basin.html>`_ from GemPy. We take
the model, assign resistivities to the lithologies, create a random topography,
move it 2 km down, fill it up with sea water, and add an air layer. The result
is what is referred to in other examples as model `GemPy-II`, a synthetic,
deep-marine resistivity model.

**Note:** The original model (*Perth_Basin*) hosted on
https://github.com/cgre-aachen/gempy_data is released under the `LGPL-3.0
License <https://www.gnu.org/licenses/lgpl-3.0.en.html>`_.

"""
import emg3d
import pyvista
import requests
import numpy as np
import gempy as gempy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use('ggplot')


###############################################################################
# Get and initiate the *Perth Basin*
# ----------------------------------

# Initiate a model
geo_model = gempy.create_model('GemPy-II')

url_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/'
url_path += 'master/data/input_data/Perth_basin/'

# Define the grid
nx, ny, nz = 100, 100, 100
extent = [337000, 400000, 6640000, 6710000, -12000, 1000]

# Importing the data from CSV-files and setting extent and resolution
gempy.init_data(
    geo_model,
    extent=extent,
    resolution=[nx, ny, nz],
    path_i=url_path+"Paper_GU2F_sc_faults_topo_Points.csv",
    path_o=url_path+"Paper_GU2F_sc_faults_topo_Foliations.csv",
)


###############################################################################
# Initiate the stratigraphies and faults
# ------------------------------------------------------------
#
#

# We just follow the example here
del_surfaces = ['Cadda', 'Woodada_Kockatea', 'Cattamarra']
geo_model.delete_surfaces(del_surfaces)

# Map the different series
gempy.map_series_to_surfaces(
    geo_model,
    {
        "fault_Abrolhos_Transfer": ["Abrolhos_Transfer"],
        "fault_Coomallo": ["Coomallo"],
        "fault_Eneabba_South": ["Eneabba_South"],
        "fault_Hypo_fault_W": ["Hypo_fault_W"],
        "fault_Hypo_fault_E": ["Hypo_fault_E"],
        "fault_Urella_North": ["Urella_North"],
        "fault_Urella_South": ["Urella_South"],
        "fault_Darling": ["Darling"],
        "Sedimentary_Series": ['Cretaceous', 'Yarragadee', 'Eneabba',
                               'Lesueur', 'Permian']
    }
)

order_series = ["fault_Abrolhos_Transfer",
                "fault_Coomallo",
                "fault_Eneabba_South",
                "fault_Hypo_fault_W",
                "fault_Hypo_fault_E",
                "fault_Urella_North",
                "fault_Darling",
                "fault_Urella_South",
                "Sedimentary_Series",
                "Basement"]

_ = geo_model.reorder_series(order_series)

# Drop input data from the deleted series:
geo_model.surface_points.df.dropna(inplace=True)
geo_model.orientations.df.dropna(inplace=True)

# Set faults
geo_model.set_is_fault(["fault_Abrolhos_Transfer",
                        "fault_Coomallo",
                        "fault_Eneabba_South",
                        "fault_Hypo_fault_W",
                        "fault_Hypo_fault_E",
                        "fault_Urella_North",
                        "fault_Darling",
                        "fault_Urella_South"])
fr = geo_model.faults.faults_relations_df.values
fr[:, :-2] = False
_ = geo_model.set_fault_relation(fr)


###############################################################################
# Compute the model with GemPy
# ----------------------------

# Set the interpolator.
gempy.set_interpolator(
    geo_model,
    compile_theano=True,
    theano_optimizer='fast_run',
    gradient=False,
    dtype='float32',
    verbose=[]
)

# Compute it.
sol = gempy.compute_model(geo_model, compute_mesh=True)

# Get the solution at the internal grid points.
sol = gempy.compute_model(geo_model)


###############################################################################
# Assign resistivities to the id's
# --------------------------------
#
# We define here a discretize mesh identical to the mesh used by GemPy, and
# subsequently assign resistivities to the different lithologies.
#
# Please note that these resistivities are made up, and do not necessarily
# relate to the actual lithologies.

# We create a mesh 20 km x 20 km x 5 km, starting at the origin.
# As long as we have the same number of cells we can trick the original grid
# into any grid we want.
hx = np.ones(nx)*20000/nx
hy = np.ones(ny)*20000/ny
hz = np.ones(nz)*5000/nz
grid = emg3d.TensorMesh([hx, hy, hz], x0=[0, 0, -5000])

# Make up some resistivities that might be interesting to model.
ids = np.round(sol.lith_block)
res = np.ones(grid.nC)
res[ids == 9] = 2.0    # Cretaceous
res[ids == 10] = 1.0   # Yarragadee
res[ids == 11] = 4.0   # Eneabba
res[ids == 12] = 50.0  # Lesueur
res[ids == 13] = 7.0   # Permian
res[ids == 14] = 10.0  # Basement


###############################################################################
# Topography
# ----------
#
# Calls to ``geo_model.set_topography(source='random')`` create a random
# topography every time. In order to have it reproducible we saved it once and
# load it now.
#
# Originally it was created and stored like this:
#
# .. code::
#
#     out = geo_model.set_topography(source='random')
#     np.save('../data/GemPy/'+topo_name, topo)

# Load the stored topography.
topo_name = 'GemPy-II-topo.npy'
topo_path = 'https://github.com/empymod/emg3d-gallery/blob/master/'
topo_path += 'examples/data/GemPy/'+topo_name+'?raw=true'
with open(topo_name, 'wb') as f:
    t = requests.get(topo_path)
    f.write(t.content)
out = geo_model.set_topography(source='saved', filepath=topo_name)
topo = out.topography.values_2d

# Apply the topography to our resistivity cube.
res = res.reshape(grid.vnC, order='C')

# Get the scaling factor between the original extent and our made-up extent.
fact = 5000/np.diff(extent[4:])

# Loop over all x-y-values and convert cells above topography to water.
for ix in range(nx):
    for iy in range(ny):
        res[ix, iy, grid.cell_centers_z > topo[ix, iy, 2]*fact] = 0.3


###############################################################################
# Extend the model by sea water and air
# -------------------------------------

# Create an emg3d-model.
model = emg3d.Model(grid, property_x=res.ravel('F'), mapping='Resistivity')

# Add 2 km water and 500 m air.
fhz = np.r_[np.ones(nz)*5000/nz, 2000, 500]
z0 = -7000

# Make the full mesh
fullgrid = emg3d.TensorMesh([hx, hy, fhz], x0=[0, 0, z0])

# Extend the model.
fullmodel = emg3d.Model(fullgrid, np.ones(fullgrid.vnC), mapping='Resistivity')
fullmodel.property_x[:, :, :-2] = model.property_x
fullmodel.property_x[:, :, -2] = 0.3
fullmodel.property_x[:, :, -1] = 1e8

fullgrid

###############################################################################
# Plot the model
# ------------------------------

# With discretize
fullgrid.plot_3d_slicer(
    fullmodel.property_x, zslice=-3000, xslice=12000,
    pcolor_opts={'cmap': 'viridis', 'norm': LogNorm(vmin=0.3, vmax=100)}
)


# With PyVista
dataset = fullgrid.toVTK({'res': np.log10(fullmodel.property_x.ravel('F'))})

# Create the rendering scene and add a grid axes
p = pyvista.Plotter(notebook=True)
p.show_grid(location='outer')

# Add spatially referenced data to the scene
dparams = {'rng': np.log10([0.3, 500]), 'cmap': 'viridis', 'show_edges': False}
xyz = (17500, 17500, -1500)
p.add_mesh(dataset.slice('x', xyz), name='x-slice', **dparams)
p.add_mesh(dataset.slice('y', xyz), name='y-slice', **dparams)

# Add a layer as 3D
p.add_mesh(dataset.threshold(
    [np.log10(49.9), np.log10(50.1)]), name='vol', **dparams)

# Show the scene!
p.camera_position = [(-10000, -41000, 8500), (10000, 10000, -3250), (0, 0, 1)]
p.show()


###############################################################################
# Store the grid and the model for use in other examples.

# emg3d.save('../data/models/GemPy-II.h5', model=fullmodel, mesh=fullgrid)

###############################################################################

emg3d.Report([gempy, pyvista, 'pandas'])
