"""
2. Simulation
=============

A basic example how to use the :class:`emg3d.surveys.Survey`- and
:class:`emg3d.simulations.Simulation`-classes to model data for an entire
survey, hence many sources and frequencies.

For this example we use the resistivity model created in the example
:ref:`sphx_glr_gallery_interactions_gempy-ii.py`.

"""
import os
import emg3d
import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import RectBivariateSpline
plt.style.use('ggplot')


###############################################################################
# Load Model
# ----------

mname = 'GemPy-II'
fname = mname+'.h5'
if not os.path.isfile(fname):
    url = ('https://github.com/empymod/emg3d-gallery/blob/master/examples/'
           'data/models/GemPy-II.h5?raw=true')
    with open(fname, 'wb') as f:
        t = requests.get(url)
        f.write(t.content)

data = emg3d.load(fname)
model, mesh = data['model'], data['mesh']


###############################################################################
# Let's check the model

model


###############################################################################
# So it is an isotropic model defined in terms of resistivities. Let's check
# the mesh

mesh


###############################################################################
# Define the survey
# -----------------
#
# If you have actual field data then this info would normally come from a data
# file or similar. Here we create our own dummy survey, and later will create
# synthetic data for it.
#
# A **Survey** instance contains all survey-related information, hence source
# and receiver positions and measured data. See the relevant documentation for
# more details: :class:`emg3d.surveys.Survey`.
#
#
# Extract seafloor to simulate source and receiver depths
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''
#
# To create a realistic survey we create a small routine that finds the
# seafloor, so we can place receivers on the seafloor and sources 50 m above
# it. We use the fact that the seawater has resistivity of 0.3 Ohm.m in the
# model, and is the lowest value.

seafloor = np.ones((mesh.nCx, mesh.nCy))
for i in range(mesh.nCx):
    for ii in range(mesh.nCy):
        # We take the seafloor to be the first cell which resistivity
        # is below 0.33
        seafloor[i, ii] = mesh.vectorNz[:-1][
                model.property_x[i, ii, :] < 0.33][0]

# Create a 2D interpolation function from it
bathymetry = RectBivariateSpline(mesh.vectorCCx, mesh.vectorCCy, seafloor)


###############################################################################
# Source and receiver positions
# '''''''''''''''''''''''''''''
#
# Sources and receivers can be defined in a few different ways. One way is by
# providing coordinates, where two coordinate formats are accepted:
#
# - ``(x0, x1, y0, y1, z0, z1)``: finite length dipole,
# - ``(x, y, z, azimuth, dip)``: point dipole.
#
# A survey can contain electric and magnetic receivers, arbitrarily rotated.
# However, the ``Simulation`` is currently still limited to electric receivers,
# and the ``optimization`` later on (gradient) is currently limited to
# x-directed electric dipoles. As we will use this example also in
# :ref:`sphx_glr_gallery_tutorials_gradient.py` we stick to x-directed electric
# dipoles at the moment.
#
# Note that the survey just knows about the sources, receivers, frequencies,
# and observed data - it does not know anything of an underlying model.

# For now just horizontal Ex point dipoles.
# Angles in degrees (see
# `coordinate_system
# <https://empymod.readthedocs.io/en/stable/examples/coordinate_system.html>`_).
dip = 0.0
azimuth = 0.0

# Acquisition source frequencies (Hz)
frequencies = [0.5, 1.0]

# Source coordinates
src_x = np.arange(1, 4)*5000
src_y = 7500
# Source depths: 50 m above seafloor
src_z = bathymetry(src_x, src_y).ravel()+50
src = (src_x, src_y, src_z, dip, azimuth)

# Receiver positions
rec_x = np.arange(3, 18)*1e3
rec_y = np.arange(3)*1e3+6500
RX, RY = np.meshgrid(rec_x, rec_y, indexing='ij')
RZ = bathymetry(rec_x, rec_y)
rec = (RX.ravel(), RY.ravel(), RZ.ravel(), dip, azimuth)


###############################################################################
# Create Survey
# '''''''''''''

survey = emg3d.surveys.Survey(
    name='GemPy-II Survey A',  # Name of the survey
    sources=src,               # Source coordinates
    receivers=rec,             # Receiver coordinates
    frequencies=frequencies,   # Two frequencies
    # data=data,               # If you have observed data
)

# Let's have a look at the survey:
survey


###############################################################################
# Our survey has our sources and receivers and initiated a variable
# ``observed``, with NaN's. Each source and receiver got a named assigned. If
# you prefer other names you would have to define the sources and receivers
# through ``emg3d.surveys.Dipole``, and provide a list of dipoles to the survey
# instead of only a tuple of coordinates.
#
# We can also look at a particular source or receiver, e.g.,

survey.sources['Tx1']


###############################################################################
# Which shows you all you need to know about a particular dipole (name, type
# [electric or magnetic], coordinates of its center, angles, and length).
#
# QC model and survey
# -------------------

mesh.plot_3d_slicer(model.property_x, xslice=12000, yslice=7000,
                    pcolor_opts={'norm': LogNorm(vmin=0.3, vmax=200)})

# Plot survey in figure above
fig = plt.gcf()
fig.suptitle('Resistivity model (Ohm.m) and survey layout')
axs = fig.get_children()
axs[1].plot(survey.rec_coords[0], survey.rec_coords[1], 'bv')
axs[2].plot(survey.rec_coords[0], survey.rec_coords[2], 'bv')
axs[3].plot(survey.rec_coords[2], survey.rec_coords[1], 'bv')
axs[1].plot(survey.src_coords[0], survey.src_coords[1], 'r*')
axs[2].plot(survey.src_coords[0], survey.src_coords[2], 'r*')
axs[3].plot(survey.src_coords[2], survey.src_coords[1], 'r*')
plt.show()


###############################################################################
# Create computational mesh
# -------------------------
#
# In the not-so-distant future the simulation class will have some automatic,
# source- and frequency-dependent gridding included. But currently we still
# have to define the computational grid manually and provide it. You can define
# it yourself or make use of some of the helper routines.

gridinput = {'freq': 1.0, 'verb': 0}

# Get cell widths and origin in each direction
xx, x0 = emg3d.meshes.get_hx_h0(
    res=[0.3, 10], fixed=survey.src_coords[0][1], min_width=100,
    domain=[survey.rec_coords[0].min()-100, survey.rec_coords[0].max()+100],
    **gridinput)
yy, y0 = emg3d.meshes.get_hx_h0(
    res=[0.3, 10], fixed=survey.src_coords[1][1], min_width=100,
    domain=[survey.rec_coords[1].min()-100, survey.rec_coords[1].max()+100],
    **gridinput)
zz, z0 = emg3d.meshes.get_hx_h0(
    res=[0.3, 1., 0.3], domain=[-5500, -2000], min_width=50,
    alpha=[1.05, 1.5, 0.01], fixed=[-2200, -2400, -2000], **gridinput)

# Initiate mesh.
comp_grid = emg3d.TensorMesh([xx, yy, zz], x0=np.array([x0, y0, z0]))
comp_grid


###############################################################################

# # To QC the computational mesh:
# res = model.interpolate2grid(mesh, comp_grid).property_x
# comp_grid.plot_3d_slicer(
#         res, pcolor_opts={'norm': LogNorm(vmin=0.3, vmax=200)})


###############################################################################
# Create a Simulation (to compute 'observed' data)
# ------------------------------------------------
#
# The simulation class combines a model with a survey, and can compute
# synthetic data for it.

simulation = emg3d.simulations.Simulation(
    name="True Model",   # A name for this simulation
    survey=survey,       # Our survey instance
    grid=mesh,           # The model mesh
    model=model,         # The model
    gridding=comp_grid,  # The computational mesh
    max_workers=4,       # How many parallel jobs
    # solver_opts,       # Any parameter to pass to emg3d.solve
)

# Let's QC our Simulation instance
simulation


###############################################################################
# Compute the data
# ''''''''''''''''
#
# We pass here the argument ``observed=True``; this way, the synthetic data is
# stored in our Survey as ``observed`` data, otherwise it would be stored as
# ``synthetic``. This is important later for optimization.
#
# This computes all results in parallel; in this case six models, three sources
# times two frequencies. You can change the number of workers at any time by
# setting ``simulation.max_workers``.

simulation.compute(observed=True)


###############################################################################
# A ``Simulation`` has a few convenience functions, e.g.:
#
# - ``simulation.get_efield('Tx1', 0.5)``: Returns the electric field of the
#   entire domain for source ``'Tx1'`` and frequency 0.5 Hz.
# - ``simulation.get_hfield``; ``simulation.get_sfield``: Similar functions to
#   retrieve the magnetic fields and the source fields.
# - ``simulation.get_model``; ``simulation.get_grid``: Similar functions to
#   retrieve the computational grid and the model for a given source and
#   frequency. As we use the same grid in our example for all sources and all
#   frequencies this is not particular useful, but for source- and
#   frequency-dependent gridding it can prove useful.
#
# When we now look at our survey we see that the observed data variable is
# filled with the responses at the receiver locations.

survey


###############################################################################
# QC Data
# -------

plt.figure()
plt.title("Inline receivers for all sources")
data = simulation.data.observed[:, 1::3, :]
for i, src in enumerate(survey.sources.keys()):
    for ii, freq in enumerate(survey.frequencies):
        plt.plot(survey.rec_coords[0][1::3],
                 abs(data.loc[src, :, freq].data.real),
                 f"C{ii}.-",
                 label=f"|Real|; freq={freq} Hz" if i == 0 else None
                 )
        plt.plot(survey.rec_coords[0][1::3],
                 abs(data.loc[src, :, freq].data.imag),
                 f"C{ii}.--",
                 label=f"|Imag|; freq={freq} Hz" if i == 0 else None
                 )

plt.yscale('log')
plt.legend(ncol=2, framealpha=1)
plt.xlabel('x-coordinate (m)')
plt.ylabel('$|E_x|$ (V/m)')
plt.show()


###############################################################################
# How to store surveys and simulations to disk
# --------------------------------------------
#
# Survey and Simulations can store (and load) themselves to (from) disk.
#
# - A survey stores all sources, receivers, frequencies, and the observed data.
# - A simulation stores the survey, the model, the synthetic data. (It can also
#   store much more, such as all electric fields, source and frequency
#   dependent meshes and models, etc. What it actually stores is defined by the
#   parameter ``what``).

# Survey file name
survey_fname = '../data/surveys/'+mname+'-survey-A.h5'

# To store, run
# survey.to_file(survey_fname)  # .h5, .json, or .npz

# To load, run
# survey = emg3d.surveys.Survey.from_file(survey_fname)

# In the same manner you could store and load the entire simulation:

# Simulation file name
# simulation_fname = file-name.ending  # for ending in [h5, json, npz]

# To store, run
# simulation.to_file(simulation_fname, what='results')

# To load, run
# simulation = emg3d.simulations.Simulation.from_file(simulation_fname)

###############################################################################

emg3d.Report()
