"""
1. empymod: 1D VTI resistivity
==============================

The code ``empymod`` is an open-source code which can model CSEM responses for
a layered medium including VTI electrical anisotropy, see `emsig.xyz
<https://emsig.xyz>`_.

Content:

1. Full-space VTI model for a finite length, finite strength, rotated bipole:

   a. Regular VTI case
   b. Tri-axial anisotropy check: Swap ``x`` and ``z`` in ``emg3d``; compare
      ``yz``-slice
   c. Tri-axial anisotropy check: Swap ``y`` and ``z`` in ``emg3d``; compare
      ``xz``-slice

2. Layered model for a deep water model with a point dipole source.

.. note::

    You also have to download the file
    :ref:`sphx_glr_gallery_comparisons_plot_comparisons.py` and place it in the
    same directory as this example.

"""
import emg3d
import empymod
import numpy as np
import plot_comparisons as plot  # <= See *Note* above.
# sphinx_gallery_thumbnail_path = '_static/thumbs/empymod-iw.png'


###############################################################################
# 1. Full-space VTI model for a finite length, finite strength, rotated bipole
# ----------------------------------------------------------------------------
#
# In order to shorten the build-time of the gallery we use here a coarse model,
# which will result in bigger errors. If you want better results run the finer
# model.
coarse_model = True


###############################################################################
# Survey and model parameters
# ```````````````````````````

# Receiver coordinates
if coarse_model:
    x = (np.arange(256))*20-2550
else:
    x = (np.arange(1025))*5-2560
rx = np.repeat([x, ], np.size(x), axis=0)
ry = rx.transpose()
frx, fry = rx.ravel(), ry.ravel()
rz = -400.0

# Source coordinates, frequency, and strength
source = emg3d.TxElectricDipole(
    coordinates=[-50, 50, -30, 30, -320., -280.],  # [x1, x2, y1, y2, z1, z2]
    strength=np.pi,  # A
)
frequency = 0.77  # Hz

# Model parameters
h_res = 1.              # Horizontal resistivity
aniso = np.sqrt(2.)     # Anisotropy
v_res = h_res*aniso**2  # Vertical resistivity


###############################################################################
# 1.a Regular VTI case
# ````````````````````
#
# empymod
# ```````
# Note: The coordinate system of empymod is positive z down, for emg3d it is
# positive z up. We have to switch therefore src_z, rec_z, and dip.

# Collect common input for empymod.
inp = {
    'src': np.r_[source.coordinates[:4], -source.coordinates[4:]],
    'depth': [],
    'res': h_res,
    'aniso': aniso,
    'strength': source.strength,
    'srcpts': 5,
    'freqtime': frequency,
    'htarg': {'pts_per_dec': -1},
}

# Compute
epm_ex = empymod.bipole(
    rec=[frx, fry, -rz, 0, 0], verb=3, **inp).reshape(np.shape(rx))
epm_ey = empymod.bipole(
    rec=[frx, fry, -rz, 90, 0], **inp).reshape(np.shape(rx))
epm_ez = empymod.bipole(
    rec=[frx, fry, -rz, 0, -90], **inp).reshape(np.shape(rx))

###############################################################################
# emg3d
# `````

if coarse_model:
    min_width_limits = 40
    stretching = [1.045, 1.045]
else:
    min_width_limits = 20
    stretching = [1.03, 1.045]

# Create stretched grid
grid = emg3d.construct_mesh(
    frequency=frequency,
    properties=h_res,
    center=source.center,
    domain=([-2500, 2500], [-2500, 2500], [-2900, 2100]),
    min_width_limits=min_width_limits,
    stretching=stretching,
    lambda_from_center=True,
    lambda_factor=0.8,
)
grid

###############################################################################

# Define the model
model = emg3d.Model(
    grid, property_x=h_res, property_z=v_res, mapping='Resistivity')

# Compute the electric field
efield = emg3d.solve_source(model, source, frequency, verb=4, plain=True)

###############################################################################
# Plot
# ````

e3d_ex = efield.get_receiver((rx, ry, rz, 0, 0))
plot.plot_sections(
        epm_ex, e3d_ex, x, r'Diffusive Fullspace $E_x$',
        vmin=-12, vmax=-6, mode='abs'
)

###############################################################################
e3d_ey = efield.get_receiver((rx, ry, rz, 90, 0))
plot.plot_sections(
        epm_ey, e3d_ey, x, r'Diffusive Fullspace $E_y$',
        vmin=-12, vmax=-6, mode='abs'
)

###############################################################################
e3d_ez = efield.get_receiver((rx, ry, rz, 0, 90))
plot.plot_sections(
        epm_ez, e3d_ez, x, r'Diffusive Fullspace $E_z$',
        vmin=-12, vmax=-6, mode='abs'
)

###############################################################################
plot.plot_line(x, x, e3d_ex.real, epm_ex.real, grid, 'E_x')


###############################################################################
# 1.b Tri-axial anisotropy check
# ``````````````````````````````
#
# Swap ``x`` and ``z`` in ``emg3d``; compare ``yz``-slice
#
# ``empymod`` can handle VTI, but not tri-axial anisotropy. To verify tri-axial
# anisotropy in ``emg3d``, we swap the ``x`` and ``z`` axes, and compare the
# ``xy``-``empymod`` result to the ``zy``-``emg3d`` result.

# ===> Swap hy and hz; ydomain and zdomain <===
grid2 = emg3d.TensorMesh(
        [grid.h[0], grid.h[2], grid.h[1]],
        origin=(grid.origin[0], grid.origin[2], grid.origin[1])
)

# ===> Swap y- and z-resistivities <===
model2 = emg3d.Model(
    grid2, property_x=h_res, property_y=v_res, mapping='Resistivity')

# ===> Swap src_y and src_z <===
coo = source.coordinates
source2 = emg3d.TxElectricDipole(
    coordinates=[coo[0], coo[1], coo[4], coo[5], coo[2], coo[3]],
    strength=source.strength,
)

efield2 = emg3d.solve_source(model2, source2, frequency, verb=4, plain=True)

###############################################################################
# ===> Swap ry and zrec <===
e3d_ex2 = efield2.get_receiver((rx, rz, ry, 0, 0))
plot.plot_sections(
        epm_ex, e3d_ex2, x, r'Diffusive Fullspace $E_x$',
        vmin=-12, vmax=-6, mode='abs'
)

###############################################################################
# ===> Swap ry and zrec; 'y'->'z' <===
e3d_ey2 = efield2.get_receiver((rx, rz, ry, 0, 90))
plot.plot_sections(
        epm_ey, e3d_ey2, x, r'Diffusive Fullspace $E_y$',
        vmin=-12, vmax=-6, mode='abs'
)

###############################################################################
# ===> Swap ry and zrec; 'z'->'y' <===
e3d_ez2 = efield2.get_receiver((rx, rz, ry, 90, 0))
plot.plot_sections(
        epm_ez, e3d_ez2, x, r'Diffusive Fullspace $E_z$',
        vmin=-12, vmax=-6, mode='abs'
)

###############################################################################
# 1.c Tri-axial anisotropy check
# ``````````````````````````````
#
# Swap ``y`` and ``z`` in ``emg3d``; compare ``xz``-slice
#
# ``empymod`` can handle VTI, but not tri-axial anisotropy. To verify tri-axial
# anisotropy in ``emg3d``, we swap the ``y`` and ``z`` axes, and compare the
# ``xy``-``empymod`` result to the ``xz``-``emg3d`` result.

# ===> Swap hx and hz; xdomain and zdomain <===
grid3 = emg3d.TensorMesh(
        [grid.h[2], grid.h[1], grid.h[0]],
        origin=(grid.origin[2], grid.origin[1], grid.origin[0])
)

# ===> Swap x- and z-resistivities <===
model3 = emg3d.Model(
    grid3, property_x=v_res, property_y=h_res, property_z=h_res,
    mapping='Resistivity'
)

# ===> Swap src_x and src_z <===
source3 = emg3d.TxElectricDipole(
    coordinates=[coo[4], coo[5], coo[2], coo[3], coo[0], coo[1]],
    strength=source.strength,
)

efield3 = emg3d.solve_source(model3, source3, frequency, verb=4, plain=True)

###############################################################################
# ===> Swap rx and zrec; 'x'->'z' <===
e3d_ex3 = efield3.get_receiver((rz, ry, rx, 0, 90))
plot.plot_sections(
        epm_ex, e3d_ex3, x, r'Diffusive Fullspace $E_x$',
        vmin=-12, vmax=-6, mode='abs'
)

###############################################################################
# ===> Swap rx and zrec <===
e3d_ey3 = efield3.get_receiver((rz, ry, rx, 90, 0))
plot.plot_sections(
        epm_ey, e3d_ey3, x, r'Diffusive Fullspace $E_y$',
        vmin=-12, vmax=-6, mode='abs'
)

###############################################################################
# ===> Swap rx and zrec; 'z'->'x' <===
e3d_ez3 = efield3.get_receiver((rz, ry, rx, 0, 0))
plot.plot_sections(
        epm_ez, e3d_ez3, x, r'Diffusive Fullspace $E_z$',
        vmin=-12, vmax=-6, mode='abs'
)


#############################################################################
# 2. Layered model for a deep water model with a point dipole source
# ------------------------------------------------------------------
#
# Survey and model parameters
# ```````````````````````````

# Receiver coordinates
if coarse_model:
    x = (np.arange(256))*20-2550
else:
    x = (np.arange(1025))*5-2560
rx = np.repeat([x, ], np.size(x), axis=0)
ry = rx.transpose()
frx, fry = rx.ravel(), ry.ravel()
rz = -950.0

# Source coordinates and frequency
source = emg3d.TxElectricDipole(coordinates=[0, 0, -900, 0, 0])
frequency = 1.0  # Hz

# Model parameters
h_res = [1, 50, 1, 0.3, 1e12]     # Horizontal resistivity
aniso = np.sqrt([2, 2, 2, 1, 1])  # Anisotropy
v_res = h_res*aniso**2            # Vertical resistivity
depth = np.array([-2200, -2000, -1000, 0])  # Layer boundaries


###############################################################################
# empymod
# ```````

inp = {
    'src': source.coordinates,
    'depth': depth,
    'res': h_res,
    'aniso': aniso,
    'freqtime': frequency,
    'htarg': {'pts_per_dec': -1},
}

# Compute
epm_dx = empymod.bipole(
    rec=[frx, fry, rz, 0, 0], verb=3, **inp).reshape(np.shape(rx))
epm_dy = empymod.bipole(
    rec=[frx, fry, rz, 90, 0], **inp).reshape(np.shape(rx))
epm_dz = empymod.bipole(
    rec=[frx, fry, rz, 0, 90], **inp).reshape(np.shape(rx))


###############################################################################
# emg3d
# `````

if coarse_model:
    min_width_limits = 40
else:
    min_width_limits = 20

# Create stretched grid
grid = emg3d.construct_mesh(
    frequency=frequency,
    properties=[h_res[3], h_res[0]],
    center=source.center,
    domain=([-2500, 2500], [-2500, 2500], None),
    vector=(None, None, -2200 + np.arange(111)*20),
    min_width_limits=min_width_limits,
    stretching=[1.1, 1.5],
)
grid

###############################################################################

# Create the model: horizontal resistivity
res_x_full = h_res[0]*np.ones(grid.n_cells)  # Background
res_x_full[grid.cell_centers[:, 2] >= depth[0]] = h_res[1]  # Target
res_x_full[grid.cell_centers[:, 2] >= depth[1]] = h_res[2]  # Overburden
res_x_full[grid.cell_centers[:, 2] >= depth[2]] = h_res[3]  # Water
res_x_full[grid.cell_centers[:, 2] >= depth[3]] = h_res[4]  # Air

# Create the model: vertical resistivity
res_z_full = v_res[0]*np.ones(grid.n_cells)  # Background
res_z_full[grid.cell_centers[:, 2] >= depth[0]] = v_res[1]
res_z_full[grid.cell_centers[:, 2] >= depth[1]] = v_res[2]
res_z_full[grid.cell_centers[:, 2] >= depth[2]] = v_res[3]
res_z_full[grid.cell_centers[:, 2] >= depth[3]] = v_res[4]

# Get the model
model = emg3d.Model(
        grid, property_x=res_x_full, property_z=res_z_full,
        mapping='Resistivity')

###############################################################################

# Compute the electric field
efield = emg3d.solve_source(model, source, frequency, verb=4)


###############################################################################
# Plot
# ````
e3d_dx = efield.get_receiver((rx, ry, rz, 0, 0))
plot.plot_sections(
        epm_dx, e3d_dx, x, r'Deep water point dipole $E_x$',
        vmin=-14, vmax=-8, mode='abs'
)


###############################################################################
e3d_dy = efield.get_receiver((rx, ry, rz, 90, 0))
plot.plot_sections(
        epm_dy, e3d_dy, x, r'Deep water point dipole $E_y$',
        vmin=-14, vmax=-8, mode='abs'
)


###############################################################################
e3d_dz = efield.get_receiver((rx, ry, rz, 0, 90))
plot.plot_sections(
        epm_dz, e3d_dz, x, r'Deep water point dipole $E_z$',
        vmin=-14, vmax=-8, mode='abs'
)

###############################################################################
plot.plot_line(x, x, e3d_dx.real, epm_dx.real, grid, 'E_x')


###############################################################################

emg3d.Report()
