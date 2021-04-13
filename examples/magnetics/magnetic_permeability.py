r"""
4. Magnetic permeability
========================

The solver ``emg3d`` uses the diffusive approximation of Maxwell's equations;
the relative electric permittivity is therefore fixed at
:math:`\varepsilon_\rm{r} = 1`. The magnetic permeability :math:`\mu_\rm{r}`,
however, is implemented in ``emg3d``, albeit only isotropically.

In this example we run the same model as in the example mentioned above: A
rotated finite length bipole in a homogeneous VTI fullspace, but here with
isotropic magnetic permeability, and compare it to the semi-analytical solution
of ``empymod``. (The code ``empymod`` is an open-source code which can model
CSEM responses for a layered medium including VTI electrical anisotropy, see
`emsig.xyz <https://emsig.xyz>`_.)

This is an adapted version of
:ref:`sphx_glr_gallery_comparisons_1D_VTI_empymod.py`. Consult that example to
see the result for the electric field.

.. note::

    You also have to download the file
    :ref:`sphx_glr_gallery_magnetics_plot_magnetics.py` and place it in the
    same directory as this example.

"""
import emg3d
import empymod
import numpy as np
import plot_magnetics as plot  # <= See *Note* above.
# sphinx_gallery_thumbnail_path = '_static/thumbs/magn-perm.png'


###############################################################################
# Full-space model for a finite length, finite strength, rotated bipole
# ---------------------------------------------------------------------
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
mperm = 2.5             # Magnetic permeability


###############################################################################
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
    'mpermH': mperm,  # <= Magnetic permeability
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

# Define the model, with magnetic permeability
model = emg3d.Model(
    grid, property_x=h_res, property_z=v_res,
    mu_r=mperm, mapping='Resistivity'
)

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
emg3d.Report()
