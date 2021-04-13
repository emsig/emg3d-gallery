r"""
1. Magnetic field due to an el. source
======================================

The solver ``emg3d`` returns the electric field in x-, y-, and z-direction.
Using Farady's law of induction we can obtain the magnetic field from it.
Faraday's law of induction in the frequency domain can be written as, in its
differential form,

.. math::
    :label: faraday

    \nabla \times \mathbf{E} = \rm{i}\omega \mathbf{B} =
    \rm{i}\omega\mu\mathbf{H}\, .

This is exactly what we do in this example, for a rotated finite length bipole
in a homogeneous VTI fullspace, and compare it to the semi-analytical solution
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
# sphinx_gallery_thumbnail_path = '_static/thumbs/magn-field.png'


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
    'htarg': {'pts_per_dec': -1},
    'mrec': True,
}

# Compute
epm_hx = empymod.bipole(
    rec=[frx, fry, -rz, 0, 0], verb=3, **inp).reshape(np.shape(rx))
epm_hy = empymod.bipole(
    rec=[frx, fry, -rz, 90, 0], **inp).reshape(np.shape(rx))
epm_hz = empymod.bipole(
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
# Compute magnetic field :math:`H` from the electric field
# --------------------------------------------------------
hfield = emg3d.get_magnetic_field(model, efield)

###############################################################################
# Plot
# ````
e3d_hx = hfield.get_receiver((rx, ry, rz, 0, 0))
plot.plot_sections(
        epm_hx, e3d_hx, x, r'Diffusive Fullspace $H_x$',
        vmin=-8, vmax=-4, mode='abs'
)

###############################################################################
e3d_hy = hfield.get_receiver((rx, ry, rz, 90, 0))
plot.plot_sections(
        epm_hy, e3d_hy, x, r'Diffusive Fullspace $H_y$',
        vmin=-8, vmax=-4, mode='abs'
)

###############################################################################
e3d_hz = hfield.get_receiver((rx, ry, rz, 0, 90))
plot.plot_sections(
        epm_hz, e3d_hz, x, r'Diffusive Fullspace $H_z$',
        vmin=-8, vmax=-4, mode='abs'
)

###############################################################################
plot.plot_line(x, x, e3d_hx.real, epm_hx.real, grid, 'H_x')

###############################################################################
emg3d.Report()
