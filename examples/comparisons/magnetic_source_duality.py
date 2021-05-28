r"""
8. Magnetic source using duality
================================

Computing the :math:`E` and :math:`H` fields from a magnetic source using the
duality principle.

We know that we can get the magnetic fields from the electric fields using
Faraday's law, see :ref:`sphx_glr_gallery_comparisons_magnetic_field.py`.

However, what about computing the fields generated by a magnetic source?
There are two ways we can achieve that:

- **using the duality principle**, which is what we do in this example, or
- creating an electric loop source, see
  :ref:`sphx_glr_gallery_comparisons_magnetic_source_el_loop.py`.

``emg3d`` solves the following equation,

.. math::
    :label: eq-maxwell

     \eta \mathbf{\hat{E}} - \nabla \times \zeta^{-1} \nabla \times
     \mathbf{\hat{E}} = -\mathbf{\hat{J}}^e_s ,

where :math:`\eta = \sigma - \mathrm{i}\omega\varepsilon`, :math:`\zeta =
\mathrm{i}\omega\mu`, :math:`\sigma` is conductivity (S/m), :math:`\omega=2\pi
f` is the angular frequency (Hz), :math:`\mu=\mu_0\mu_\mathrm{r}` is magnetic
permeability (H/m), :math:`\varepsilon=\varepsilon_0\varepsilon_\mathrm{r}` is
electric permittivity (F/m), :math:`\mathbf{\hat{E}}` the electric field in the
frequency domain (V/m), and  :math:`\mathbf{\hat{J}}^e_s` source current.

This is the electric field due to an electric source. One can obtain the
magnetic field due to a magnetic field by substituting

- :math:`\eta \leftrightarrow -\zeta` ,
- :math:`\mathbf{\hat{E}} \leftrightarrow -\mathbf{\hat{H}}` ,
- :math:`\mathbf{\hat{J}}^e_s \leftrightarrow \mathbf{\hat{J}}^m_s` ,

which is called the **duality principle**.

Carrying out the substitution yields

.. math::
    :label: dualdip

    \zeta \mathbf{\hat{H}} - \nabla \times \eta^{-1} \nabla \times
    \mathbf{\hat{H}} = -\mathbf{\hat{J}}^m_s ,

which is for a magnetic dipole. Changing it for a loop source adds a term
:math:`\mathrm{i}\omega\mu` to the source term, resulting in

.. math::
    :label: dualloop

    \zeta \mathbf{\hat{H}} - \nabla \times \eta^{-1} \nabla \times
    \mathbf{\hat{H}} = -\mathrm{i}\omega\mu\mathbf{\hat{J}}^m_s ;

see `Dipoles and Loops
<https://empymod.emsig.xyz/en/stable/examples/educational/dipoles_and_loops.html>`_
for more information.

``emg3d`` is not ideal for the duality principle. Magnetic permeability is
implemented isotropically, and discontinuities in magnetic permeabilities can
lead to first-order errors in contrary to second-order errors for
discontinuities in conductivity. However, we can still abuse the code and use
it with the duality principle, at least for isotropic media.

The actual implemented equation in ``emg3d`` is a slightly modified version of
Equation :eq:`eq-maxwell`, using the diffusive approximation
:math:`\varepsilon=0`,

.. math::
    :label: dualdiff

    \mathrm{i}\omega \mu_0 \sigma \mathbf{\hat{E}} - \nabla \times
    \mu_r^{-1} \nabla \times \mathbf{\hat{E}} =
    -\mathrm{i}\omega\mu_0\mathbf{\hat{J}}_s .

We therefore only need to interchange :math:`\sigma` with
:math:`\mu_\mathrm{r}^{-1}` or :math:`\rho` with :math:`\mu_\mathrm{r}` to get
from :eq:`dualdiff` to :eq:`dualloop`.

This is what we do in this example, for an arbitrarily rotated loop in a
homogeneous, isotropic fullspace. We compare the result to the semi-analytical
solution of ``empymod``. (The code ``empymod`` is an open-source code which can
model CSEM responses for a layered medium including VTI electrical anisotropy,
see `emsig.xyz <https://emsig.xyz>`_.)

"""
import emg3d
import empymod
import numpy as np
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_path = '_static/thumbs/duality.png'


###############################################################################
# Full-space model for a rotated magnetic loop
# --------------------------------------------
#
# In order to shorten the build-time of the gallery we use a coarse model.
# Set ``coarse_model = False`` to obtain a result of higher accuracy.
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
azimuth = 25
elevation = 10

# Source coordinates, frequency, and strength
source = emg3d.TxElectricDipole(
    coordinates=[0, 0, -300, 10, 70],  # [x, y, z, azimuth, elevation]
    strength=np.pi,  # A
)
frequency = 0.77  # Hz

# Model parameters
h_res = 2.              # Horizontal resistivity


###############################################################################
# empymod
# ```````
# Note: The coordinate system of empymod is positive z down, for emg3d it is
# positive z up. We have to switch therefore src_z, rec_z, and elevation.

# Collect common input for empymod.
inp = {
    'src': np.r_[source.coordinates[:2], -source.coordinates[2],
                 source.coordinates[3], -source.coordinates[4]],
    'depth': [],
    'res': h_res,
    'strength': source.strength,
    'freqtime': frequency,
    'htarg': {'pts_per_dec': -1},
}

# Compute e-field
epm_e = -empymod.loop(
    rec=[frx, fry, -rz, azimuth, -elevation], mrec=False, verb=3, **inp
).reshape(np.shape(rx))

# Compute h-field
epm_h = empymod.loop(
    rec=[frx, fry, -rz, azimuth, -elevation], **inp
).reshape(np.shape(rx))


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
# Abuse the parameters to take advantage of the duality principle
# ---------------------------------------------------------------
#
# See text at the top. We set here :math:`\rho=1` and :math:`\mu_\mathrm{r} =
# 1/\rho` to get:
#
# .. math::
#     :label: iweta2iwu
#
#     \mathrm{i}\omega\mu_0(1-\mathrm{i}\omega\varepsilon) =
#     \mathrm{i}\omega\mu_0+\omega^2\mu_0\varepsilon \approx
#     \mathrm{i}\omega\mu
#
# (in the diffusive regime), and
#
# .. math::
#     :label: mu2sigma
#
#     \mu_\mathrm{r} = 1/\rho = \sigma \, .

# Define the model        => Set property_x = 1 and mu_r = 1./h_res
model = emg3d.Model(
    grid, property_x=1., mu_r=1./h_res, mapping='Resistivity')

# Compute the electric field
hfield = emg3d.solve_source(model, source, frequency, verb=4, plain=True)


###############################################################################
# Plot function
# `````````````

def plot(epm, e3d, title, vmin, vmax):

    # Start figure.
    a_kwargs = {'cmap': "viridis", 'vmin': vmin, 'vmax': vmax,
                'shading': 'nearest'}

    e_kwargs = {'cmap': plt.cm.get_cmap("RdBu_r", 8),
                'vmin': -2, 'vmax': 2, 'shading': 'nearest'}

    fig, axs = plt.subplots(2, 3, figsize=(10, 5.5), sharex=True, sharey=True,
                            subplot_kw={'box_aspect': 1})

    ((ax1, ax2, ax3), (ax4, ax5, ax6)) = axs
    x3 = x/1000  # km

    # Plot Re(data)
    ax1.set_title(r"(a) |Re(empymod)|")
    cf0 = ax1.pcolormesh(x3, x3, np.log10(epm.real.amp()), **a_kwargs)

    ax2.set_title(r"(b) |Re(emg3d)|")
    ax2.pcolormesh(x3, x3, np.log10(e3d.real.amp()), **a_kwargs)

    ax3.set_title(r"(c) Error real part")
    rel_error = 100*np.abs((epm.real - e3d.real) / epm.real)
    cf2 = ax3.pcolormesh(x3, x3, np.log10(rel_error), **e_kwargs)

    # Plot Im(data)
    ax4.set_title(r"(d) |Im(empymod)|")
    ax4.pcolormesh(x3, x3, np.log10(epm.imag.amp()), **a_kwargs)

    ax5.set_title(r"(e) |Im(emg3d)|")
    ax5.pcolormesh(x3, x3, np.log10(e3d.imag.amp()), **a_kwargs)

    ax6.set_title(r"(f) Error imaginary part")
    rel_error = 100*np.abs((epm.imag - e3d.imag) / epm.imag)
    ax6.pcolormesh(x3, x3, np.log10(rel_error), **e_kwargs)

    # Colorbars
    unit = "(V/m)" if "E" in title else "(A/m)"
    fig.colorbar(cf0, ax=axs[0, :], label=r"$\log_{10}$ Amplitude "+unit)
    cbar = fig.colorbar(cf2, ax=axs[1, :], label=r"Relative Error")
    cbar.set_ticks([-2, -1, 0, 1, 2])
    cbar.ax.set_yticklabels([r"$0.01\,\%$", r"$0.1\,\%$", r"$1\,\%$",
                             r"$10\,\%$", r"$100\,\%$"])

    ax1.set_xlim(min(x3), max(x3))
    ax1.set_ylim(min(x3), max(x3))

    # Axis label
    fig.text(0.4, 0.05, "Inline Offset (km)", fontsize=14)
    fig.text(0.05, 0.3, "Crossline Offset (km)", rotation=90, fontsize=14)
    fig.suptitle(title, y=1, fontsize=20)

    print(f"- Source: {source}")
    print(f"- Frequency: {frequency} Hz")
    rtype = "Electric" if "E" in title else "Magnetic"
    print(f"- {rtype} receivers: z={rz} m; θ={azimuth}°, φ={elevation}°")

    fig.show()


###############################################################################
# Compare the magnetic field generated from the magnetic source
# -------------------------------------------------------------
e3d_h = hfield.get_receiver((rx, ry, rz, azimuth, elevation))
plot(epm_h, e3d_h, r'Diffusive Fullspace $H$', vmin=-15, vmax=-8)


###############################################################################
# Compare the electric field generated from the magnetic source
# -------------------------------------------------------------
#
# ``get_magnetic_field`` gets the :math:`H`-field from the :math:`E`-field with
# Faraday's law,
#
# .. math::
#     :label: faraday2
#
#     \nabla \times \mathbf{E} = \rm{i}\omega \mathbf{B} =
#     \rm{i}\omega\mu\mathbf{H}\, .
#
# Using the substitutions introduced in the beginning, and using the same
# function but to get the :math:`E`-field from the :math:`H`-field, we have to
# multiply the result by
#
# .. math::
#     :label: iwu
#
#     \rm{i}\omega\mu\, .
#
# Compute electric field :math:`E` from the magnetic field
# ````````````````````````````````````````````````````````

efield = emg3d.get_magnetic_field(model, hfield)
efield.field *= efield.smu0

e3d_e = efield.get_receiver((rx, ry, rz, azimuth, elevation))
plot(epm_e, e3d_e, r'Diffusive Fullspace $E$', vmin=-17, vmax=-10)


###############################################################################
emg3d.Report()
