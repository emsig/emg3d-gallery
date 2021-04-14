r"""
2. Ensure computation domain is big enough
==========================================

Ensure the boundary in :math:`\pm x`, :math:`\pm y`, and :math:`+ z` is big
enough for :math:`\rho_\text{air}`.

The air is very resistive, and EM waves propagate at the speed of light as a
wave, not diffusive any longer. The whole concept of skin depth does therefore
not apply to the air layer. The only attenuation is caused by geometrical
spreading. In order to not have any effects from the boundary one has to choose
the air layer appropriately.

The important bit is that the resistivity of air has to be taken into account
also for the horizontal directions, not only for positive :math:`z` (upwards
into the sky). This is an example to test boundaries on a simple marine model
(air, water, subsurface) and compare them to the 1D result.
"""
import emg3d
import empymod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use('ggplot')
# sphinx_gallery_thumbnail_number = 2

return  # will break but create the title # TODO Not Updated Yet


###############################################################################
# Model, Survey, and Analytical Solution
# --------------------------------------

water_depth = 500                    # 500 m water depth
off = np.linspace(2000, 7000, 501)   # Offsets
src = [np.array([0]), np.array([0]),
       np.array([-water_depth+50])]  # Source at origin, 50 m above seafloor
rec = [off, off*0, -water_depth]     # Receivers on the seafloor
depth = [-water_depth, 0]            # Simple model
res = [1, 0.3, 1e8]                  # Simple model
freq = 0.1                           # Frequency

# Compute analytical solution
epm = empymod.dipole(src, rec, depth, res, freq)

###############################################################################
# 3D Modelling
# ------------

# Parameter we keep the same for both grids
x_inp = {'fixed': src[0], 'domain': [src[0][0]-500, off[-1]+500]}
y_inp = {'fixed': src[1], 'domain': [src[1][0], src[1][0]]}
z_inp = {'fixed': [0, -water_depth-100, 100], 'domain': [-600, 100]}
inp = {'freq': freq, 'alpha': [1, 1.25, 0.01], 'min_width': 100}

# Solver parameters
solver_inp = {
        'verb': 4,
        'sslsolver': True,
        'semicoarsening': True,
        'linerelaxation': True
}


###############################################################################
# 1st grid, only considering air resistivity for +z
# -------------------------------------------------
#
# Here we are in the water, so the signal is attenuated before it enters the
# air. So we don't use the resistivity of air to compute the required
# boundary, but 100 Ohm.m instead. (100 is the result of a quick parameter test
# with :math:`\rho=1e4, 1e3, 1e2, 1e1`, and the result was that after 100 there
# is not much improvement any longer.)
#
# Also note that the function ``emg3d.meshes.get_hx_h0`` internally uses six
# times the skin depth for the boundary. For :math:`\rho` = 100 Ohm.m and
# :math:`f` = 0.1 Hz, the skin depth :math:`\delta` is roughly 16 km, which
# therefore results in a boundary of roughly 96 km.
#
# See the documentation of ``emg3d.meshes.get_hx_h0`` for more information on
# how the grid is created.

# Get cell widths and origin in each direction
xx_1, x0_1 = emg3d.meshes.get_hx_h0(res=[res[1], res[0]], **x_inp, **inp)
yy_1, y0_1 = emg3d.meshes.get_hx_h0(res=[res[1], res[0]], **y_inp, **inp)
zz_1, z0_1 = emg3d.meshes.get_hx_h0(res=[res[1], res[0], 100], **z_inp, **inp)

# Create grid and correpsoding model
grid_1 = emg3d.TensorMesh([xx_1, yy_1, zz_1], x0=np.array([x0_1, y0_1, z0_1]))
res_1 = res[0]*np.ones(grid_1.n_cells)
res_1[grid_1.cell_centers[:, 2] > -water_depth] = res[1]
res_1[grid_1.cell_centers[:, 2] > 0] = res[2]
model_1 = emg3d.Model(grid_1, property_x=res_1, mapping='Resistivity')

# QC
grid_1.plot_3d_slicer(
        np.log10(model_1.property_x), zlim=(-2000, 100), clim=[-1, 2])

# Define source and solve the system
sfield_1 = emg3d.get_source_field(
        grid_1, [src[0], src[1], src[2], 0, 0], freq)
efield_1 = emg3d.solve(grid_1, model_1, sfield_1, **solver_inp)


###############################################################################
# 2nd grid, considering air resistivity for +/- x, +/- y, and +z
# --------------------------------------------------------------
#
# See comments below the heading of the 1st grid regarding boundary.

# Get cell widths and origin in each direction
xx_2, x0_2 = emg3d.meshes.get_hx_h0(res=[res[1], 100], **x_inp, **inp)
yy_2, y0_2 = emg3d.meshes.get_hx_h0(res=[res[1], 100], **y_inp, **inp)
zz_2, z0_2 = emg3d.meshes.get_hx_h0(res=[res[1], res[0], 100], **z_inp, **inp)

# Create grid and correpsoding model
grid_2 = emg3d.TensorMesh([xx_2, yy_2, zz_2], x0=np.array([x0_2, y0_2, z0_2]))
res_2 = res[0]*np.ones(grid_2.n_cells)
res_2[grid_2.cell_centers[:, 2] > -water_depth] = res[1]
res_2[grid_2.cell_centers[:, 2] > 0] = res[2]
model_2 = emg3d.Model(grid_2, property_x=res_2, mapping='Resistivity')

# QC
# grid_2.plot_3d_slicer(
#         np.log10(model_2.property_x), zlim=(-2000, 100), clim=[-1, 2])

# Define source and solve the system
sfield_2 = emg3d.get_source_field(
        grid_2, [src[0], src[1], src[2], 0, 0], freq)
efield_2 = emg3d.solve(grid_2, model_2, sfield_2, **solver_inp)


###############################################################################
# Plot receiver responses
# -----------------------

# Interpolate fields at receiver positions
emg_1 = emg3d.get_receiver(
        grid_1, efield_1.fx, (rec[0], rec[1], rec[2]))
emg_2 = emg3d.get_receiver(
        grid_2, efield_2.fx, (rec[0], rec[1], rec[2]))


###############################################################################

plt.figure(figsize=(10, 7))

# Real, log-lin
ax1 = plt.subplot(321)
plt.title('(a) lin-lin Real')
plt.plot(off/1e3, epm.real, 'k', lw=2, label='analytical')
plt.plot(off/1e3, emg_1.real, 'C0--', label='grid 1')
plt.plot(off/1e3, emg_2.real, 'C1:', label='grid 2')
plt.ylabel('$E_x$ (V/m)')
plt.legend()

# Real, log-symlog
ax3 = plt.subplot(323, sharex=ax1)
plt.title('(c) lin-symlog Real')
plt.plot(off/1e3, epm.real, 'k')
plt.plot(off/1e3, emg_1.real, 'C0--')
plt.plot(off/1e3, emg_2.real, 'C1:')
plt.ylabel('$E_x$ (V/m)')
plt.yscale('symlog', linthresh=1e-15)

# Real, error
ax5 = plt.subplot(325, sharex=ax3)
plt.title('(e) clipped 0.01-10')

# Compute the error
err_real_1 = np.clip(100*abs((epm.real-emg_1.real)/epm.real), 0.01, 10)
err_real_2 = np.clip(100*abs((epm.real-emg_2.real)/epm.real), 0.01, 10)

plt.ylabel('Rel. error %')
plt.plot(off/1e3, err_real_1, 'C0--')
plt.plot(off/1e3, err_real_2, 'C1:')
plt.axhline(1, color='.4')

plt.yscale('log')
plt.ylim([0.008, 12])
plt.xlabel('Offset (km)')

# Imaginary, log-lin
ax2 = plt.subplot(322)
plt.title('(b) lin-lin Imag')
plt.plot(off/1e3, epm.imag, 'k')
plt.plot(off/1e3, emg_1.imag, 'C0--')
plt.plot(off/1e3, emg_2.imag, 'C1:')

# Imaginary, log-symlog
ax4 = plt.subplot(324, sharex=ax2)
plt.title('(d) lin-symlog Imag')
plt.plot(off/1e3, epm.imag, 'k')
plt.plot(off/1e3, emg_1.imag, 'C0--')
plt.plot(off/1e3, emg_2.imag, 'C1:')

plt.yscale('symlog', linthresh=1e-15)

# Imaginary, error
ax6 = plt.subplot(326, sharex=ax2)
plt.title('(f) clipped 0.01-10')

# Compute error
err_imag_1 = np.clip(100*abs((epm.imag-emg_1.imag)/epm.imag), 0.01, 10)
err_imag_2 = np.clip(100*abs((epm.imag-emg_2.imag)/epm.imag), 0.01, 10)

plt.plot(off/1e3, err_imag_1, 'C0--')
plt.plot(off/1e3, err_imag_2, 'C1:')
plt.axhline(1, color='.4')

plt.yscale('log')
plt.ylim([0.008, 12])
plt.xlabel('Offset (km)')

plt.tight_layout()
plt.show()


###############################################################################
# Plot entire fields to analyze and compare
# -----------------------------------------
#
# 1st grid
# ````````
#
# Upper plot shows the entire grid. One can see that the airwave attenuates to
# amplitudes in the order of 1e-17 at the boundary, absolutely good enough.
# However, the amplitudes in the horizontal directions are very high even at
# the boundaries :math:`\pm x` and :math:`\pm y`.

grid_1.plot_3d_slicer(
    efield_1.fx.ravel('F'), view='abs', v_type='Ex',
    xslice=src[0], yslice=src[1], zslice=rec[2],
    pcolor_opts={'norm': LogNorm(vmin=1e-17, vmax=1e-9)})
grid_1.plot_3d_slicer(
    efield_1.fx.ravel('F'), view='abs', v_type='Ex',
    zlim=[-5000, 1000],
    xslice=src[0], yslice=src[1], zslice=rec[2],
    pcolor_opts={'norm': LogNorm(vmin=1e-17, vmax=1e-9)})


###############################################################################
# 2nd grid
# ````````
#
# Again, upper plot shows the entire grid. One can see that the field
# attenuated sufficiently in all directions. Lower plot shows the same cut-out
# as the lower plot for the first grid, our zone of interest.

grid_2.plot_3d_slicer(
    efield_2.fx.ravel('F'), view='abs', v_type='Ex',
    xslice=src[0], yslice=src[1], zslice=rec[2],
    pcolor_opts={'norm': LogNorm(vmin=1e-17, vmax=1e-9)})
grid_2.plot_3d_slicer(
    efield_2.fx.ravel('F'), view='abs', v_type='Ex',
    xlim=[grid_1.nodes_x[0], grid_1.nodes_x[-1]],  # Same square as grid_1
    ylim=[grid_1.nodes_y[0], grid_1.nodes_y[-1]],  # Same square as grid_1
    zlim=[-5000, 1000],
    xslice=src[0], yslice=src[1], zslice=rec[2],
    pcolor_opts={'norm': LogNorm(vmin=1e-17, vmax=1e-9)})


###############################################################################

emg3d.Report()
