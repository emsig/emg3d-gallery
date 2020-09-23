r"""
4. Adjoint-state vs. FD gradient
================================

The **gradient of the misfit function**, as implemented in `emg3d`, uses the
adjoint-state method following [PlMu08]_ (see :func:`emg3d.optimize.gradient`).
The method has the advantage that it is very fast. However, it can be tricky to
implement and it is always good to verify the implementation against another
method.

We compare in this example the adjoint-state gradient to a simple forward
finite-difference gradient. (See :ref:`sphx_glr_gallery_tutorials_gradient.py`
for more details regarding the adjoint-state gradient.)

"""
import emg3d
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm, SymLogNorm
plt.style.use('ggplot')
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# 1. Create a survey and a simple model
# -------------------------------------
#
# Create a simple block model and a survey for the comparison.
#
# Survey
# ''''''
# The survey consists of one source, one receiver, both electric, x-directed
# point-dipoles, where the receiver is on the seafloor and the source 50 m
# above. Offset is 3.2 km, and acquisition frequency is 1 Hz.

survey = emg3d.surveys.Survey(
    name='Gradient Test-Survey',
    sources=(-1600, 0, -1950, 0, 0),
    receivers=(1600, 0, -2000, 0, 0),
    frequencies=1.0,
)

###############################################################################
# Model
# '''''
#
# As `emg3d` internally computes with conductivities we use conductivities to
# compare the gradient in its purest implementation. Note that if we define our
# model in terms of resistivities or :math:`\log_{\{e;10\}}(\{\sigma;\rho\})`,
# the gradient would look differently.

# Create a simple block model.
hx = np.array([1000, 1500, 1000, 1500, 1000])
hy = np.array([1000, 1500, 1000, 1500, 1000])
hz = np.array([1800., 200., 200., 200., 300., 300., 2000.,  500.])
model_grid = emg3d.TensorMesh([hx, hy, hz], x0=np.array([-3000, -3000, -5000]))

# Initiate model with conductivities of 1 S/m.
model = emg3d.Model(model_grid, np.ones(model_grid.nC), mapping='Conductivity')
model.property_x[:, :, -1] = 1e-8  # Add air layer.
model.property_x[:, :, -2] = 3.33  # Add seawater layer.
model_bg = model.copy()  # Make a copy for the background model.

# Add three blocks.
model.property_x[1, 1:3, 1:3] = 0.02
model.property_x[3, 2:4, 2:4] = 0.01
model.property_x[2, 1:4, 4] = 0.005

###############################################################################
# QC
# ''

model_grid.plot_3d_slicer(model.property_x.ravel('F'), zslice=-2900,
                          pcolor_opts={'norm': LogNorm(vmin=0.002, vmax=3.5)})

plt.suptitle('Conductivity (S/m)')
axs = plt.gcf().get_children()
axs[1].plot(survey.rec_coords[0], survey.rec_coords[1], 'bv')
axs[2].plot(survey.rec_coords[0], survey.rec_coords[2], 'bv')
axs[3].plot(survey.rec_coords[2], survey.rec_coords[1], 'bv')
axs[1].plot(survey.src_coords[0], survey.src_coords[1], 'r*')
axs[2].plot(survey.src_coords[0], survey.src_coords[2], 'r*')
axs[3].plot(survey.src_coords[2], survey.src_coords[1], 'r*')
plt.show()


###############################################################################
# Generate synthetic data
# -----------------------
#
# Generate a computational grid to create synthetic data (note that in the
# future we can use the automatic gridding functionality, which should land in
# v0.13.0).

# Get cell widths and origin in each direction.
gridinput = {'min_width': 100, 'freq': survey.frequencies[0], 'verb': 0}
xy, xy0 = emg3d.meshes.get_hx_h0(  # Same in x and y.
    res=[0.3, 1], fixed=0, domain=[-2000, 2000], **gridinput)
zz, z0 = emg3d.meshes.get_hx_h0(
    res=[0.3, 1, 0.3], domain=[-3200, -2000], **gridinput, fixed=[-2000, 0])


data_grid = emg3d.TensorMesh([xy, xy, zz], x0=np.array([xy0, xy0, z0]))

# Define a simulation for the data.
simulation_data = emg3d.simulations.Simulation(
    name='Data for Gradient Test',
    survey=survey,
    grid=data_grid,
    model=model.interpolate2grid(model_grid, data_grid),
    gridding='same',  # Same grid as for input model.
    max_workers=4,
)

# Simulate the data and store them as observed.
simulation_data.compute(observed=True)

# Let's print the survey to check that the observed data are now set.
survey

###############################################################################
# Adjoint-state gradient
# ----------------------
#
# For the actual comparison we use a very coarse mesh. The reason is that we
# have to compute an extra forward model for each cell for the forward
# finite-difference approximation, so we try to keep that number low (even
# though we only do a cross-section, not the entire cube).

# Get cell widths and origin in each direction
gridinput = {'min_width': 200, 'freq': survey.frequencies[0], 'verb': 0}
xy, xy0 = emg3d.meshes.get_hx_h0(  # Same in x and y.
    res=[0.3, 1], fixed=0, domain=[-2000, 2000], **gridinput)
zz, z0 = emg3d.meshes.get_hx_h0(
    res=[0.3, 1, 0.3], domain=[-3200, -2000], **gridinput, fixed=[-2000, 0])

comp_grid = emg3d.TensorMesh([xy, xy, zz], x0=np.array([xy0, xy0, z0]))

comp_grid

###############################################################################
# Our computational grid has only 16,384 cells, which should be fast enough to
# compute the FD gradient of a cross-section in a few minutes. A
# cross-section along the survey-line has :math:`32 \times 11 = 352` cells, so
# we need to compute an extra 352 forward models. (There are 16 cells in z, but
# only 11 below the seafloor.)

# Interpolate the background model onto the computational grid.
comp_model = model_bg.interpolate2grid(model_grid, comp_grid)

# Switch-off all data weighting in order to compare the pure gradient.
data_weight_opts = {
    'gamma_d': 0,
    'beta_d': 0,
    'beta_f': 0,
    'min_off': 0,
    'noise_floor': 0,
}

# AS gradient simulation.
simulation_as = emg3d.simulations.Simulation(
    name='AS Gradient Test',
    survey=survey,
    grid=comp_grid,
    model=comp_model,
    gridding='same',  # Same grid as for input model.
    max_workers=4,    # For parallel workers, adjust if you have more.
    data_weight_opts=data_weight_opts,
)

simulation_as

###############################################################################

# Get the misfit and the gradient of the misfit.
data_misfit = simulation_as.misfit
as_grad = simulation_as.gradient

# Set water and air gradient to NaN for the plots.
as_grad[:, :, comp_grid.vectorCCz > -2000] = np.nan


###############################################################################
# Finite-Difference gradient
# --------------------------
#
# To test if the adjoint-state gradient indeed returns the gradient we can
# compare it to a one-sided finite-difference approximated gradient as given by
#
# .. math::
#         \left(\nabla_p J \left(\textbf{p}\right)\right)_{FD} =
#         \frac{J(\textbf{p}+\epsilon) - J(\textbf{p})}{\epsilon} \ .
#
# Define a fct to compute FD gradient for one cell
# ''''''''''''''''''''''''''''''''''''''''''''''''

# Define epsilon (some small conductivity value, S/m).
epsilon = 0.0001

# Define the cross-section.
iy = comp_grid.nCy//2


def comp_fd_grad(ixiz):
    """Compute forward-FD gradient for one cell."""

    # Copy the computational model.
    fd_model = comp_model.copy()

    # Add conductivity-epsilon to this (ix, iy, iz) cell.
    fd_model.property_x[ixiz[0], iy, ixiz[1]] += epsilon

    # Create a new simulation with this model
    simulation_fd = emg3d.simulations.Simulation(
        name='FD Gradient Test',
        survey=survey, grid=comp_grid, model=fd_model, gridding='same',
        max_workers=1, data_weight_opts=data_weight_opts,
        solver_opts={'verb': 1})

    # Switch-of progress bar in this case
    simulation_fd._tqdm_opts['disable'] = True

    # Get misfit
    fd_data_misfit = simulation_fd.misfit

    # Return gradient
    return float((fd_data_misfit - data_misfit)/epsilon)


###############################################################################
# Loop over all required cells
# ''''''''''''''''''''''''''''

# Initiate FD gradient.
fd_grad = np.zeros_like(as_grad)

# Get all ix-iz combinations (without air/water).
ixiz = list(itertools.product(
    range(comp_grid.nCx),
    range(len(comp_grid.vectorCCz[comp_grid.vectorCCz < -2000])))
)

# Wrap it asynchronously
out = emg3d.simulations.process_map(
        comp_fd_grad,
        ixiz,
        max_workers=4,  # Adjust max worker here!
)

# Collect result
for i, (ix, iz) in enumerate(ixiz):
    fd_grad[ix, iy, iz] = out[i]


###############################################################################
# Compare the two gradients
# '''''''''''''''''''''''''

# Compute NRMSD between AS and FD (%).
nrmsd = 200*abs(as_grad-fd_grad)/(abs(as_grad)+abs(fd_grad))
nrmsd[fd_grad == 0] = np.nan

# Compute sign.
diff_sign = np.sign(as_grad/fd_grad)


def plot_diff(ax, diff):
    """Helper routine to show cells of big NRMSD or different sign."""

    for ix in range(comp_grid.hx.size):
        for iz in range(comp_grid.hz.size):

            if diff_sign[ix, iy, iz] < 0:
                ax.add_patch(
                        Rectangle(
                            (comp_grid.vectorNx[ix], comp_grid.vectorNz[iz]),
                            comp_grid.hx[ix], comp_grid.hz[iz], fill=False,
                            color='k', lw=1))

            if nrmsd[ix, iy, iz] >= diff:
                ax.add_patch(
                        Rectangle(
                            (comp_grid.vectorNx[ix], comp_grid.vectorNz[iz]),
                            comp_grid.hx[ix], comp_grid.hz[iz], fill=False,
                            color='m', linestyle='--', lw=0.5))


def set_axis(axs, i):
    """Helper routine to adjust subplots."""

    # Show source and receiver.
    axs[i].plot(survey.rec_coords[0], survey.rec_coords[2], 'bv')
    axs[i].plot(survey.src_coords[0], survey.src_coords[2], 'r*')

    # x-label.
    axs[i].set_xlabel('Easting')

    # y-label depending on column.
    if i == 0:
        axs[i].set_ylabel('Depth')
    else:
        axs[i].set_ylabel('')
        axs[i].axes.yaxis.set_ticklabels([])

    # Set limits.
    axs[i].set_xlim(-3000, 3000)
    axs[i].set_ylim(-4000, -1900)


# Plotting options.
vmin, vmax = 1e-31, 1e-26
pcolor_opts = {'cmap': 'RdBu_r',
               'norm': SymLogNorm(linthresh=vmin, base=10,
                       vmin=-vmax, vmax=vmax)}

fig, axs = plt.subplots(figsize=(9, 6), nrows=1, ncols=2)

# Adjoint-State Gradient
f0 = comp_grid.plotSlice(as_grad, normal='Y', ind=iy, ax=axs[0],
                         clim=[-vmax, vmax], pcolor_opts=pcolor_opts)
axs[0].set_title("Adjoint-State Gradient")
set_axis(axs, 0)
plot_diff(axs[0], 10)

# Finite-Difference Gradient
f1 = comp_grid.plotSlice(fd_grad, normal='Y', ind=iy, ax=axs[1],
                         clim=[-vmax, vmax], pcolor_opts=pcolor_opts)
axs[1].set_title("Finite-Difference Gradient")
set_axis(axs, 1)
plot_diff(axs[1], 10)

plt.tight_layout()
fig.colorbar(f0[0], ax=axs, orientation='horizontal', fraction=0.05)
plt.show()

###############################################################################
#
# Visually the two gradients are almost identical. This is amazing, given that
# the adjoint-state gradient requires one (1) extra forward computation for the
# entire cube, whereas the finite-difference gradient requires one extra
# forward computation for each cell, for this cross-section 352 (!).
#
# There are differences, and they are highlighted:
#
#   - Cells surrounded by a dashed, magenta line: NRMSD is bigger than 10 %.
#   - Cells surrounded by a black line: The two gradients have different signs.
#
# These differences only happen in three distinct regions:
#
#   1. Close to the source and the receiver;
#   2. where the gradient rapidly changes (changes sign);
#   3. where the amplitude of the gradient is very small.
#
# The first point is mainly because of the coarse meshing we used, using a more
# appropriate, finer mesh would decrease this difference. The second point
# comes mainly from the fact that both, AS and FD, are approximations, and
# where the gradient changes rapidly this can lead to differences. Again, using
# a finer mesh would most likely eliminate these errors. The last point is more
# a difficulty in computing the error between two values which both go to zero,
# and not really related to the gradients.

emg3d.Report()
