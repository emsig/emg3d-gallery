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
`empymod.github.io <https://empymod.github.io>`_.)

This is an adapted version of
:ref:`sphx_glr_gallery_comparisons_1D_VTI_empymod.py`. Consult that example to
see the result for the electric field.

"""
import emg3d
import empymod
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as sint
from matplotlib.colors import SymLogNorm
plt.style.use('ggplot')
# sphinx_gallery_thumbnail_path = '_static/thumbs/magn-perm.png'


###############################################################################
def plot_data_rel(ax, name, data, x, vmin=-15., vmax=-7., mode="log"):
    """Plot function."""

    ax.set_title(name)
    ax.set_xlim(min(x)/1000, max(x)/1000)
    ax.set_ylim(min(x)/1000, max(x)/1000)
    ax.axis("equal")

    if isinstance(mode, str):
        if mode == "abs":
            cf = ax.pcolormesh(
                    x/1000, x/1000, np.log10(np.abs(data)), linewidth=0,
                    rasterized=True, cmap="viridis", vmin=vmin, vmax=vmax,
                    shading='nearest')
        else:
            cf = ax.pcolormesh(
                    x/1000, x/1000, data, linewidth=0, rasterized=True,
                    cmap="PuOr_r",
                    norm=SymLogNorm(linthresh=10**vmin,
                                    vmin=-10**vmax, vmax=10**vmax),
                    shading='nearest')
    else:
        cf = ax.pcolormesh(
                x/1000, x/1000, np.log10(data), vmin=vmin, vmax=vmax,
                linewidth=0, rasterized=True,
                cmap=plt.cm.get_cmap("RdBu_r", 8), shading='nearest')

    return cf


###############################################################################
def plot_result_rel(depm, de3d, x, title, vmin=-15., vmax=-7., mode="log"):
    fig, axs = plt.subplots(figsize=(18, 10), nrows=2, ncols=3)

    if mode == "log":
        case = ""
    else:
        case = "|"

    # Plot Re(data)
    cf0 = plot_data_rel(axs[0, 0], r"(a) "+case+"Re(empymod)"+case,
                        depm.real, x, vmin, vmax, mode)
    plot_data_rel(axs[0, 1], r"(b) "+case+"Re(emg3d)"+case,
                  de3d.real, x, vmin, vmax, mode)
    cf2 = plot_data_rel(axs[0, 2], r"(c) Error real part",
                        np.abs((depm.real-de3d.real)/depm.real)*100, x,
                        vmin=-2, vmax=2, mode=True)

    # Plot Im(data)
    plot_data_rel(axs[1, 0], r"(d) "+case+"Im(empymod)"+case,
                  depm.imag, x, vmin, vmax, mode)
    plot_data_rel(axs[1, 1], r"(e) "+case+"Im(emg3d)"+case,
                  de3d.imag, x, vmin, vmax, mode)
    plot_data_rel(axs[1, 2], r"(f) Error imaginary part",
                  np.abs((depm.imag-de3d.imag)/depm.imag)*100,
                  x, vmin=-2, vmax=2, mode=True)

    # Colorbars
    fig.colorbar(cf0, ax=axs[0, :], label=r"$\log_{10}$ Amplitude (V/m)")
    cbar = fig.colorbar(cf2, ax=axs[1, :], label=r"Relative Error")
    cbar.set_ticks([-2, -1, 0, 1, 2])
    cbar.ax.set_yticklabels([r"$0.01\,\%$", r"$0.1\,\%$", r"$1\,\%$",
                             r"$10\,\%$", r"$100\,\%$"])

    # Axis label
    fig.text(0.4, 0.05, "Inline Offset (km)", fontsize=14)
    fig.text(0.08, 0.6, "Crossline Offset (km)", rotation=90, fontsize=14)

    # Title
    fig.suptitle(title, y=1, fontsize=20)
    plt.show()


###############################################################################
def plot_lineplot_ex(x, y, data, epm_fs, grid):
    xi = x.size//2
    yi = y.size//2

    fn = sint.interp1d(x, data[:, xi], bounds_error=False)
    x1 = fn(grid.nodes_x)

    fn = sint.interp1d(y, data[yi, :], bounds_error=False)
    y1 = fn(grid.nodes_x)

    plt.figure(figsize=(15, 8))

    plt.plot(x/1e3, np.abs(epm_fs[:, xi]), 'C0', lw=3, label='Inline empymod')
    plt.plot(x/1e3, np.abs(data[:, xi]), 'k--', label='Inline emg3d')
    plt.plot(grid.nodes_x/1e3, np.abs(x1), 'k*')

    plt.plot(y/1e3, np.abs(epm_fs[yi, :]), 'C1', lw=3,
             label='Crossline empymod')
    plt.plot(y/1e3, np.abs(data[yi, :]), 'k:', label='Crossline emg3d')
    plt.plot(grid.nodes_x/1e3, np.abs(y1), 'k*', label='Grid points emg3d')

    plt.yscale('log')
    plt.title(r'Inline and crossline $E_x$', fontsize=20)
    plt.xlabel('Offset (km)', fontsize=14)
    plt.ylabel(r'|Amplitude (V/m)|', fontsize=14)
    plt.legend()
    plt.show()


###############################################################################
# Full-space model for a finite length, finite strength, rotated bipole
# ---------------------------------------------------------------------
#
# empymod
# ```````

# Survey parameters
x = (np.arange(1025))*5-2560
rx = np.repeat([x, ], np.size(x), axis=0)
ry = rx.transpose()

# Model parameters
resh = 1.              # Horizontal resistivity
aniso = np.sqrt(2.)    # Anisotropy
resv = resh*aniso**2   # Vertical resistivity
mperm = 2.5            # Magnetic permeability
src = [-50, 50, -30, 30, -320., -280.]  # Source: [x1, x2, y1, y2, z1, z2]
src_c = np.mean(np.array(src).reshape(3, 2), 1).ravel()  # Center pts of source
zrec = -400.           # Receiver depth
freq = 0.77            # Frequency
strength = np.pi       # Source strength

# Input for empymod
model = {
    'src': src,
    'depth': [],
    'res': resh,
    'aniso': aniso,
    'strength': strength,
    'srcpts': 5,
    'freqtime': freq,
    'mpermH': mperm,  # <= Magnetic permeability
    'mpermV': mperm,  # <= Magnetic permeability
    'htarg': {'pts_per_dec': -1},
}

###############################################################################

epm_fs_x = empymod.bipole(rec=[rx.ravel(), ry.ravel(), zrec, 0, 0],
                          verb=3, **model).reshape(np.shape(rx))
epm_fs_y = empymod.bipole(rec=[rx.ravel(), ry.ravel(), zrec, 90, 0],
                          verb=1, **model).reshape(np.shape(rx))
epm_fs_z = empymod.bipole(rec=[rx.ravel(), ry.ravel(), zrec, 0, 90],
                          verb=1, **model).reshape(np.shape(rx))


###############################################################################
# emg3d
# `````

# Get computation domain as a function of frequency (resp., skin depth)
hx_min, xdomain = emg3d.meshes.get_domain(x0=src[0], freq=0.1, min_width=20)
hz_min, zdomain = emg3d.meshes.get_domain(x0=src[2], freq=0.1, min_width=20)

# Create stretched grid
nx = 2**7
hx = emg3d.meshes.get_stretched_h(hx_min, xdomain, nx, src_c[0])
hy = emg3d.meshes.get_stretched_h(hx_min, xdomain, nx, src_c[1])
hz = emg3d.meshes.get_stretched_h(hz_min, zdomain, nx, src_c[2])
pgrid = emg3d.TensorMesh([hx, hy, hz], x0=(xdomain[0], xdomain[0], zdomain[0]))
pgrid

###############################################################################

# Get the model, with magnetic permeability
pmodel = emg3d.Model(pgrid, property_x=resh, property_z=resv,
                     mu_r=mperm, mapping='Resistivity')

# Get the source field
sfield = emg3d.get_source_field(pgrid, src, freq, strength)

# Compute the electric field
efield = emg3d.solve(pgrid, pmodel, sfield, verb=4)


###############################################################################
# Plot
# ````
e3d_fs_x = emg3d.get_receiver(pgrid, efield.fx, (rx, ry, zrec))
plot_result_rel(epm_fs_x, e3d_fs_x, x, r'Diffusive Fullspace $E_x$',
                vmin=-12, vmax=-6, mode='abs')


###############################################################################

e3d_fs_y = emg3d.get_receiver(pgrid, efield.fy, (rx, ry, zrec))
plot_result_rel(epm_fs_y, e3d_fs_y, x, r'Diffusive Fullspace $E_y$',
                vmin=-12, vmax=-6, mode='abs')


###############################################################################

e3d_fs_z = emg3d.get_receiver(pgrid, efield.fz, (rx, ry, zrec))
plot_result_rel(epm_fs_z, e3d_fs_z, x, r'Diffusive Fullspace $E_z$',
                vmin=-12, vmax=-6, mode='abs')


###############################################################################

plot_lineplot_ex(x, x, e3d_fs_x.real, epm_fs_x.real, pgrid)

###############################################################################
emg3d.Report()
