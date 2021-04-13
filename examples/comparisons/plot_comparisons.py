"""
Plot routines for electric examples
===================================
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as sint
from matplotlib.colors import SymLogNorm
plt.style.use('ggplot')
# sphinx_gallery_thumbnail_path = '_static/thumbs/tools.png'


###############################################################################
def plot_data(ax, data, x, vmin, vmax, mode):
    """Plot single slice."""

    # Set limits.
    ax.set_xlim(min(x)/1000, max(x)/1000)
    ax.set_ylim(min(x)/1000, max(x)/1000)
    ax.axis("equal")

    # pcolormesh, depending on mode.
    if isinstance(mode, str):
        if mode == "abs":
            dat = np.log10(np.abs(data))
            kwargs = {'cmap': "viridis", 'vmin': vmin, 'vmax': vmax}
        else:
            dat = data
            norm = SymLogNorm(
                    linthresh=10**vmin, vmin=-10**vmax, vmax=10**vmax)
            kwargs = {'cmap': "PuOr_r", 'norm': norm}
    else:
        dat = np.log10(data)
        cmap = plt.cm.get_cmap("RdBu_r", 8)
        kwargs = {'cmap': cmap, 'vmin': vmin, 'vmax': vmax}

    cf = ax.pcolormesh(
            x/1000, x/1000, dat, **kwargs, linewidth=0, shading='nearest')

    return cf


###############################################################################
def plot_sections(epm, e3d, x, title, vmin=-15., vmax=-7., mode="log"):
    """Plot Re/Im sections for empymod/emg3d, and the relative error."""

    # Start figure.
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.3))
    case = "" if mode == "log" else "|"

    # Plot Re(data)
    axs[0, 0].set_title(r"(a) "+case+"Re(empymod)"+case)
    cf0 = plot_data(axs[0, 0], epm.real, x, vmin, vmax, mode)

    axs[0, 1].set_title(r"(b) "+case+"Re(emg3d)"+case)
    plot_data(axs[0, 1], e3d.real, x, vmin, vmax, mode)

    axs[0, 2].set_title(r"(c) Error real part")
    rel_error = 100*np.abs((epm.real - e3d.real) / epm.real)
    cf2 = plot_data(axs[0, 2], rel_error, x, vmin=-2, vmax=2, mode=True)

    # Plot Im(data)
    axs[1, 0].set_title(r"(d) "+case+"Im(empymod)"+case)
    plot_data(axs[1, 0], epm.imag, x, vmin, vmax, mode)

    axs[1, 1].set_title(r"(e) "+case+"Im(emg3d)"+case)
    plot_data(axs[1, 1], e3d.imag, x, vmin, vmax, mode)

    axs[1, 2].set_title(r"(f) Error imaginary part")
    rel_error = 100*np.abs((epm.imag - e3d.imag) / epm.imag)
    plot_data(axs[1, 2], rel_error, x, vmin=-2, vmax=2, mode=True)

    # Colorbars
    unit = "A/m" if 'H' in title else "V/m"
    fig.colorbar(cf0, ax=axs[0, :], label=r"$\log_{10}$ Amplitude"+f"({unit})")
    cbar = fig.colorbar(cf2, ax=axs[1, :], label=r"Relative Error")
    cbar.set_ticks([-2, -1, 0, 1, 2])
    cbar.ax.set_yticklabels([r"$0.01\,\%$", r"$0.1\,\%$", r"$1\,\%$",
                             r"$10\,\%$", r"$100\,\%$"])

    # Axis label
    fig.text(0.4, 0.05, "Inline Offset (km)", fontsize=14)
    fig.text(0.08, 0.4, "Crossline Offset (km)", rotation=90, fontsize=14)

    # Title
    fig.suptitle(title, y=1, fontsize=20)
    fig.show()


###############################################################################
def plot_line(x, y, e3d, epm, grid, add_title):
    """Plot an inline and a crossline comparison btw. emg3d and empymod."""

    # Get middle indices.
    xi = x.size//2
    yi = y.size//2

    # Interpolate fields at corresponding cell centers.
    xfn = sint.interp1d(x, e3d[:, xi], bounds_error=False)
    ccx = grid.cell_centers_x

    yfn = sint.interp1d(y, e3d[yi, :], bounds_error=False)
    ccy = grid.cell_centers_y

    # Plot.
    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot inline comparison.
    ax.plot(x/1e3, epm[:, xi].amp(), 'C0', lw=3, label='Inline empymod')
    ax.plot(x/1e3, e3d[:, xi].amp(), 'k--', label='Inline emg3d')
    ax.plot(ccx/1e3, np.abs(xfn(ccx)), 'k*')

    # Plot crossline comparison.
    ax.plot(y/1e3, epm[yi, :].amp(), 'C1', lw=3, label='Crossline empymod')
    ax.plot(y/1e3, e3d[yi, :].amp(), 'k:', label='Crossline emg3d')
    ax.plot(ccy/1e3, np.abs(yfn(ccy)), 'k*', label='Grid points emg3d')

    # Labels etc.
    ax.set_yscale('log')
    unit = "A/m" if 'H' in add_title else "V/m"
    ax.set_title(f"Inline and crossline ${add_title}$", fontsize=20)
    ax.set_xlabel("Offset (km)", fontsize=14)
    ax.set_ylabel(f"|Amplitude ({unit})|", fontsize=14)
    ax.legend()
    fig.show()
