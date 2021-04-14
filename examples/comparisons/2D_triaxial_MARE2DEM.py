"""
2. MARE2DEM: 2D with tri-axial anisotropy
=========================================

``MARE2DEM`` is an open-source, finite element 2.5D code for controlled-source
electromagnetic (CSEM) and magnetotelluric (MT) data, see
`mare2dem.bitbucket.io <https://mare2dem.bitbucket.io>`_.

"""
import emg3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# sphinx_gallery_thumbnail_path = '_static/thumbs/MARE2DEM.png'

return  # will break but create the title # TODO Not Updated Yet


###############################################################################
# Load MARE2DEM result
# --------------------

# Location of data files.
data_url = 'https://raw.githubusercontent.com/emsig/emg3d-gallery/'
data_url += 'master/examples/data/MARE2DEM/'

mar_tg = np.loadtxt(data_url+'triaxial.0.resp', skiprows=93, usecols=6)
mar_tg = mar_tg[::2] + 1j*mar_tg[1::2]

mar_bg = np.loadtxt(data_url+'triaxial-BG.0.resp', skiprows=93, usecols=6)
mar_bg = mar_bg[::2] + 1j*mar_bg[1::2]


###############################################################################
# emg3d
# -----

src = (50, 0, -1950, 0, 0)  # Source location [x, y, z, azimuth, dip]
rec = (np.arange(80)*100+2050, 0, -1999.9)
freq = 0.5                  # Frequency (Hz)


###############################################################################

gridinput = {
    'min_width': 50,
    'verb': 0,
    'freq': freq,
    'max_domain': 50000,
}

# Get cell widths and origin in each direction
xx, x0 = emg3d.meshes.get_hx_h0(
    res=[0.3, 1e5], fixed=src[0], domain=[-100, 10100],
    **gridinput)
yy, y0 = emg3d.meshes.get_hx_h0(
    res=[0.3, 1e5], fixed=src[1], domain=[400, 400], **gridinput)
zz, z0 = emg3d.meshes.get_hx_h0(
    res=[0.3, 1., 1e5], domain=[-4200, 0], **gridinput,
    fixed=[-2000, 0, -4200])

# Initiate mesh.
grid = emg3d.TensorMesh([xx, yy, zz], x0=np.array([x0, y0, z0]))
grid

###############################################################################

xx = (grid.cell_centers[:, 0] > 0)*(grid.cell_centers[:, 0] <= 6000)
zz = (grid.cell_centers[:, 2] > -4200)*(grid.cell_centers[:, 2] < -4000)


###############################################################################

# Background
res_x_full = 2*np.ones(grid.n_cells)
res_y_full = 1*np.ones(grid.n_cells)
res_z_full = 3*np.ones(grid.n_cells)

# Water - isotropic
res_x_full[grid.cell_centers[:, 2] >= -2000] = 0.3
res_y_full[grid.cell_centers[:, 2] >= -2000] = 0.3
res_z_full[grid.cell_centers[:, 2] >= -2000] = 0.3

# Air - isotropic
res_x_full[grid.cell_centers[:, 2] >= 0] = 1e10
res_y_full[grid.cell_centers[:, 2] >= 0] = 1e10
res_z_full[grid.cell_centers[:, 2] >= 0] = 1e10

# Target
res_x_full_tg = res_x_full.copy()
res_y_full_tg = res_y_full.copy()
res_z_full_tg = res_z_full.copy()
res_x_full_tg[xx*zz] = 200
res_y_full_tg[xx*zz] = 100
res_z_full_tg[xx*zz] = 300

# Collect models
model_bg = emg3d.Model(
        grid, property_x=res_x_full, property_y=res_y_full,
        property_z=res_z_full, mapping='Resistivity')
model_tg = emg3d.Model(
        grid, property_x=res_x_full_tg, property_y=res_y_full_tg,
        property_z=res_z_full_tg, mapping='Resistivity')

# Create source field
sfield = emg3d.get_source_field(grid, src, freq, 0)

# Solver parameters
sparams = {
    'verb': 4,
    'sslsolver': True,
    'semicoarsening': True,
    'linerelaxation': True
}

# QC model
grid.plot_3d_slicer(
        model_tg.property_x, zlim=[-6000, 500],
        pcolor_opts={'norm': LogNorm(vmin=0.3, vmax=300)})


###############################################################################
# Model background
# ````````````````

efield_bg = emg3d.solve(grid, model_bg, sfield, **sparams)
em3_bg = emg3d.get_receiver(grid, efield_bg.fx, rec)


###############################################################################
# Model target
# ````````````

efield_tg = emg3d.solve(grid, model_tg, sfield, **sparams)
em3_tg = emg3d.get_receiver(grid, efield_tg.fx, rec)


###############################################################################
# Plot
# ----

plt.figure(figsize=(9, 4))

# REAL PART
ax1 = plt.subplot2grid((4, 2), (0, 0), rowspan=3)
plt.title(r'|Real|')

plt.plot(rec[0]/1e3, 1e12*np.abs(mar_tg.real), '-', label='MARE2DEM target')
plt.plot(rec[0]/1e3, 1e12*np.abs(mar_bg.real), '-', label='MARE2DEM BG')

plt.plot(rec[0]/1e3, 1e12*np.abs(em3_tg.real), 'k--')
plt.plot(rec[0]/1e3, 1e12*np.abs(em3_bg.real), 'k-.')

plt.ylabel('$E_x$ (pV/m)')
ax1.set_xticklabels([])
plt.yscale('log')
plt.legend(loc=4, ncol=2)
plt.grid(axis='y', c='0.9')


# NORMALIZED DIFFERENCE REAL
ax2 = plt.subplot2grid((4, 2), (3, 0))

nd_bg_re = np.clip(200*abs(mar_bg.real-em3_bg.real) /
                   (abs(mar_bg.real) + abs(em3_bg.real)), 0.1, 10)
nd_tg_re = np.clip(200*abs(mar_tg.real-em3_tg.real) /
                   (abs(mar_tg.real) + abs(em3_tg.real)), 0.1, 10)

plt.semilogy(rec[0]/1e3, nd_tg_re, '.', label='target')
plt.semilogy(rec[0]/1e3, nd_bg_re, '.', label='background')

plt.ylabel('Norm. Diff (%)')
plt.xlabel('Offset (km)')
plt.yscale('log')
plt.xlim(ax1.get_xlim())
plt.ylim([8e-2, 12])
plt.yticks([0.1, 1, 10], ('0.1', '1', '10'))
plt.grid(axis='y', c='0.9')


# IMAGINARY PART
ax3 = plt.subplot2grid((4, 2), (0, 1), rowspan=3, sharey=ax1)
plt.title(r'|Imaginary|')

plt.plot(rec[0]/1e3, 1e12*np.abs(mar_tg.imag), '-')
plt.plot(rec[0]/1e3, 1e12*np.abs(mar_bg.imag), '-')

plt.plot(rec[0]/1e3, 1e12*np.abs(em3_tg.imag), 'k--', label='emg3d target')
plt.plot(rec[0]/1e3, 1e12*np.abs(em3_bg.imag), 'k-.', label='emg3d BG')

plt.ylabel('$E_x$ (pV/m)')
ax3.set_xticklabels([])
plt.legend(loc=3, ncol=2)
ax3.yaxis.tick_right()
ax3.yaxis.set_label_position("right")
plt.grid(axis='y', c='0.9')


# NORMALIZED DIFFERENCE IMAG
ax4 = plt.subplot2grid((4, 2), (3, 1))

nd_bg_im = np.clip(200*abs(mar_bg.imag-em3_bg.imag) /
                   (abs(mar_bg.imag) + abs(em3_bg.imag)), 0.1, 10)
nd_tg_im = np.clip(200*abs(mar_tg.imag-em3_tg.imag) /
                   (abs(mar_tg.imag) + abs(em3_tg.imag)), 0.1, 10)

plt.semilogy(rec[0]/1e3, nd_tg_im, 'C0.', label='Target')
plt.semilogy(rec[0]/1e3, nd_bg_im, 'C1.', label='BGC')

plt.xlabel('Offset (km)')
plt.yscale('log')
plt.xlim(ax1.get_xlim())
plt.ylabel('Norm. diff. %')
plt.ylim([8e-2, 12])
plt.yticks([0.1, 1, 10], ('0.1', '1', '10'))
ax4.yaxis.tick_right()
ax4.yaxis.set_label_position("right")
plt.grid(axis='y', c='0.9')


# SWITCH OFF SPINES
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.spines['left'].set_visible(False)

plt.tight_layout()
plt.show()


###############################################################################

emg3d.Report()
