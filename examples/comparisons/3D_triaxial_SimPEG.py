"""
3. SimPEG: 3D with tri-axial anisotropy
=======================================

`SimPEG <https://simpeg.xyz>`_ is an open source python package for simulation
and gradient based parameter estimation in geophysical applications. Here we
compare ``emg3d`` with ``SimPEG`` using the forward solver ``Pardiso``.

Note, in order to use the ``Pardiso``-solver ``pymatsolver`` has to be
installed via ``conda``, not via ``pip``!
"""
import emg3d
import SimPEG
import discretize
import numpy as np
import pymatsolver
import SimPEG.electromagnetics.frequency_domain as FDEM
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# sphinx_gallery_thumbnail_path = '_static/thumbs/SimPEG.png'

###############################################################################
# Model and survey parameters
# ---------------------------

# Depths (0 is sea-surface)
water_depth = 1000
target_x = np.r_[-500, 500]
target_y = target_x
target_z = -water_depth + np.r_[-400, -100]

# Resistivities
res_air = 2e8
res_sea = 0.33
res_back = [1., 2., 3.]  # Background in x-, y-, and z-directions
res_target = 100.

freq = 1.0

src = [-100, 100, 0, 0, -900, -900]


###############################################################################
# Mesh and source-field
# ---------------------

# skin depth
skin_depth = 503*np.sqrt(res_back[0]/freq)
print(f"\nThe skin_depth is {skin_depth} m.\n")

cs = 100    # 100 m min_width of cells

pf = 1.15   # Padding factor x- and y-directions
pfz = 1.35  # .              z-direction
npadx = 12  # Nr of padding in x- and y-directions
npadz = 9   # .                z-direction

domain_x = 4000            # x- and y-domain
domain_z = - target_z[0]   # z-domain

# Create mesh
mesh = discretize.TensorMesh(
    [[(cs, npadx, -pf), (cs, int(domain_x/cs)), (cs, npadx, pf)],
     [(cs, npadx, -pf), (cs, int(domain_x/cs)), (cs, npadx, pf)],
     [(cs, npadz, -pfz), (cs, int(domain_z/cs)), (cs, npadz, pfz)]],
)

# Center mesh
mesh.x0 = np.r_[-mesh.h[0].sum()/2, -mesh.h[1].sum()/2,
                -mesh.h[2][:-npadz].sum()]

# Create the source field for this mesh and given frequency
sfield = emg3d.get_source_field(mesh, src, freq, strength=0)

# We take the receiver locations at the actual CCx-locations
rec_x = mesh.cell_centers_x[12:-12]
print(f"Receiver locations:\n{rec_x}\n")

mesh


###############################################################################
# Create model
# ------------

# Layered_background
res_x = res_air*np.ones(mesh.nC)
res_x[mesh.gridCC[:, 2] <= 0] = res_sea

res_y = res_x.copy()
res_z = res_x.copy()

res_x[mesh.gridCC[:, 2] <= -water_depth] = res_back[0]
res_y[mesh.gridCC[:, 2] <= -water_depth] = res_back[1]
res_z[mesh.gridCC[:, 2] <= -water_depth] = res_back[2]

res_x_bg = res_x.copy()
res_y_bg = res_y.copy()
res_z_bg = res_z.copy()

# Include the target
target_inds = (
    (mesh.gridCC[:, 0] >= target_x[0]) & (mesh.gridCC[:, 0] <= target_x[1]) &
    (mesh.gridCC[:, 1] >= target_y[0]) & (mesh.gridCC[:, 1] <= target_y[1]) &
    (mesh.gridCC[:, 2] >= target_z[0]) & (mesh.gridCC[:, 2] <= target_z[1])
)
res_x[target_inds] = res_target
res_y[target_inds] = res_target
res_z[target_inds] = res_target

# Create emg3d-models for given frequency
pmodel = emg3d.Model(
        mesh, property_x=res_x, property_y=res_y,
        property_z=res_z, mapping='Resistivity')
pmodel_bg = emg3d.Model(
        mesh, property_x=res_x_bg, property_y=res_y_bg,
        property_z=res_z_bg, mapping='Resistivity')

# Plot a slice
mesh.plot_3d_slicer(pmodel.property_x, zslice=-1100, clim=[0, 2],
                    xlim=(-4000, 4000), ylim=(-4000, 4000), zlim=(-2000, 500))


###############################################################################
# Compute ``emg3d``
# -------------------

em3_tg = emg3d.solve(mesh, pmodel, sfield, verb=4, nu_pre=0,
                     semicoarsening=True)


###############################################################################

em3_bg = emg3d.solve(mesh, pmodel_bg, sfield, verb=4, nu_pre=0,
                     semicoarsening=True)


###############################################################################
# Compute ``SimPEG``
# --------------------

# Set up the receivers
rx_locs = discretize.utils.ndgrid([rec_x, np.r_[0], np.r_[-water_depth]])
rx_list = [
    FDEM.receivers.PointElectricField(
        orientation='x', component="real", locations=rx_locs),
    FDEM.receivers.PointElectricField(
        orientation='x', component="imag", locations=rx_locs)
]

# We use the emg3d-source-vector, to ensure we use the same in both cases
src_sp = FDEM.sources.RawVec_e(rx_list, s_e=sfield.vector, frequency=freq)
src_list = [src_sp]
survey = FDEM.Survey(src_list)

# Define the Simulation
sim = FDEM.simulation.Simulation3DElectricField(
        mesh,
        survey=survey,
        sigmaMap=SimPEG.maps.IdentityMap(mesh),
        solver=pymatsolver.Pardiso,
)

###############################################################################
spg_tg_dobs = sim.dpred(np.vstack([1./res_x, 1./res_y, 1./res_z]).T)
spg_tg = SimPEG.survey.Data(survey, dobs=spg_tg_dobs)


###############################################################################
spg_bg_dobs = sim.dpred(
        np.vstack([1./res_x_bg, 1./res_y_bg, 1./res_z_bg]).T)
spg_bg = SimPEG.survey.Data(survey, dobs=spg_bg_dobs)


###############################################################################
# Plot result
# -----------
ix1, ix2 = 12, 12
iy = 32
iz = 13

plt.figure(figsize=(9, 6))

plt.subplot(221)
plt.title('|Real(response)|')
plt.semilogy(rec_x/1e3, np.abs(em3_bg.fx[ix1:-ix2, iy, iz].real))
plt.semilogy(rec_x/1e3, np.abs(em3_tg.fx[ix1:-ix2, iy, iz].real))
plt.semilogy(rec_x/1e3, np.abs(spg_bg[src_sp, rx_list[0]]), 'C4--')
plt.semilogy(rec_x/1e3, np.abs(spg_tg[src_sp, rx_list[0]]), 'C5--')
plt.xlabel('Offset (km)')
plt.ylabel('$E_x$ (V/m)')

plt.subplot(223)
plt.title('|Imag(response)|')
plt.semilogy(rec_x/1e3, np.abs(em3_bg.fx[ix1:-ix2, iy, iz].imag),
             label='emg3d BG')
plt.semilogy(rec_x/1e3, np.abs(em3_tg.fx[ix1:-ix2, iy, iz].imag),
             label='emg3d target')
plt.semilogy(rec_x/1e3, np.abs(spg_bg[src_sp, rx_list[1]]), 'C4--',
             label='SimPEG BG')
plt.semilogy(rec_x/1e3, np.abs(spg_tg[src_sp, rx_list[1]]), 'C5--',
             label='SimPEG target')
plt.xlabel('Offset (km)')
plt.ylabel('$E_x$ (V/m)')
plt.legend()

plt.subplot(222)
plt.title('Difference Real')

nrmsd_bg = 200*(abs(spg_bg[src_sp, rx_list[0]] -
                    em3_bg.fx[ix1:-ix2, iy, iz].real) /
                (abs(em3_bg.fx[ix1:-ix2, iy, iz].real) +
                 abs(spg_bg[src_sp, rx_list[0]])))
nrmsd_tg = 200*(abs(spg_tg[src_sp, rx_list[0]] -
                    em3_tg.fx[ix1:-ix2, iy, iz].real) /
                (abs(em3_tg.fx[ix1:-ix2, iy, iz].real) +
                 abs(spg_tg[src_sp, rx_list[0]])))

plt.semilogy(rec_x/1e3, nrmsd_bg, label='BG')
plt.semilogy(rec_x/1e3, nrmsd_tg, label='target')

plt.xlabel('Offset (km)')
plt.ylabel('NRMSD (%)')
plt.legend()

plt.subplot(224)
plt.title('Difference Imag')

nrmsd_bg = 200*(abs(spg_bg[src_sp, rx_list[1]] -
                    em3_bg.fx[ix1:-ix2, iy, iz].imag) /
                (abs(em3_bg.fx[ix1:-ix2, iy, iz].imag) +
                 abs(spg_bg[src_sp, rx_list[1]])))
nrmsd_tg = 200*(abs(spg_tg[src_sp, rx_list[1]] -
                    em3_tg.fx[ix1:-ix2, iy, iz].imag) /
                (abs(em3_tg.fx[ix1:-ix2, iy, iz].imag) +
                 abs(spg_tg[src_sp, rx_list[1]])))

plt.semilogy(rec_x/1e3, nrmsd_bg, label='BG')
plt.semilogy(rec_x/1e3, nrmsd_tg, label='target')

plt.xlabel('Offset (km)')
plt.ylabel('NRMSD (%)')
plt.legend()

plt.tight_layout()
plt.show()


###############################################################################

emg3d.Report([SimPEG, pymatsolver])
