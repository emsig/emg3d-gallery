"""
2. Transient CSEM for a marine model
====================================

Example how to use ``emg3d`` to model time-domain data using FFTLog.

"""
import emg3d
import empymod
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
clim = np.log10([0.2, 200])
# sphinx_gallery_thumbnail_number = 2

# Name is used to store the data for each frequency.
name = 'Marine-1D'


###############################################################################
# Model and Survey
# ----------------
#
# Model
# `````
# - 1 km water depth, 0.3 Ohm.m.
# - Target of 100 Ohm.m, 500 m thick, 1 km below seafloor.
# - Air set to 1e8 Ohm.m, background is 1 Ohm.m.
#
# Survey
# ``````
# - Source at origin, 50 m above seafloor.
# - Receiver on the seafloor at an inline-offset of 4 km.
# - Both source and receiver are x-directed electric dipoles.

src = [0, 0, -950]
rec = [4000, 0, -1000]
res = np.array([1, 100, 1, 0.3, 1e8])
depth = np.array([-2500, -2000, -1000, 0])


###############################################################################
# Here we create a dummy mesh with one cell in x- and y-directions, and our 1D
# model in z-direction. From this, we can interpolate the model to our varying
# meshes afterwards.

# Create the mesh.
orig_mesh = emg3d.TensorMesh(
    [[1, ], [1, ], np.r_[1000, np.diff(depth), 1000]],
    x0=('C', 'C', depth[0]-1000))

# Create a resistivity model using the 1D model and the above mesh.
orig_model = emg3d.Model(
        orig_mesh, property_x=np.array(res), mapping='Resistivity')

# QC.
orig_mesh.plot_3d_slicer(
        np.log10(orig_model.property_x), zlim=[-3000, 500], clim=clim)

# Get figure and axes
fig = plt.gcf()
axs = fig.get_children()

fig.suptitle(r'Resistivity model')

# Adjust the y-labels on the first subplot (XY)
axs[1].set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
axs[1].set_yticklabels(['', '', '0.0', '', ''])
axs[1].set_ylabel('y-direction (m)')

# Adjust x- and y-labels on the second subplot (XZ)
axs[2].set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
axs[2].set_xticklabels(['', '', '0.0', '', ''])
axs[2].set_xlabel('Easting (m)')

# plt.setp(axs[2].yaxis.get_majorticklabels(), rotation=90)
axs[2].set_yticks([0, -1000, -2000, -2500])
axs[2].set_yticklabels(['$0.0$', '-1.0', '-2.0', '-2.5'])
axs[2].set_ylabel('Elevation (km)')

# Adjust x-labels on the third subplot (ZY)
axs[3].set_xticks([400, 0, -1000, -2000, -2500, -3000])
axs[3].set_xticklabels(['', '$0.0$', '-1.0', '-2.0', '-2.5', '-3.0'])

# Adjust colorbar
axs[4].set_ylabel(r'$\rm{log}_{10}-$resistivity ($\Omega\,$m)')

# Ensure sufficient margins so nothing is clipped
plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9)


###############################################################################
# Fourier Transform parameters
# ----------------------------
#
# We only compute frequencies :math:`0.003 < f < 5` Hz, which yields enough
# precision for our purpose.
#
# This means, instead of 30 frequencies from 0.00002 - 12.6 Hz, we only need 16
# frequencies from 0.003 - 3.2 Hz.

# Define desired times.
time = np.logspace(-1, 2, 201)

# Initiate a Fourier instance
Fourier = emg3d.Fourier(
    time=time,
    fmin=0.003,
    fmax=5,
    ft='fftlog',  # Fourier transform to use
    ftarg={'pts_per_dec': 5, 'add_dec': [-2, 1], 'q': 0},
)

# Dense frequencies for comparison reasons
freq_dense = np.logspace(
        np.log10(Fourier.freq_req.min()),
        np.log10(Fourier.freq_req.max()),
        301
)


###############################################################################
# Frequency-domain computation
# ----------------------------

# To store the info of each frequency.
values = {}

gridinput = {
    'min_width': 100,    # Fix cell width within the survey domain to 100 m.
    'return_info': True,  # To get back some information for later.
    'max_domain': 50000,
    'verb': 0,
}

# Start the timer.
runtime = emg3d.utils.Time()

# Loop over frequencies, going from high to low.
for fi, frq in enumerate(Fourier.freq_calc[::-1]):
    print(f"  {fi+1:2}/{Fourier.freq_calc.size} :: {frq:10.6f} Hz")

    # Key is used to store the data etc.
    key = int(frq*1e6)

    # Initiate log for this frequency.
    values[key] = {}
    values[key]['freq'] = frq

    # Get cell widths and origin in each direction
    xx, x0, hix = emg3d.meshes.get_hx_h0(
        freq=frq, res=[0.3, 1e5], fixed=src[0], domain=[-100, 7100],
        **gridinput)
    yy, y0, hiy = emg3d.meshes.get_hx_h0(
        freq=frq, res=[0.3, 1e5], fixed=src[1], domain=[400, 400], **gridinput)
    zz, z0, hiz = emg3d.meshes.get_hx_h0(
        freq=frq, res=[0.3, 1., 1e5], domain=[-2300, 0], **gridinput,
        fixed=[depth[2], depth[3], depth[0]])

    # Store values in log.
    values[key]['alpha'] = [np.min([hix['amin'], hiy['amin'], hiz['amin']]),
                            np.max([hix['amax'], hiy['amax'], hiz['amax']])]
    values[key]['dminmax'] = [np.min([hix['dmin'], hiy['dmin'], hiz['dmin']]),
                              np.max([hix['dmax'], hiy['dmax'], hiz['dmax']])]

    # Initiate mesh.
    grid = emg3d.TensorMesh([xx, yy, zz], x0=np.array([x0, y0, z0]))
    # print(grid)
    values[key]['nC'] = grid.nC  # Store number of cells in log.

    # Generate model (interpolate on log-scale from our coarse model).
    res_x = 10**emg3d.maps.grid2grid(
            orig_mesh, np.log10(orig_model.property_x), grid, 'volume')
    model = emg3d.Model(grid, property_x=res_x, mapping='Resistivity')

    # QC
    # grid.plot_3d_slicer(np.log10(model.property_x),
    #                     zlim=[-3000, 500], clim=clim)

    # Define source.
    sfield = emg3d.get_source_field(
        grid, [src[0], src[1], src[2], 0, 0], frq, strength=0)

    # Solve the system.
    efield, info = emg3d.solve(
        grid, model, sfield, verb=2, return_info=True,
        sslsolver=True,  semicoarsening=True, linerelaxation=True,
    )

    # Store info
    values[key]['info'] = info

    # Store value
    values[key]['data'] = emg3d.get_receiver(
            grid, efield.fx, (rec[0], rec[1], rec[2]))

# Stop the timer.
total_time = runtime.runtime

# Store data and info to disk
emg3d.save(name+'.npz', values=values)


###############################################################################

# Load info and data
values = emg3d.load(name+'.npz')['values']

runtime = 0
for key, value in values.items():
    print(f"  {value['freq']:7.3f} Hz: "
          f"{value['info']['it_mg']:2g}/{value['info']['it_ssl']:g} it; "
          f"{value['info']['time']:4.0f} s; "
          f"a: {value['alpha'][0]:.3f} / {value['alpha'][1]:.3f} ; "
          f"nC: {value['nC']:8,.0f}; "
          f"a: {value['dminmax'][0]:5.0f} / {value['dminmax'][1]:7.0f}")
    runtime += value['info']['time']

print(f"\n                **** TOTAL RUNTIME :: "
      f"{runtime//60:.0f} min {runtime%60:.1f} s ****\n")


###############################################################################
# Load data, interpolate at receiver location
# ```````````````````````````````````````````

# Initiate data with zeros.
data = np.zeros((Fourier.freq_calc.size), dtype=complex)

# Loop over frequencies.
for fi, frq in enumerate(Fourier.freq_calc):
    key = str(int(frq*1e6))
    data[fi] = values[key]['data']


###############################################################################
# Interpolate missing frequencies and compute analytical result
# `````````````````````````````````````````````````````````````

data_int = Fourier.interpolate(data)

# Compute analytical result using empymod (epm)
epm_req = empymod.dipole(src, rec, depth, res, Fourier.freq_req, verb=1)
epm_calc = empymod.dipole(src, rec, depth, res, Fourier.freq_calc, verb=1)
epm_dense = empymod.dipole(src, rec, depth, res, freq_dense, verb=1)


###############################################################################
# Plot frequency-domain result
# ````````````````````````````

plt.figure(figsize=(10, 7))

# Real, log-lin
ax1 = plt.subplot(321)
plt.title('(a) log-lin Real')
plt.plot(freq_dense, 1e9*epm_dense.real, 'C1')
plt.plot(Fourier.freq_req, 1e9*data_int.real, 'k.', label='interpolated')
plt.plot(Fourier.freq_calc, 1e9*data.real, 'C0*')
plt.ylabel('$E_x$ (nV/m)')
plt.xscale('log')

# Real, log-symlog
ax3 = plt.subplot(323, sharex=ax1)
plt.title('(c) log-symlog Real')
plt.plot(freq_dense, 1e9*epm_dense.real, 'C1')
plt.plot(Fourier.freq_req, 1e9*data_int.real, 'k.')
plt.plot(Fourier.freq_calc, 1e9*data.real, 'C0*')
plt.ylabel('$E_x$ (nV/m)')
plt.xscale('log')
plt.yscale('symlog', linthresh=1e-5)

# Real, error
ax5 = plt.subplot(325, sharex=ax3)
plt.title('(e) clipped 0.01-10')

# Compute the error
err_int_r = np.clip(100*abs((data_int.real-epm_req.real) /
                            epm_req.real), 0.01, 10)
err_cal_r = np.clip(100*abs((data.real-epm_calc.real) /
                            epm_calc.real), 0.01, 10)

plt.ylabel('Rel. error %')
plt.plot(Fourier.freq_req, err_int_r, 'k.')
plt.plot(Fourier.freq_calc, err_cal_r, 'C0*')
plt.axhline(1, color='.4')

plt.xscale('log')
plt.yscale('log')
plt.ylim([0.008, 12])
plt.xlabel('Frequency (Hz)')

# Imaginary, log-lin
ax2 = plt.subplot(322)
plt.title('(b) log-lin Imag')
plt.plot(freq_dense, 1e9*epm_dense.imag, 'C1')
plt.plot(Fourier.freq_req, 1e9*data_int.imag, 'k.', label='interpolated')
plt.plot(Fourier.freq_calc, 1e9*data.imag, 'C0*')
plt.xscale('log')

# Imaginary, log-symlog
ax4 = plt.subplot(324, sharex=ax2)
plt.title('(d) log-symlog Imag')
plt.plot(freq_dense, 1e9*epm_dense.imag, 'C1')
plt.plot(Fourier.freq_req, 1e9*data_int.imag, 'k.')
plt.plot(Fourier.freq_calc, 1e9*data.imag, 'C0*')

plt.xscale('log')
plt.yscale('symlog', linthresh=1e-5)

# Imaginary, error
ax6 = plt.subplot(326, sharex=ax2)
plt.title('(f) clipped 0.01-10')

# Compute error
err_int_i = np.clip(100*abs((data_int.imag-epm_req.imag) /
                            epm_req.imag), 0.01, 10)
err_cal_i = np.clip(100*abs((data.imag-epm_calc.imag) /
                            epm_calc.imag), 0.01, 10)

plt.plot(Fourier.freq_req, err_int_i, 'k.')
plt.plot(Fourier.freq_calc, err_cal_i, 'C0*')
plt.axhline(1, color='.4')

plt.xscale('log')
plt.yscale('log')
plt.ylim([0.008, 12])
plt.xlabel('Frequency (Hz)')

plt.tight_layout()
plt.show()


###############################################################################
# Fourier Transform
# -----------------
#
# Carry-out Fourier transform, compute analytical result

# Compute corresponding time-domain signal.
data_time = Fourier.freq2time(data, rec[0])

# Analytical result
epm_time_precise = empymod.dipole(src, rec, depth, res, time, signal=0, verb=1)
epm_time = empymod.dipole(
        src, rec, depth, res, time, signal=0,
        ft=Fourier.ft, ftarg=Fourier.ftarg, verb=1)


###############################################################################
# Plot time-domain result
# ```````````````````````

fig = plt.figure(figsize=(8, 6))

# lin-lin
plt.subplot(221)
plt.title('(a) lin-lin')
plt.plot(time, epm_time_precise*1e9, 'k', lw=2)
plt.plot(time, epm_time*1e9, 'C1')
plt.plot(time, data_time*1e9, 'C0')
plt.ylabel('$E_x$ (nV/m)')
plt.xlim([-0.1, 12])
plt.xlabel('Time (s)')

# log-log
ax2 = plt.subplot(222)
plt.title('(b) log-log')
plt.plot(time, epm_time_precise*1e9, 'k', lw=2, label='empymod, analytical')
plt.plot(time, epm_time*1e9, 'C1', label='empymod, same FT as emg3d')
plt.plot(time, data_time*1e9, 'C0', label='emg3d, FFTLog')
perr = 100*(max(data_time)-max(epm_time_precise))/max(epm_time_precise)
plt.plot(-1, 1e9, 'k>', label=f"Peak error: {perr:.2f} %")
plt.ylim([1e-8, 1e-3])
plt.xscale('log')
plt.yscale('log')

# Error
ax4 = plt.subplot(224, sharex=ax2)
plt.title('(c) clipped 0.01-10 %')

# Compute error
err = np.clip(100*abs((data_time-epm_time_precise)/epm_time_precise), 0.01, 10)
err2 = np.clip(100*abs((epm_time-epm_time_precise)/epm_time_precise), 0.01, 10)

plt.loglog(time, err2, 'C1.')
plt.loglog(time, err, 'C0.')
plt.hlines(1, 0, 100)
plt.xlabel('Time (s)')
plt.ylabel('Rel. error %')
plt.ylim([0.008, 12])

plt.tight_layout()

# Plot peak error
ax2.legend(bbox_to_anchor=(-0.5, -0.5))

plt.show()


###############################################################################
# Further explanations to the results in the above figure:
# ````````````````````````````````````````````````````````
#
# - The black line is a very accurate result using ``empymod`` and the
#   following characteristics for the Fourier transform:
#
#   - Filter: Key 201 CosSin (2012)
#   - DLF type: Lagged Convolution
#   - Required frequencies: 251, from 1.5e-9 to 1.8e6 Hz
#
# - The blue result was equally obtained with ``empymod``, but with the
#   Fourier-transform parameters as used for ``emg3d``, hence FFTLog with 5 pts
#   per decade. However, in contrary to the red response, all frequencies are
#   computed, with a very high precision.
# - The red result is the result obtain with ``emg3d``.
#

emg3d.Report()
