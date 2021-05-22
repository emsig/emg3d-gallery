"""
9. Transient CSEM
=================

blabla

The multigrid solver ``emg3d`` is a frequency-domain (or Laplace-domain)
code. However, we can also use ``emg3d`` to model time-domain data.

See the repo https://github.com/emsig/article-TDEM for more info.

blabla

Example how to use ``emg3d`` to model time-domain data, using FFTLog and DLF.

This example is based on the first example (Figures 3-4) of [MuWS08]_, those
original results are shown at the bottom of this example.

See the repo https://github.com/emsig/article-TDEM for more info.

Interactive frequency selection
-------------------------------

The time domain examples :ref:`sphx_glr_gallery_t-domain_fullspace.py` and
:ref:`sphx_glr_gallery_t-domain_marine_1D.py` use a relatively small range
of frequencies to go from the frequency domain to the time domain. The chosen
frequencies where designed in an interactive GUI, which you can find in the
repo `empymod/frequency-selection
<https://github.com/emsig/frequency-selection>`_.

The repo contains a Python file ``freqselect.py``, which contains the routines,
and two notebooks:

1. ``AdaptiveFrequencySelection.ipynb``: Reproduces and improves a previously
   published, adaptive frequency-selection scheme.
2. ``InteractiveFrequencySelection.ipynb``: The interactive GUI that was used
   to design the selected frequencies in the above mentioned examples.


A screenshot of the GUI for the interactive frequency selection is shown in the
following figure:

.. figure:: ../../_static/images/GUI-freqselect.png
   :scale: 66 %
   :align: center
   :alt: Frequency-selection App
   :name: freqselect


The GUI uses the 1D modeller ``empymod`` and a layered model, and internally
the ``Fourier`` class of the 3D modeller ``emg3d``. The following parameters
can be specified interactively:

- points per decade
- frequency range (min/max)
- offset
- Fourier transform (FFTLog or DLF with different filters)
- signal (impulse or switch-on/-off)

Other parameters have to be specified fix when initiating the widget.

"""
import os
import emg3d
import empymod
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')
# sphinx_gallery_thumbnail_number = 2


# path and name is used to store the data for each frequency.
path = os.path.join('..', 'download', '..')
name = 'Fullspace'

###############################################################################
# Model and Survey
# ----------------
#
# Model
# `````
#
# - Homogeneous fullspace of 1 Ohm.m.
#
# Survey
# ``````
# - Source at origin.
# - Receiver at an inline-offset of 900 m.
# - Both source and receiver are x-directed electric dipoles.

src = [0, 0, 0, 0, 0]
rec = [900, 0, 0, 0, 0]
res = 1                  # Fullspace resistivity
depth = []


###############################################################################
# Fourier Transforms parameters
# -----------------------------
#
# We only compute frequencies :math:`0.05 < f < 21` Hz, which yields enough
# precision for our purpose.
#
# This means, instead of 30 frequencies from 0.0002 - 126.4 Hz, we only need 14
# frequencies from 0.05 - 20.0 Hz.

# Define desired times.
time = np.logspace(-2, 1, 201)

# Initiate a Fourier instance
Fourier = emg3d.Fourier(
    time=time,
    fmin=0.05,
    fmax=21,
    ft='fftlog',  # Fourier transform to use
    ftarg={'pts_per_dec': 5, 'add_dec': [-2, 1], 'q': 0},
)

# Dense frequencies for comparison reasons
freq_dense = np.logspace(
        np.log10(Fourier.freq_required.min()),
        np.log10(Fourier.freq_required.max()),
        301
)


###############################################################################
# Frequency-domain computation
# ----------------------------

# To store the info of each frequency.
values = {}
#
gridinput = {
    'res': res,               # Fullspace resistivity.
    'min_width': [20., 40.],  # Restr. the cell width within the survey domain.
    'return_info': True,      # To get back some information for later.
    'pps': 12,                # Many points, to have a small min cell width.
    'alpha': [1, 1.3, 0.01],  # Lower the alpha will improve the result, but
    'verb': 0,                # slow down computation.
}
#
# Start the timer.
runtime = emg3d.utils.Timer()
#
# Loop over frequencies, going from high to low.
old_grid = None
for fi, frq in enumerate(Fourier.freq_compute[::-1]):
    print(f"  {fi+1:2}/{Fourier.freq_compute.size} :: {frq:10.6f} Hz")

    # Initiate log for this frequency.
    thislog = {}
    thislog['freq'] = frq

    # Get cell widths and origin in each direction
    xx, x0, hix = emg3d.meshes.get_hx_h0(
        freq=frq, fixed=src[0], domain=[-200, 1100], **gridinput)
    yz, yz0, hiyz = emg3d.meshes.get_hx_h0(
        freq=frq, fixed=src[1], domain=[-50, 50], **gridinput)

    # Store values in log.
    thislog['alpha'] = [np.min([hix['amin'], hiyz['amin']]),
                        np.max([hix['amax'], hiyz['amax']])]
    thislog['dminmax'] = [np.min([hix['dmin'], hiyz['dmin']]),
                          np.max([hix['dmax'], hiyz['dmax']])]

    # Initiate mesh.
    grid = emg3d.TensorMesh([xx, yz, yz], origin=np.array([x0, yz0, yz0]))
    # print(grid)
    thislog['nC'] = grid.nC  # Store number of cells in log.

    # Interpolate the starting electric field from the last one (can speed-up
    # the computation).
    if fi == 0:
        efield = emg3d.Field(grid, freq=frq)
    else:
        efield = emg3d.maps.grid2grid(old_grid, efield, grid,
                                      method='cubic', extrapolate=False)
        efield = emg3d.Field(grid, efield, freq=frq)

    # Generate model
    model = emg3d.Model(grid, property_x=res, mapping='Resistivity')

    # Define source.
    sfield = emg3d.get_source_field(
        grid, [src[0], src[1], src[2], 0, 0], frq, strength=0)

    # Solve the system.
    info = emg3d.solve(
        grid, model, sfield, efield=efield,
        verb=2, return_info=True,
        sslsolver=True,  semicoarsening=True, linerelaxation=True,
    )

    # Store info
    thislog['info'] = info

    # Store value
    thislog['data'] = emg3d.get_receiver(
            grid, efield.fx, (rec[0], rec[1], rec[2]))

    # Store thislog in values.
    values[int(frq*1e6)] = thislog

    # Store the grid for the interpolation.
    old_grid = grid

# Stop the timer.
total_time = runtime.runtime

# Store data and info to disk
emg3d.save(path + name + '.npz', values=values)


###############################################################################

# Load info and data
values = emg3d.load(path + name + '.npz')['values']

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
data = np.zeros(Fourier.freq_compute.size, dtype=complex)

# Loop over frequencies.
for fi, frq in enumerate(Fourier.freq_compute):
    key = str(int(frq*1e6))
    data[fi] = values[key]['data']


###############################################################################
# 1. Using FFTLog
# ---------------
#
# Interpolate missing frequencies and compute analytical result
# `````````````````````````````````````````````````````````````

data_int = Fourier.interpolate(data)

# Compute analytical result using empymod (epm)
epm_req = empymod.dipole(src, rec, depth, res, Fourier.freq_required, verb=1)
epm_calc = empymod.dipole(src, rec, depth, res, Fourier.freq_compute, verb=1)
epm_dense = empymod.dipole(src, rec, depth, res, freq_dense, verb=1)


###############################################################################
# Plot frequency-domain result
# ````````````````````````````

plt.figure(figsize=(10, 7))

# Real, log-lin
ax1 = plt.subplot(321)
plt.title('(a) log-lin Real')
plt.plot(freq_dense, 1e9*epm_dense.real, 'C1')
plt.plot(Fourier.freq_required, 1e9*data_int.real, 'k.', label='interpolated')
plt.plot(Fourier.freq_compute, 1e9*data.real, 'C0*')
plt.ylabel('$E_x$ (nV/m)')
plt.xscale('log')

# Real, log-symlog
ax3 = plt.subplot(323, sharex=ax1)
plt.title('(c) log-symlog Real')
plt.plot(freq_dense, 1e9*epm_dense.real, 'C1')
plt.plot(Fourier.freq_required, 1e9*data_int.real, 'k.')
plt.plot(Fourier.freq_compute, 1e9*data.real, 'C0*')
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
plt.plot(Fourier.freq_required, err_int_r, 'k.')
plt.plot(Fourier.freq_compute, err_cal_r, 'C0*')
plt.axhline(1, color='.4')

plt.xscale('log')
plt.yscale('log')
plt.ylim([0.008, 12])
plt.xlabel('Frequency (Hz)')

# Imaginary, log-lin
ax2 = plt.subplot(322)
plt.title('(b) log-lin Imag')
plt.plot(freq_dense, 1e9*epm_dense.imag, 'C1')
plt.plot(Fourier.freq_required, 1e9*data_int.imag, 'k.', label='interpolated')
plt.plot(Fourier.freq_compute, 1e9*data.imag, 'C0*')
plt.xscale('log')

# Imaginary, log-symlog
ax4 = plt.subplot(324, sharex=ax2)
plt.title('(d) log-symlog Imag')
plt.plot(freq_dense, 1e9*epm_dense.imag, 'C1')
plt.plot(Fourier.freq_required, 1e9*data_int.imag, 'k.')
plt.plot(Fourier.freq_compute, 1e9*data.imag, 'C0*')

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

plt.plot(Fourier.freq_required, err_int_i, 'k.')
plt.plot(Fourier.freq_compute, err_cal_i, 'C0*')
plt.axhline(1, color='.4')

plt.xscale('log')
plt.yscale('log')
plt.ylim([0.008, 12])
plt.xlabel('Frequency (Hz)')

plt.tight_layout()
plt.show()


###############################################################################
# Fourier Transform
# `````````````````
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
plt.xlim([0, 2])
plt.xlabel('Time (s)')

# log-log
ax2 = plt.subplot(222)
plt.title('(b) log-log')
plt.plot(time, epm_time_precise*1e9, 'k', lw=2, label='empymod, analytical')
plt.plot(time, epm_time*1e9, 'C1', label='empymod, same FT as emg3d')
plt.plot(time, data_time*1e9, 'C0', label='emg3d, FFTLog')
perr = 100*(max(data_time)-max(epm_time_precise))/max(epm_time_precise)
plt.plot(-1, 1e9, 'k>', label=f"Peak error: {perr:.2f} %")
plt.xlim([1.5e-2, 2e0])
plt.ylim([1e-6, 1e0])
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
# - The black line is the analytical fullspace solution in the time-domain.
# - The blue result was obtained with empymod, using the same Fourier-transform
#   parameters as used for ``emg3d``, hence FFTLog with 5 pts per decade.
#   However, in contrary to the red response, all frequencies are computed,
#   with a very high precision.
# - The red result is the result obtain with ``emg3d``.
#
# 2. Using DLF
# ------------
#
# We use the same frequencies and computed data as in the FFTLog example, but
# apply the digital-linear-filter method for the transformation.
#
# Fourier Transform parameters for DLF
# ````````````````````````````````````

Fourier_dlf = emg3d.Fourier(
    time=time,
    fmin=0.05,
    fmax=21,
    ft='dlf',  # Fourier transform to use
    ftarg={'pts_per_dec': -1},
    input_freq=Fourier.freq_required,  # Use same freqs as in above example
)


# Dense frequencies for comparison reasons
freq_dense_dlf = np.logspace(
        np.log10(Fourier_dlf.freq_required.min()),
        np.log10(Fourier_dlf.freq_required.max()),
        301)

# Get data
data_int_dlf = Fourier_dlf.interpolate(data)

# Compute analytical result using empymod (epm)
epm_req_dlf = empymod.dipole(
        src, rec, depth, res, Fourier_dlf.freq_required, verb=1)
epm_calc_dlf = empymod.dipole(
        src, rec, depth, res, Fourier_dlf.freq_compute, verb=1)
epm_dense_dlf = empymod.dipole(src, rec, depth, res, freq_dense_dlf, verb=1)


###############################################################################
# Interpolate missing frequencies and compute analytical result
# `````````````````````````````````````````````````````````````

data_int_dlf = Fourier_dlf.interpolate(data)

# Compute analytical result using empymod (epm)
epm_req_dlf = empymod.dipole(
        src, rec, depth, res, Fourier_dlf.freq_required, verb=1)
epm_calc_dlf = empymod.dipole(
        src, rec, depth, res, Fourier_dlf.freq_compute, verb=1)
epm_dense_dlf = empymod.dipole(src, rec, depth, res, freq_dense_dlf, verb=1)


###############################################################################
# Plot frequency-domain result
# ````````````````````````````

plt.figure(figsize=(10, 7))

# Real, log-lin
ax1 = plt.subplot(321)
plt.title('(a) log-lin Real')
plt.plot(freq_dense_dlf, 1e9*epm_dense_dlf.real, 'C1')
plt.plot(Fourier_dlf.freq_required, 1e9*data_int_dlf.real, 'k--',
         label='interpolated')
plt.plot(Fourier_dlf.freq_compute, 1e9*data.real, 'C0*')
plt.ylabel('$E_x$ (nV/m)')
plt.xscale('log')

# Real, log-symlog
ax3 = plt.subplot(323, sharex=ax1)
plt.title('(c) log-symlog Real')
plt.plot(freq_dense_dlf, 1e9*epm_dense_dlf.real, 'C1')
plt.plot(Fourier_dlf.freq_required, 1e9*data_int_dlf.real, 'k--')
plt.plot(Fourier_dlf.freq_compute, 1e9*data.real, 'C0*')
plt.ylabel('$E_x$ (nV/m)')
plt.xscale('log')
plt.yscale('symlog', linthresh=1e-5)

# Real, error
ax5 = plt.subplot(325, sharex=ax3)
plt.title('(e) clipped 0.01-10')

# Compute the error
err_int_r = np.clip(100*abs((data_int_dlf.real-epm_req_dlf.real) /
                            epm_req_dlf.real), 0.01, 10)
err_cal_r = np.clip(100*abs((data.real-epm_calc_dlf.real) /
                            epm_calc_dlf.real), 0.01, 10)

plt.ylabel('Rel. error %')
plt.plot(Fourier_dlf.freq_required, err_int_r, 'k.')
plt.plot(Fourier_dlf.freq_compute, err_cal_r, 'C0*')
plt.axhline(1, color='.4')

plt.xscale('log')
plt.yscale('log')
plt.ylim([0.008, 12])
plt.xlabel('Frequency (Hz)')

# Imaginary, log-lin
ax2 = plt.subplot(322)
plt.title('(b) log-lin Imag')
plt.plot(freq_dense_dlf, 1e9*epm_dense_dlf.imag, 'C1')
plt.plot(Fourier_dlf.freq_required, 1e9*data_int_dlf.imag, 'k--',
         label='interpolated')
plt.plot(Fourier_dlf.freq_compute, 1e9*data.imag, 'C0*')
plt.xscale('log')

# Imaginary, log-symlog
ax4 = plt.subplot(324, sharex=ax2)
plt.title('(d) log-symlog Imag')
plt.plot(freq_dense_dlf, 1e9*epm_dense_dlf.imag, 'C1')
plt.plot(Fourier_dlf.freq_required, 1e9*data_int_dlf.imag, 'k--')
plt.plot(Fourier_dlf.freq_compute, 1e9*data.imag, 'C0*')

plt.xscale('log')
plt.yscale('symlog', linthresh=1e-5)

# Imaginary, error
ax6 = plt.subplot(326, sharex=ax2)
plt.title('(f) clipped 0.01-10')

# Compute error
err_int_i = np.clip(100*abs((data_int_dlf.imag-epm_req_dlf.imag) /
                            epm_req_dlf.imag), 0.01, 10)
err_cal_i = np.clip(100*abs((data.imag-epm_calc_dlf.imag) /
                            epm_calc_dlf.imag), 0.01, 10)

plt.plot(Fourier_dlf.freq_required, err_int_i, 'k.')
plt.plot(Fourier_dlf.freq_compute, err_cal_i, 'C0*')
plt.axhline(1, color='.4')

plt.xscale('log')
plt.yscale('log')
plt.ylim([0.008, 12])
plt.xlabel('Frequency (Hz)')

plt.tight_layout()
plt.show()


###############################################################################
# Fourier Transform
# `````````````````
#
# Carry-out Fourier transform.

# Compute corresponding time-domain signal.
data_time_dlf = Fourier_dlf.freq2time(data, rec[0])


###############################################################################
# Plot time-domain result
# ```````````````````````

fig = plt.figure(figsize=(8, 6))

# lin-lin
plt.subplot(221)
plt.title('(a) lin-lin')
plt.plot(time, epm_time_precise*1e9, 'k', lw=2)
plt.plot(time, epm_time*1e9, 'C1')
plt.plot(time, data_time_dlf*1e9, 'C0')
plt.ylabel('$E_x$ (nV/m)')
plt.xlim([0, 2])
plt.xlabel('Time (s)')

# log-log
ax2 = plt.subplot(222)
plt.title('(b) log-log')
plt.plot(time, epm_time_precise*1e9, 'k', lw=2, label='empymod, analytical')
plt.plot(time, epm_time*1e9, 'C1', label='empymod, same FT as emg3d')
plt.plot(time, data_time_dlf*1e9, 'C0', label='emg3d, DLF')
perr = 100*(max(data_time_dlf)-max(epm_time_precise))/max(epm_time_precise)
plt.plot(-1, 1e9, 'k>', label=f"Peak error: {perr:.2f} %")
plt.xlim([1.5e-2, 2e0])
plt.ylim([1e-6, 1e0])
plt.xscale('log')
plt.yscale('log')

# Error
ax4 = plt.subplot(224, sharex=ax2)
plt.title('(c) clipped 0.01-10 %')

# Compute error
err = np.clip(100*abs((data_time_dlf-epm_time_precise)/epm_time_precise),
              0.01, 10)
err2 = np.clip(100*abs((epm_time-epm_time_precise)/epm_time_precise),
               0.01, 10)

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
# 3. Results from Mulder et al., 2008, Geophysics
# -----------------------------------------------
#
# Total computation time (CPU) is 13,632 s, which corresponds to
# 3 h 47 min 12 s.
#
# .. figure:: ../../_static/images/Mulder2008_Figs_3-4_Tab_1.png
#    :scale: 66 %
#    :align: center
#    :alt: Results Mulder et al., 2008.
#    :name: Muld08_Fig3-4
#
#    Figures 3 and 4 and Table 1, page F5 of Mulder et al., 2008, Geophysics.
#
# The published example took roughly **3.75 hours**, whereas here we need just
# a **few minutes**. There are two main reasons for the speed gain:
#
# 1. gridding and
# 2. frequency selection.
#
# Note re first point, gridding: We implemented here an adaptive gridding with
# various number of cells. Our computation uses meshes between 36,864
# (64x24x24) and 102,400 (64x40x40) cells, whereas Mulder et al., 2008, used
# 2,097,152 (128x128x128) for all frequencies.
#
# Note re second point, frequency selection: We only used 14 frequencies from
# 0.05-20 Hz, whereas Mulder et al., 2008, used 26 frequencies from 0.01-100
# Hz. Have a look at the example
# :ref:`sphx_glr_gallery_t-domain_freqselect.py`.

emg3d.Report()
