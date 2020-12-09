"""
3. Interactive frequency selection
==================================

The time domain examples :ref:`sphx_glr_gallery_time_domain_fullspace.py` and
:ref:`sphx_glr_gallery_time_domain_marine_1D.py` use a relatively small range
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
import emg3d
# sphinx_gallery_thumbnail_path = '_static/images/GUI-freqselect.png'


###############################################################################

emg3d.Report()
