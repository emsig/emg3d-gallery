"""
1. CPU and RAM usage
====================

Measuring and estimating runtime and memory usage of ``emg3d`` as a function of
model size.

The actually computed results further down are only for relatively small
models, as these examples in the gallery are run very often. Here therefore the
results of two larger runs that were run on a cluster:

Example CPU
-----------

.. figure:: ../../_static/images/CPU.png
   :scale: 66 %
   :align: center
   :alt: Runtime
   :name: cpu-usage

Example RAM
-----------

.. figure:: ../../_static/images/RAM.png
   :scale: 66 %
   :align: center
   :alt: RAM
   :name: ram-usage

Check-out the old versions for more information with regards to the above
figures:

- `4a_RAM-requirements.ipynb
  <https://github.com/empymod/emg3d-examples/blob/master/4a_RAM-requirements.ipynb>`_,
- `4b_Runtime.ipynb
  <https://github.com/empymod/emg3d-examples/blob/master/4b_Runtime.ipynb>`_.

"""
import emg3d
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
plt.style.use('ggplot')
# sphinx_gallery_thumbnail_path = '_static/images/RAM.png'


###############################################################################
# Model
# -----
#
# This is the actual model it runs. Adjust this to your needs.

def compute(nx):
    """Simple computation routine.

    - Model size is nx * nx * nx, centered around the origin.
    - Source is at the origin, x-directed.
    - Frequency is 1 Hz.
    - Homogenous space of 1 Ohm.m.

    """

    # Grid
    hx = np.ones(nx)*50
    x0 = -nx//2*50
    grid = emg3d.TensorMesh([hx, hx, hx], x0=(x0, x0, x0))

    # Source location and frequency
    src = [0, 0, 0, 0, 0]
    freq = 1.0

    # Resistivity model
    res = 1.

    # Model and source field
    model = emg3d.Model(grid, property_x=res, mapping='Resistivity')
    sfield = emg3d.get_source_field(grid, src, freq=freq, strength=0)

    # Compute the field
    _, inf = emg3d.solve(grid, model, sfield, verb=1, return_info=True)

    return inf['time']


###############################################################################
# Loop over model sizes
# ---------------------
#
# These are the actual ``nx``-sizes it tests. Adjust to your needs.

nsizes = np.array([32, 48, 64, 96, 128, 192])  # , 256, 384, 512, 768, 1024])
memory = np.zeros(nsizes.shape)
runtime = np.zeros(nsizes.shape)

# Loop over nx
for i, nx in enumerate(nsizes):
    print(f"  => {nx}^3 = {nx**3:12,d} cells")
    mem, time = memory_usage((compute, (nx, ), {}), retval=True)
    memory[i] = max(mem)
    runtime[i] = time


###############################################################################
# Plot CPU
# ````````

plt.figure()
plt.title('Runtime')
plt.loglog(nsizes**3/1e6, runtime, '.-')
plt.xlabel('Number of cells (in millions)')
plt.ylabel('CPU (s)')
plt.axis('equal')
plt.show()

###############################################################################
# Plot RAM
# ````````


plt.figure()
plt.title('Memory')
plt.loglog(nsizes**3/1e6, memory/1e3, '-', zorder=10)
plt.xlabel('Number of cells (in millions)')
plt.ylabel('RAM (GB)')
plt.axis('equal')
plt.show()


###############################################################################

emg3d.Report('memory_profiler')
