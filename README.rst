.. image:: https://raw.githubusercontent.com/emsig/emg3d-logo/master/logo-emg3d-cut.png
   :target: https://emsig.github.io
   :alt: emg3d logo
   
----

.. sphinx-inclusion-marker


About ``emg3d``
===============

A multigrid solver for 3D electromagnetic diffusion with tri-axial electrical
anisotropy. The matrix-free solver can be used as main solver or as
preconditioner for one of the Krylov subspace methods implemented in
`scipy.sparse.linalg`, and the governing equations are discretized on a
staggered Yee grid. The code is written completely in Python using the
NumPy/SciPy-stack, where the most time- and memory-consuming parts are sped up
through jitted numba-functions.


More information
================

- **Website**: https://emsig.github.io,
- **Documentation**: https://emg3d.rtfd.io,
- **Source Code**: https://github.com/emsig/emg3d,


Workflow
========

To create the gallery:

.. code-block:: console

    make install
    conda activate emg3d-gallery
    make doc

To remove the environment:

.. code-block:: console

    make remove
