#############
emg3d Gallery
#############

:Release: |version|
:Date: |today|
:Source: `github.com/emsig/emg3d-gallery <https://github.com/emsig/emg3d-gallery>`_
:emsig: `emsig.xyz <https://emsig.xyz>`_

----

Gallery for `emg3d <https://emg3d.emsig.xyz>`_, a multigrid solver for 3D
electromagnetic diffusion [WeMS19]_.


.. toctree::
   :hidden:

   gallery/index
   references


Workflow
========

To install and activate the environment:

.. code-block:: console

    make install
    conda activate emg3d-gallery

To create the entire gallery:

.. code-block:: console

    make doc

To build the docs for just a particular file:

.. code-block:: console

    make example FILE=minimum_example.py

To remove the environment:

.. code-block:: console

    make remove
