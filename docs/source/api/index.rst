.. module:: TrajAtlas

API
===
Import TrajAtlas as::

    import TrajAtlas as tja

TrajAtlas has a modular API, organized around multiple modules

- :mod:`TrajAtlas.model` projected osteogensis datasets to Differentiation Atas and OPCST model to reconstructed osteoblast differentiation.
- :mod:`TrajAtlas.TrajDiff` provided stastic framework to detect differential pseudotime abundance and differential gene expressions.
- :mod:`TrajAtlas.TRAVMap` utilized NMF to detect pseudotemporal gene modules.


.. toctree::
    :caption: API
    :maxdepth: 2

    model
    trajdiff
    TRAVMap
