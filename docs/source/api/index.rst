.. module:: TrajAtlas

API
===
Import TrajAtlas as::

    import TrajAtlas as tja

TrajAtlas has a modular API, organized around multiple modules

- :mod:`TrajAtlas.model` compute cell-cell transition matrices using various input data modalities,
  including RNA velocity, any pseudotime, a developmental potential, experimental time points, and more.
- :mod:`TrajAtlas.TrajDiff` use the cell-cell transition matrix to derive insights about cellular dynamics,
  for example, they compute initial and terminal states, fate probabilities, and driver genes. Our recommended
  estimator is the :class:`~cellrank.estimators.GPCCA` estimator.
- :mod:`TrajAtlas.TRAVMap` use the cell-cell transition matrix to derive insights about cellular dynamics,
  for example, they compute initial and terminal states, fate probabilities, and driver genes. Our recommended
  estimator is the :class:`~cellrank.estimators.GPCCA` estimator.


.. toctree::
    :caption: API
    :maxdepth: 2

    trajdiff
    model
    TRAVMap
