.. module:: TrajAtlas.model

Model
=======
**Model** enable user to easily projecting their own datasets to our **Differentiation Atlas** and **OsteoProgenitor Cells-Specific Trajectory (OPCST) model**.


With this module, users are able to:

1. Obtain the latent space of their own datasets.
2. Annotate their datasets with seven-level annotations.
3. Predict differentiation paths from different osteoprogenitor cells (OPCs).
4. Obtain common pseudotime to indicate osteoblast differentiation progression.


.. currentmodule:: TrajAtlas

.. autosummary::
    :toctree: _autosummary/model

    model.ProjectData
    model.label_transfer
    model.pseduo_predict

