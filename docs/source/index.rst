Welcome to TrajAtlas' documentation!
==============================================

**TrajAtlas** is a trajectory-centric framework for unnraveling multi-scale differentiation heterogeneity 
across population-level trajectories. With **TrajAtlas**, you are able to explore heterogeneity among cells, genes, and
gene modules across large-scale trajectories!


The central idea of **TrajAtlas** revolves around axis-based analysis. Although initially designed for osteogenesis datasets, 
it can be applied to almost any type of dataset as long as you have an "axis" (such as pseudotime or gene expression patterns).

If you want to do pseudotime analysis, please check 

   :doc:`Differential Abundance analysis <tutorial/Step2_differential_abundance>`

   :doc:`Differential Expression analysis <tutorial/step3_DE>`


If you are interested in gene expression programs, we have developed a pipeline based on it. Please check 

   :doc:`TrajAtlas meets gene expression program <tutorial/4.15_GEP>`


If you are using gene set scoring for function inference, you can utilize TrajAtlas to identify which genes are more significant within the gene sets.

   :doc:`TrajAtlas meets gene set scoring <tutorial/gssaxis>`

If you have osteogenesis datasets, you can project your datasets onto our model.

   :doc:`Projecting osteogenesis datasets <tutorial/1_OPCST_projecting>`

To grasp the foundational concepts of TrajAtlas, please refer to the detailed information provided in :doc:`introduction/index`.

.. image:: ../../img/Fig1_v2.png
    :width: 600px
    :align: center
    :class: only-light

.. image:: ../../img/Fig1_v2.png
    :width: 600px
    :align: center
    :class: only-dark




.. note::

   This project is under active development.


TrajAtlas' Key Applications
---------------------------
- Differential pseudotime analysis, including:

   :doc:`Differential Abundance analysis <tutorial/Step2_differential_abundance>`

   :doc:`Differential Expression analysis <tutorial/step3_DE>`

- Detecting pseudotemporal gene module

   :doc:`Detecting pseudotemporal gene module <tutorial/pseudotemporal_gene_module>`

- Projecting osteogenesis datasets to Differential Atlas and OPCST model, including:

   :doc:`Projecting osteogenesis datasets <tutorial/1_OPCST_projecting>`

- Beyond Trajectory, including:

   :doc:`TrajAtlas meets gene expression program <tutorial/4.15_GEP>`

   :doc:`TrajAtlas meets gene set scoring <tutorial/gssaxis>`

- … and much more, check out our :doc:`API <api/index>`

Getting Started with TrajAtlas
-----------------------------
We have :doc:`tutorial/index` to help you getting started. To see TrajAtlas in action, please explore our
`manuscript <https://www.biorxiv.org/content/10.1101/2024.05.28.596174v1.full>`_ .

Contributing
------------
We actively encourage any contribution! To get started, please check out the :doc:`contributing`.



.. toctree::
   :caption: General
   :maxdepth: 3
   :hidden:

   installation
   tutorial/index
   api/index
   release_notes
   contributing
   references

.. toctree::
   :caption: About
   :maxdepth: 3
   :hidden:

   introduction/index
   about/team
   about/cite
   GitHub <https://github.com/GilbertHan1011/TrajAtlas>

