<p align="center">
  <img width="300" src="img/logo.png">
</p>

<h1><p align="center">Unraveling multi-scale differentiation heterogeneity with trajectory-centric framework</p></h1>


The central idea of **TrajAtlas** revolves around axis-based analysis. Although initially designed for osteogenesis datasets, 
it can be applied to almost any type of dataset as long as you have an "axis" (such as pseudotime or gene expression patterns).

<p align="center">
  <img width="600" src="img/Fig1_v2.png">
</p>

## Getting started

Please refer to the [documentation](https://trajatlas.readthedocs.io/en/stable/). In particular:

- [API documentation](https://trajatlas.readthedocs.io/en/stable/api/index.html)
- [Tutorial](https://trajatlas.readthedocs.io/en/stable/tutorial/index.html)

If you want to do pseudotime analysis, please check 


   [Differential Abundance analysis](https://trajatlas.readthedocs.io/en/stable/tutorial/Step2_differential_abundance.html)

   [Differential Expression analysis](https://trajatlas.readthedocs.io/en/stable/tutorial/step3_DE.html)


If you are interested in gene expression programs, we have developed a pipeline based on it. Please check 

  [TrajAtlas meets gene expression program](https://trajatlas.readthedocs.io/en/stable/tutorial/4.15_GEP.html)


If you are using gene set scoring for function inference, you can utilize TrajAtlas to identify which genes are more significant within the gene sets.

  [TrajAtlas meets gene set scoring](https://trajatlas.readthedocs.io/en/stable/tutorial/gssaxis.html)

If you have osteogenesis datasets, you can project your datasets onto our model.

   [Projecting osteogenesis datasets](https://trajatlas.readthedocs.io/en/stable/tutorial/1_OPCST_projecting.html)


## Installation

You need to have Python 3.8 or newer installed on your system. If you don't have Python installed, we recommend installing [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

You can install **TrajAtlas** with following codes

```bash
pip install TrajAtlas
```

## Citation

**TrajAtlas** recently has posted on bioRxiv. If **TrajAtlas** helps you, please cite our [work](https://www.biorxiv.org/content/10.1101/2024.05.28.596174v1.full).