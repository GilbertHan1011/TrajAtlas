About TrajAtlas
===============================

Basic thoughts behind TrajAtlas
-------------------------

This software is designed to dissect dynamic process in a comprehensive way. 
Instead of explore static perspective on cell type and gene expression, we propose a “trajectory-centric analysis” method
for understanding dynamic processes, such as differentiation, cell cycle.

We initially chose osteoblast differentiation as a template, considering that osteoblasts can differentiate from various 
osteoprogenitors across a wide range of tissues and ages :cite:`mizoguchiDiverseOriginBoneforming2021` :cite:`liInsightsSkeletalStem2022`. This diversity suggests significant heterogeneity within this process.

We integrated trajectories from 27 datasets to build **Differentiation Atlas**, to explore the heterogeneity of osteoprogenitor cells.
Then we reconstructed osteogensis differentiation with a OsteoProgenitor Cells-Specific trajectory (OPCST) Model. In this model, we inferred 
differentation path from four osteoprogenitors to osteoblast and a common pseudotime to predict differentiation process.
We implemented this functionality within the TrajAtlas.model, allowing users to apply it to their own datasets.

In multi-stage differentiation processes like osteoblast differentiation, it is crucial to accurately identify the stage at which 
differential genes exert their influence. However, existing methods like **Lamian** :cite:`houStatisticalFrameworkDifferential2023` 
and **Condiments**:cite:`rouxdebezieuxTrajectoryInferenceMultiple2024` often struggle to infer differential 
abundance and expression within specific differentiation stages. This gap motivated the development of TrajDiff, which aims to uncover 
changes in cell abundance and expression across differentiation stages.

While **TrajDiff** is capable of detecting differences among multiple trajectories, dissecting heterogeneity within population-level trajectories 
remains challenging. To address this, we developed **TRAVMap**. **TRAVMap** is designed to identify pseudotemporal gene modules among trajectories. 
Furthermore, to enhance our understanding of heterogeneity among trajectories, we employed trajectory representation learning 
using information from genes and gene modules. Through trajectory reduction techniques, we enable the visualization of gene module activity 
and gene expression across population-level trajectories.

