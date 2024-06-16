from __future__ import annotations
from joblib import Parallel, delayed
from functools import partial
from tqdm import tqdm
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import muon as mu
from mudata import MuData
import scanpy as sc
import PyComplexHeatmap as pch
from anndata import AnnData
import matplotlib.pyplot as plt
from typing import List
from TrajAtlas.utils._docs import d

def makeDotTable(tvMap: MuData, gene: List,sample: List):
    """Get dataframe to plot trajectory dotplot

    Parameters
    ----------
        tvMap
            Mudata generated from TrajAtlas.utils.getAttribute.
        gene
            Genes to plot
        sample
            Samples to plot

        
    Returns:
        Dataframe.

    """

    exprMod=tvMap["expr"].layers["mod"]
    peakMod=tvMap["peak"].layers["mod"]
    corrMod=tvMap["corr"].layers["mod"]
    exprMod=pd.DataFrame(tvMap["expr"].layers["mod"],index=tvMap.obs_names,columns=tvMap["expr"].var_names).T
    peakMod=pd.DataFrame(tvMap["peak"].layers["mod"],index=tvMap.obs_names,columns=tvMap["peak"].var_names).T
    corrMod=pd.DataFrame(tvMap["corr"].layers["mod"],index=tvMap.obs_names,columns=tvMap["corr"].var_names).T
    selectCorr=corrMod.loc[gene]
    selectExpr=exprMod.loc[gene]
    selectPeak=peakMod.loc[gene]
    selectCorr=selectCorr.loc[:,sample]
    selectExpr=selectExpr.loc[:,sample]
    selectPeak=selectPeak.loc[:,sample]
    corrLong=selectCorr.stack().reset_index(name="Corr")
    exprLong=selectExpr.stack().reset_index(name="Expr")
    peakLong=selectPeak.stack().reset_index(name="Peak")
    peakLong['Stage']=peakLong.Peak.apply(lambda x:'End' if x>=7 else 'Middle' if x >= 3 else 'Start')
    combineDf=corrLong
    combineDf["Expr"]=exprLong['Expr']
    combineDf["Peak"]=peakLong['Stage']
    return(combineDf)
    
def trajDotplot(geneTb,col_split:int or pd.Series or pd.DataFrame or None, 
                ratio:int = 60, 
                show_rownames: bool =True,
                show_colnames:bool = False,
                spines : bool=True,
                **kwargs):
    """Plot trajectory dotplot.

    Parameters
    ----------
        geneTb
            Dataframe generated with TrajAtlas.utils.makeDotTable.
        col_split
            Column split
        show_rownames
            Whether to show rownames
        show_colnames
            Whether to show colnames
        spines
            Whether to have spine
        
    Returns:
        Nothing. Plot trajectory dotplot.

    """
    
    cm = pch.DotClustermapPlotter(geneTb,x='level_1',y='level_0',value='Corr',c='Corr',hue='Peak',s="Expr",
                               marker={'Start':'o','Middle':'D','End':'s'},col_split=col_split,
                                 ratio=ratio,show_rownames=show_rownames,
                                  spines=spines,show_colnames=show_colnames,**kwargs)


@d.dedent
def split_umap(
    adata:AnnData, 
    split_by:str,
    basis:str="X_umap",
    ncol:int=2,
    nrow=None,
      **kwargs):
    """Create split view of gene expression on reduction.

    .. seealso::
        - See :doc:`../../../tutorial/4.15_GEP` for how to
        identified gene expression program associated with differentiated genes.

    Parameters
    ----------
        %(adata)s
        split_by
            Slot in :attr:`adata.obs <anndata.AnnData.obs>` to splited by
        basis
            Reduction name in :attr:`adata.obsm <anndata.AnnData.obsm>`
        ncol
            Number of column.
        nrow
            Number of row.
        
    Returns:
        Nothing. Plot reduction with a view of gene expression splited by the interested covariate..

    """
    categories = adata.obs[split_by].cat.categories
    if nrow is None:
        nrow = int(np.ceil(len(categories) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(5*ncol, 4*nrow))
    axs = axs.flatten()
    for i, cat in enumerate(categories):
        ax = axs[i]
        sc.pl.embedding(adata[adata.obs[split_by] == cat], ax=ax, show=False, title=cat,basis = basis, **kwargs)
    plt.tight_layout()