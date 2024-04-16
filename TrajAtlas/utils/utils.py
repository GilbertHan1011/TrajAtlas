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

def process_subset(i, j, sampleDict, lineageDict, adata, timeDict, timeBin, cell_threshold):
    agg_dict = {gene: "mean" for gene in adata.var_names}
    subsetCell = list(set(sampleDict[i]) & set(lineageDict[j]))
    if len(subsetCell) < cell_threshold:
        return None
    subsetAdata = adata[subsetCell]
    timeVal = list(map(lambda val: timeDict[val], subsetCell))
    timeBinVal = list(map(lambda val: timeBin[val], subsetCell))
    subsetAdata = adata[subsetCell]
    geneMat = pd.DataFrame(subsetAdata.X.toarray())
    varName = subsetAdata.var_names
    geneMat.columns=subsetAdata.var_names
    geneMat.index=subsetAdata.obs_names
    geneMat["dpt_bin"]=np.array(timeBinVal)
    gene_agg = geneMat.groupby("dpt_bin").agg(agg_dict)
    bin_mask=geneMat.groupby("dpt_bin").size()<5
    gene_agg[bin_mask]=np.nan
    geneMat=geneMat.loc[:,varName]
    pearsonCorrDict = {}
    maxRowsDict={}
    sumValuesDict={}
    for k in range(geneMat.shape[1]):
        geneArr = geneMat.iloc[:, k]
        geneAggArr=gene_agg.iloc[:,k]
        if geneAggArr.sum()== 0:
            geneName = varName[k]
            maxRowsDict[geneName]=0
            sumValuesDict[geneName]=0
            pearsonCorrDict[geneName]=0
        else:
            pearson, _ = pearsonr(geneArr, np.array(timeVal))
            geneName = varName[k]
            pearsonCorrDict[geneName] = pearson
            max_row = geneAggArr.idxmax()
            maxRowsDict[geneName] = max_row
            sumValuesDict[geneName]=geneAggArr.sum()
    
    pearsonCorrDf = pd.DataFrame.from_dict(pearsonCorrDict, orient="index").fillna(0)
    pearsonCorrDf.columns = [i + "_sep_" + j]
    maxRowDf = pd.DataFrame.from_dict(maxRowsDict, orient="index").fillna(0)
    maxRowDf.columns = [i + "_sep_" + j]
    sumValDf = pd.DataFrame.from_dict(sumValuesDict, orient="index").fillna(0)
    sumValDf.columns = [i + "_sep_" + j]
  
    return pearsonCorrDf, maxRowDf, sumValDf

def getAttribute(adata,lineage: list or None = ["Fibroblast", "LepR_BMSC", "MSC", "Chondro"],
                 peudotime_key="pseduoPred",njobs=-1,cell_threshold: int or None=40):
    adata.obs[['pred_lineage_lepr', 'pred_lineage_msc', 'pred_lineage_chondro',"pred_lineage_fibro"]]=adata.obs[['pred_lineage_lepr', 'pred_lineage_msc', 'pred_lineage_chondro',"pred_lineage_fibro"]].astype("bool")
    lineageDict={"Chondro":adata.obs.index[adata.obs["pred_lineage_chondro"]],
            "LepR_BMSC":adata.obs.index[adata.obs["pred_lineage_lepr"]],
            "Fibroblast":adata.obs.index[adata.obs["pred_lineage_fibro"]],
            "MSC":adata.obs.index[adata.obs["pred_lineage_msc"]]}


    adata.obs["Cell"]=adata.obs_names
    result_df = adata.obs.groupby('sample')['Cell'].agg(list).reset_index()
    sampleDict = dict(zip(result_df['sample'], result_df['Cell']))
    dptValue=adata.obs[peudotime_key]
    timeDict=pd.DataFrame(dptValue).iloc[:,0].to_dict()
    #prepare pseudotime bin
    timeDict=pd.DataFrame(dptValue).iloc[:,0].to_dict()
    num_bins = 10
    hist, bin_edges = np.histogram(dptValue, bins=num_bins)
    timeBin=np.digitize(dptValue, bin_edges)
    timeBin=pd.DataFrame(timeBin)
    timeBin.index=dptValue.index
    timeBin[timeBin==11]=10
    timeBin=timeBin.squeeze().to_dict()
    dfs = pd.DataFrame()
    df=adata.obs["sample"]
    for i in lineage:
        keyDf=pd.DataFrame(df.unique(),columns=["sample"])
        keyDf["Lineage"]=i
        dfs=pd.concat([dfs,keyDf])
    key_pairs= [(dfs['sample'].iloc[i], dfs['Lineage'].iloc[i]) for i in range(dfs.shape[0])]
    partial_process_subset = partial(process_subset, sampleDict=sampleDict, lineageDict=lineageDict, adata=adata, timeDict=timeDict, timeBin=timeBin, cell_threshold=cell_threshold)
    results = Parallel(n_jobs=njobs)(delayed(partial_process_subset)(*key_pair) for key_pair in tqdm(key_pairs))
    pearson_results = [result[0] for result in results if result is not None]
    peak_results = [result[1] for result in results if result is not None]
    expr_results = [result[2] for result in results if result is not None]

    # Combine the results into the final DataFrame
    corr = pd.concat(pearson_results, axis=1)
    peak = pd.concat(peak_results, axis=1)
    expr = pd.concat(expr_results, axis=1)
    #=form mudata
    exprAdata=sc.AnnData(expr.T)
    peakAdata=sc.AnnData(peak.T)
    corrAdata=sc.AnnData(corr.T)
    corr_mod = np.where(corr >= 0, np.sqrt(corr), -np.sqrt(-corr))
    corr_mod=pd.DataFrame(corr_mod)
    corr_mod.columns=corr_mod.columns
    corr_mod.index=corr_mod.index
    expr_mod = expr.apply(lambda row: (row) / (row.max()), axis=1)
    exprAdata=sc.AnnData(expr.T)
    peakAdata=sc.AnnData(peak.T)
    corrAdata=sc.AnnData(corr.T)
    exprAdata.layers["mod"]=expr_mod.T
    peakAdata.layers["mod"]=peak.T
    corrAdata.layers["mod"]=corr_mod.T
    tvmap=mu.MuData({"corr":corrAdata, "expr": exprAdata,"peak":peakAdata})
    return(tvmap)

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