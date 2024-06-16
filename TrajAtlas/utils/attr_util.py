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
from scipy.stats import ttest_rel
from TrajAtlas.utils._docs import d

def _process_gene(chunk,geneMat,gene_agg,varName,timeVal):
    pearsonCorrDict = {}
    maxRowsDict={}
    sumValuesDict={}
    for k in chunk:
        geneArr = geneMat.iloc[:, k]
        geneAggArr = gene_agg.iloc[:, k]
        if geneAggArr.sum() == 0:
            geneName = varName[k]
            maxRowsDict[geneName] = 0
            sumValuesDict[geneName] = 0
            pearsonCorrDict[geneName] = 0
        else:
            pearson, _ = pearsonr(geneArr, np.array(timeVal))
            geneName = varName[k]
            pearsonCorrDict[geneName] = pearson
            max_row = geneAggArr.idxmax()
            maxRowsDict[geneName] = max_row
            sumValuesDict[geneName] = geneAggArr.sum()
    return([maxRowsDict,sumValuesDict,pearsonCorrDict])

def _process_subset(adata, timeDict,subsetCell:list = None, subsetGene:List = None,num_bins = 10, cell_threshold=40,njobs=-1):
    if subsetGene is None:
        subsetGene=adata.var_names
    if subsetCell is None:
        subsetCell=adata.obs_names
    agg_dict = {gene: "mean" for gene in subsetGene}
    if len(subsetCell) < cell_threshold:
        return None
    subsetAdata = adata[subsetCell,subsetGene]
    # make bin dict
    dptValue = np.array(list(timeDict.values()))
    hist, bin_edges = np.histogram(dptValue, bins=num_bins)
    timeBin=np.digitize(dptValue, bin_edges)
    timeBin=pd.DataFrame(timeBin)
    timeBin.index=np.array(list(timeDict.keys()))
    timeBin[timeBin==11]=10
    timeBin=timeBin.squeeze().to_dict()
    timeVal = list(map(lambda val: timeDict[val], subsetCell))
    timeBinVal = list(map(lambda val: timeBin[val], subsetCell))
    geneMat = pd.DataFrame(subsetAdata.X.toarray())
    varName = subsetAdata.var_names
    geneMat.columns=subsetAdata.var_names
    geneMat.index=subsetAdata.obs_names
    geneMat["dpt_bin"]=np.array(timeBinVal)
    gene_agg = geneMat.groupby("dpt_bin").agg(agg_dict)
    bin_mask=geneMat.groupby("dpt_bin").size()<5
    gene_agg[bin_mask]=np.nan
    geneMat=geneMat.loc[:,varName]
    # chunk and parallel
        # chunk and parallel
    if njobs == -1:
        chunkSplit = 100
    else:
        chunkSplit=njobs
    chunk_size = geneMat.shape[1] // chunkSplit
    chunks = [range(i, min(i + chunk_size, geneMat.shape[1])) for i in range(0, geneMat.shape[1], chunk_size)]

    # Parallelize the computation
    partial_process_gene = partial(_process_gene, geneMat=geneMat, gene_agg=gene_agg, varName=varName,timeVal=timeVal)
    results = Parallel(n_jobs=njobs)(delayed(partial_process_gene)(chunk) for chunk in chunks)
    res1 = [result[0] for result in results if result is not None]
    res2 = [result[1] for result in results if result is not None]
    res3 = [result[2] for result in results if result is not None]
    combined_dict = {}
    # Iterate over each dictionary in the list and update the combined dictionary
    for d in res1:
        combined_dict.update(d)
    peak = pd.DataFrame.from_dict(combined_dict,orient='index', columns=['peak'])
    # Iterate over each dictionary in the list and update the combined dictionary
    for d in res2:
        combined_dict.update(d)
    expr = pd.DataFrame.from_dict(combined_dict,orient='index', columns=['expr'])
    # Iterate over each dictionary in the list and update the combined dictionary
    for d in res3:
        combined_dict.update(d)
    corr = pd.DataFrame.from_dict(combined_dict,orient='index', columns=['corr'])
    tripleDf = pd.concat([peak,expr,corr],axis=1)
  
    return tripleDf

def _concat_results(resDict, key):
    return pd.concat([resDict[result][key] for result in resDict.keys()], axis=1, keys=resDict.keys())

def dict2Mudata(resDict):
    # Concatenate results for 'peak', 'expr', and 'corr'
    peak = _concat_results(resDict, "peak")
    expr = _concat_results(resDict, "expr")
    corr = _concat_results(resDict, "corr")
    exprAdata=sc.AnnData(expr.T)
    peakAdata=sc.AnnData(peak.T)
    corrAdata=sc.AnnData(corr.T)
    exprAdata.layers["raw"] = exprAdata.X
    peakAdata.layers["raw"] = peakAdata.X
    corrAdata.layers["raw"] = corrAdata.X
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

def getAttributeBase(
    adata:AnnData, 
    axis_key, 
    sampleKey: str = "sample",
    subsetCell:List = None,
    subsetGene:List = None, 
    cell_threshold : int = 40, 
    njobs : int = -1,
    **kwargs):
    """Get attribute (peak, expression, correlation) base on axis.

    .. seealso::
        - See :doc:`../../../tutorial/1_OPCST_projecting` for how to
        make trajecoty dotplot.

    Parameters
    ----------
        %(adata)s
        featureKey
            The specific slot in :attr:`adata.var <anndata.AnnData.var>` that houses the gene-feature relationship information
        sampleKey
            The specific slot in :attr:`adata.obs <anndata.AnnData.obs>` that houses the sample information
        patternKey
            Pattern names. If not specific, we use all pattern in adata.obs.featureKey
        cell_threshold
            Minimal cells to keep 
        njobs
            number of cores to use
        
    Returns:
        MuData object which contains correlation, expression, peak modal.

    """
    # simply split by sample
    if subsetCell is None:
        subsetCell=adata.obs_names
    adata.obs["Cell"]=adata.obs_names
    sampleDf = adata.obs.groupby(sampleKey)['Cell'].agg(list).reset_index()
    sampleDict = dict(zip(sampleDf[sampleKey], sampleDf['Cell']))
    dptValue=adata.obs[axis_key]
    timeDict=pd.DataFrame(dptValue).iloc[:,0].to_dict()
    resDict = {}
    for i in tqdm(sampleDict.keys(), desc="Processing Samples"):
        selectCell = sampleDict[i]
        selectCell=np.intersect1d(selectCell,subsetCell)
        loopDf = _process_subset(adata,subsetCell=selectCell,subsetGene=subsetGene,
                                timeDict=timeDict,cell_threshold=cell_threshold,njobs=njob, **kwargs)
        resDict[i] = loopDf
    tvMap = dict2Mudata(resDict)
    return(tvMap)

def getAttributeGEP(
    adata :AnnData,
    featureKey : str = "pattern",
    sampleKey : str = "sample",
    patternKey : List = None,
    cell_threshold : int = 5,
    njobs : int = -1,
    **kwargs
):
    """Get attribute (peak, expression, correlation) base on gene expression pattern axis.

    .. seealso::
        - See :doc:`../../../tutorial/4.15_GEP` for how to
        identified gene expression program associated with differentiated genes.

    Parameters
    ----------
        %(adata)s
        featureKey
            The specific slot in :attr:`adata.var <anndata.AnnData.var>` that houses the gene-feature relationship information
        sampleKey
            The specific slot in :attr:`adata.obs <anndata.AnnData.obs>` that houses the sample information
        patternKey
            Pattern names. If not specific, we use all pattern in adata.obs.featureKey
        cell_threshold
            Minimal cells to keep 
        njobs
            number of cores to use
        
    Returns:
        MuData object. Contains correlation, expression, peak modal.

    """
    varDf = pd.DataFrame(adata.var[featureKey].dropna())
    varDf["GENES"] = varDf.index
    featureDict = varDf.groupby(featureKey)['GENES'].apply(list).to_dict()
    if patternKey == None:
        patternKey = featureDict.keys()
    counter = 0
    for i in patternKey:
        patternLoop = getAttributeBase(adata,axis_key=i,sampleKey=sampleKey,subsetGene=featureDict[i],njob=njobs,**kwargs)
        if counter == 0:
            patternWhole = patternLoop.copy()
        else:
            patternWhole.mod["corr"] = sc.concat([patternWhole["corr"],patternLoop["corr"]],axis=1)
            patternWhole.mod["expr"] = sc.concat([patternWhole["expr"],patternLoop["expr"]],axis=1)
            patternWhole.mod["peak"] = sc.concat([patternWhole["peak"],patternLoop["peak"]],axis=1)
        counter = counter+1
    return(patternWhole)


@d.dedent
def attrTTest(
    adata: AnnData,
    group1Name: List,
    group2Name: List):
    """perform t-test to attribute

    .. seealso::
        - See :doc:`../../../tutorial/4.15_GEP` for how to
        identified gene expression program associated with differentiated genes.

    Parameters
    ----------
        %(adata)s
        group1Name
            Sample name of group1
        group2Name
            Sample name of group2

        
    Returns:
        Dataframe of t-test statics

    """
    corrDf = pd.DataFrame(adata.X)
    corrDf.columns = adata.var_names
    corrDf.index = adata.obs_names
    corrDf=corrDf.loc[:,np.sum(corrDf==0,axis=0)<len(corrDf)]
    group1 = corrDf.loc[group1Name]
    group2 = corrDf.loc[group2Name]
    results=[]
    for col1 in group1.columns:
        statistic, p_value = ttest_rel(group1[col1], group2[col1])
        results.append({'Gene': col1,  'Statistic': statistic, 'P-value': p_value})

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)
    results_df["logFC"] = np.array(np.log(group1.mean()/group2.mean()))
    return(results_df)
