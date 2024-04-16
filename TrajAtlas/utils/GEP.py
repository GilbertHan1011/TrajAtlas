from __future__ import annotations
from joblib import Parallel, delayed
from functools import partial
from tqdm import tqdm
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import muon as mu
import scanpy as sc
from mudata import MuData
from TrajAtlas.utils._docs import d
import PyComplexHeatmap as pch
from anndata import AnnData
from typing import List
from scipy.stats import ttest_rel

def _process_subset(parrtern,adata, featureDict, cell_threshold=5):
    features = np.intersect1d(featureDict[parrtern], adata.var_names)
    adataSubset = adata[:,features]
    
    dptValue = adataSubset.obs[parrtern]
    num_bins = 10
    hist, bin_edges = np.histogram(dptValue, bins=num_bins)
    timeBin=np.digitize(dptValue, bin_edges)
    timeBin=pd.DataFrame(timeBin)
    timeBin.index=dptValue.index
    timeBin[timeBin==11]=10
    
    agg_dict = {gene: "mean" for gene in adataSubset.var_names}
    timeVal = dptValue
    geneMat = pd.DataFrame(adataSubset.X.toarray())
    varName = adataSubset.var_names
    geneMat.columns=adataSubset.var_names
    geneMat.index=adataSubset.obs_names
    geneMat["dpt_bin"]=np.array(timeBin)
    gene_agg = geneMat.groupby("dpt_bin").agg(agg_dict)
    bin_mask=geneMat.groupby("dpt_bin").size()<cell_threshold
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
    pearsonCorrDf.columns = ["correlation"]
    maxRowDf = pd.DataFrame.from_dict(maxRowsDict, orient="index").fillna(0)
    maxRowDf.columns = ["peak"]
    sumValDf = pd.DataFrame.from_dict(sumValuesDict, orient="index").fillna(0)
    sumValDf.columns = ["expr"]

    return pearsonCorrDf, maxRowDf, sumValDf

def _getAttributeFun(adata,featureDict,patternKey,cell_threshold=5,njobs=-1):
    patternList=patternKey
    partial_process_subset = partial(_process_subset, adata=adata, featureDict=featureDict,cell_threshold=cell_threshold)
    results = Parallel(n_jobs=njobs)(delayed(partial_process_subset)(pattern) for pattern in tqdm(patternList))
    corrDf = pd.concat([r[0] for r in results])
    peakDf = pd.concat([r[1] for r in results])
    exprDf = pd.concat([r[2] for r in results])
    concatDf=pd.concat([corrDf,peakDf,exprDf],axis=1)
    return(concatDf)

@d.dedent
def getAttributeGEP(
    adata : AnnData,
    featureKey : str = "pattern",
    sampleKey : str = "sample",
    patternKey : List = None,
    cell_threshold : int = 5,
    njobs : int = -1):
    """Get gene expression pattern (GEP) based attribute

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
    concatDf= _getAttributeFun(adata = adata,featureDict=featureDict,patternKey = patternKey, cell_threshold=cell_threshold,njobs=njobs)
    resDict={}
    for i in set(adata.obs[sampleKey]):
        scSubset=adata[adata.obs[sampleKey]==i].copy()
        res= _getAttributeFun(adata = scSubset,featureDict=featureDict,patternKey = patternKey, cell_threshold=cell_threshold,njobs=njobs)
        resDict[i]= res     

    resDictCorr={}
    resDictPeak={}
    resDictExpr={}
    for i in resDict.keys():
        resDictCorr[i] = resDict[i]["correlation"]
        resDictPeak[i] = resDict[i]["peak"]
        resDictExpr[i] = resDict[i]["expr"]
        
    corr=pd.DataFrame(resDictCorr)
    expr=pd.DataFrame(resDictExpr)
    peak=pd.DataFrame(resDictPeak)

    exprAdata=sc.AnnData(expr.T)
    peakAdata=sc.AnnData(peak.T)
    corrAdata=sc.AnnData(corr.T)
    exprAdata.layers["raw"]=exprAdata.X
    peakAdata.layers["raw"]=peakAdata.X
    corrAdata.layers["raw"]=corrAdata.X
    corr_mod = np.where(corr >= 0, np.sqrt(corr), -np.sqrt(-corr))
    corr_mod=pd.DataFrame(corr_mod)
    corr_mod.columns=corr_mod.columns
    corr_mod.index=corr_mod.index
    expr_mod = expr.apply(lambda row: (row) / (row.max()), axis=1)
    exprAdata.layers["mod"]=expr_mod.T  
    peakAdata.layers["mod"]=peak.T
    corrAdata.layers["mod"]=corr_mod.T
    tvmap=mu.MuData({"corr":corrAdata, "expr": exprAdata,"peak":peakAdata})
    return(tvmap)

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

