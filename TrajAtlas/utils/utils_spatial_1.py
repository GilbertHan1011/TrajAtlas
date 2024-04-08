from joblib import Parallel, delayed
from functools import partial
from tqdm import tqdm
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import muon as mu
import scanpy as sc
import PyComplexHeatmap as pch


def process_subset(parrtern,adata, featuredict, cell_threshold=5):
    features = np.intersect1d(featuredict[parrtern], adata.var_names)
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

def getAttribute(adata,featuredict,cell_threshold=5,njobs=-1):
    patternList=['pattern0', 'pattern1', 'pattern2', 'pattern3', 'pattern4', 'pattern5', 'pattern6', 'pattern7']
    partial_process_subset = partial(process_subset, adata=adata, featuredict=featuredict,cell_threshold=cell_threshold)
    results = Parallel(n_jobs=njobs)(delayed(partial_process_subset)(pattern) for pattern in tqdm(patternList))
    corrDf = pd.concat([r[0] for r in results])
    peakDf = pd.concat([r[1] for r in results])
    exprDf = pd.concat([r[2] for r in results])
    concatDf=pd.concat([corrDf,peakDf,exprDf],axis=1)
    return(concatDf)