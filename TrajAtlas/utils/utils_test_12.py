from joblib import Parallel, delayed
from functools import partial
from tqdm import tqdm
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import muon as mu
import scanpy as sc
import PyComplexHeatmap as pch



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
    pearsonCoorDict = {}
    maxRowsDict={}
    sumValuesDict={}
    for k in range(geneMat.shape[1]):
        geneArr = geneMat.iloc[:, k]
        geneAggArr=gene_agg.iloc[:,k]
        if geneAggArr.sum()== 0:
            geneName = varName[k]
            maxRowsDict[geneName]=0
            sumValuesDict[geneName]=0
            pearsonCoorDict[geneName]=0
        else:
            pearson, _ = pearsonr(geneArr, np.array(timeVal))
            geneName = varName[k]
            pearsonCoorDict[geneName] = pearson
            max_row = geneAggArr.idxmax()
            maxRowsDict[geneName] = max_row
            sumValuesDict[geneName]=geneAggArr.sum()
    
    pearsonCoorDf = pd.DataFrame.from_dict(pearsonCoorDict, orient="index").fillna(0)
    pearsonCoorDf.columns = [i + "_sep_" + j]
    maxRowDf = pd.DataFrame.from_dict(maxRowsDict, orient="index").fillna(0)
    maxRowDf.columns = [i + "_sep_" + j]
    sumValDf = pd.DataFrame.from_dict(sumValuesDict, orient="index").fillna(0)
    sumValDf.columns = [i + "_sep_" + j]
  
    return pearsonCoorDf, maxRowDf, sumValDf

def getAttribute(adata,lineage: list or None = ["Fibroblast", "LepR_BMSC", "MSC", "Chondro"],
                 peudotime_key="pseduoPred",n_jobs=-1,cell_threshold: int or None=40):
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
    results = Parallel(n_jobs=n_jobs)(delayed(partial_process_subset)(*key_pair) for key_pair in tqdm(key_pairs))
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
    coorAdata=sc.AnnData(corr.T)
    coor_mod = np.where(corr >= 0, np.sqrt(corr), -np.sqrt(-corr))
    coor_mod=pd.DataFrame(coor_mod)
    coor_mod.columns=coor_mod.columns
    coor_mod.index=coor_mod.index
    expr_mod = expr.apply(lambda row: (row) / (row.max()), axis=1)
    exprAdata=sc.AnnData(expr.T)
    peakAdata=sc.AnnData(peak.T)
    coorAdata=sc.AnnData(corr.T)
    exprAdata.layers["mod"]=expr_mod.T
    peakAdata.layers["mod"]=peak.T
    coorAdata.layers["mod"]=coor_mod.T
    tvmap=mu.MuData({"corr":coorAdata, "expr": exprAdata,"peak":peakAdata})
    return(tvmap)

def makeDotTable(tvMap,gene,sample):
    exprMod=tvMap["expr"].layers["mod"]
    peakMod=tvMap["peak"].layers["mod"]
    corrMod=tvMap["coor"].layers["mod"]
    exprMod=pd.DataFrame(tvMap["expr"].layers["mod"],index=tvMap.obs_names,columns=tvMap["expr"].var_names).T
    peakMod=pd.DataFrame(tvMap["peak"].layers["mod"],index=tvMap.obs_names,columns=tvMap["peak"].var_names).T
    corrMod=pd.DataFrame(tvMap["coor"].layers["mod"],index=tvMap.obs_names,columns=tvMap["coor"].var_names).T
    selectCoor=corrMod.loc[gene]
    selectExpr=exprMod.loc[gene]
    selectPeak=peakMod.loc[gene]
    selectCoor=selectCoor.loc[:,sample]
    selectExpr=selectExpr.loc[:,sample]
    selectPeak=selectPeak.loc[:,sample]
    coorLong=selectCoor.stack().reset_index(name="Corr")
    exprLong=selectExpr.stack().reset_index(name="Expr")
    peakLong=selectPeak.stack().reset_index(name="Peak")
    peakLong['Stage']=peakLong.Peak.apply(lambda x:'End' if x>=7 else 'Middle' if x >= 3 else 'Start')
    combineDf=coorLong
    combineDf["Expr"]=exprLong['Expr']
    combineDf["Peak"]=peakLong['Stage']
    return(combineDf)
    
def trajDotplot(geneTb,col_split:int or pd.Series or pd.DataFrame or None, ratio=60, 
                show_rownames=True,show_colnames=False,spines=True,**kwargs):
    cm = pch.DotClustermapPlotter(geneTb,x='level_1',y='level_0',value='Corr',c='Corr',hue='Peak',s="Expr",
                               marker={'Start':'o','Middle':'D','End':'s'},col_split=col_split,
                                 ratio=ratio,show_rownames=show_rownames,
                                  spines=spines,show_colnames=show_colnames,**kwargs)