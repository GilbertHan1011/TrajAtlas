from __future__ import annotations
from scipy.stats import binom
import numpy as np
from anndata import AnnData
from mudata import MuData
from rich import print
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
from TrajAtlas.utils._docs import d
from typing import (
    Literal,
)

def _test_binom(length_df,
                times:int = 20):
    # Use binom distribution to test
    sumVal = length_df["true"] + length_df["false"]
    trueVal=length_df["true"]
    null=length_df["null"]
    p_val_list=[]
    for i in range(len(length_df)):
        if null[i]==0:
            null[i]=1/(sumVal[i]*times) # minimal
        p_val= 1- binom.cdf(trueVal[i], sumVal[i], null[i])
        if trueVal[i] == 0:
            p_val=1
        p_val_list.append(p_val)
    length_df["binom_p"]=p_val_list
    return(length_df)
    
def _test_gene_binom(mdata: MuData):
    try:
        sample_adata = mdata["tdiff"]
        pseudobulk=mdata["pseudobulk"]
    except KeyError:
        print(
            "tdiff_mdata should be a MuData object with three slots: feature_key ,'tdiff', and 'pseudobulk' - please run tdiff.count_nhoods(adata) first"
        )
        raise
    times=pseudobulk.uns["shuffle_times"]
    varTable=pseudobulk.varm
    nullTable=varTable["nullPoint"]

    #rateVal=
    sumTable=pseudobulk.uns["sum"]
    trueTable=varTable["truePoint"]
    trueTable = trueTable.fillna(0)
    nullTable = nullTable.fillna(0)
    nullVal = np.where(nullTable == 0, 1 / (sumTable.T * times), nullTable/np.array(sumTable.T * times))
    p_val_array = 1 - binom.cdf(trueTable, sumTable.T, nullVal)
    p_val_array[trueTable == 0] = 1
    p_val_df = pd.DataFrame(p_val_array, index=trueTable.index, columns=trueTable.columns)
    result_df = p_val_df.apply(lambda column: sm.stats.multipletests(column, method='fdr_bh'))
    p_adj_df=result_df.apply(lambda x: x[1])
    p_adj_df.index=p_val_df.index
    p_adj_df.columns=p_val_df.columns.astype("str")
    pseudobulk.varm["gene_p_adj"]=p_adj_df




def _row_scale(row):
    # apply z-score to scale row
    return (row - row.mean()) / row.std()


def _graph_spatial_fdr(
    sample_adata: AnnData,
    neighbors_key: str | None = None,
):
    """FDR correction weighted on inverse of connectivity of neighbourhoods. The distance to the k-th nearest neighbor is used as a measure of connectivity.

    Args:
        sample_adata: Sample-level AnnData.
        neighbors_key: The key in `adata.obsp` to use as KNN graph. Defaults to None.
    """
    # use 1/connectivity as the weighting for the weighted BH adjustment from Cydar
    w = 1 / sample_adata.var["kth_distance"]
    w[np.isinf(w)] = 0

    # Computing a density-weighted q-value.
    pvalues = sample_adata.var["PValue"]
    keep_nhoods = ~pvalues.isna()  # Filtering in case of test on subset of nhoods
    o = pvalues[keep_nhoods].argsort()
    pvalues = pvalues[keep_nhoods][o]
    w = w[keep_nhoods][o]

    adjp = np.zeros(shape=len(o))
    adjp[o] = (sum(w) * pvalues / np.cumsum(w))[::-1].cummin()[::-1]
    adjp = np.array([x if x < 1 else 1 for x in adjp])

    sample_adata.var["SpatialFDR"] = np.nan
    sample_adata.var.loc[keep_nhoods, "SpatialFDR"] = adjp

def _mergeVar(varTable,
            table):
    return(pd.merge(varTable,table,left_index=True,right_index=True,how="left"))


def _test_whole_gene(mdata: MuData):
    try:
        tdiff = mdata["tdiff"]
    except KeyError:
        print(
            "tdiff_mdata should be a MuData object with two slots: feature_key and 'tdiff' - please run tdiff.count_nhoods(adata) first"
        )
        raise
    pseudobulk=mdata["pseudobulk"]
    times=pseudobulk.uns["shuffle_times"]
    varTable=tdiff.varm
    trueVal=np.sum(varTable["Accept"],axis=0)
    sumVal=varTable["Accept"].shape[0]
    nullVal=np.sum(varTable["null_mean"],axis=0)
    
    pval_list=[]
    for i in range(len(nullVal)):
        if(trueVal[i]==0):
            pval_list.append(1)
        else:
            if (nullVal[i]==0):
                nullVal[i]=1/(sumVal*times)
            else:
                    nullVal[i]=nullVal[i]/(sumVal*times)
            pval_list.append(1- binom.cdf(trueVal[i], sumVal, nullVal[i]))
    rejected, adjusted_p_values, _, _ = sm.stats.multipletests(pval_list, method='fdr_bh')
    pseudobulk.var["overall_gene_p"]=np.array(adjusted_p_values)


@d.dedent
def split_gene(
    mdata : MuData,
    mode : Literal["Kmean","Stage"]="Kmean",
    FDR : int = 0.05,
    kmean_cluster : int = 10,
    stage_threshold: int = 10,
    feature_key: str = "rna",
    select_genes:list = None
    ):
    """ Group genes into gene clusters based on their pseudotemporal expression patterns. Currently, we offer two clustering strategies: Kmeans and Stage.
    In `Kmeans` mode, genes are grouped using Kmeans clustering, which identifies differential expression patterns between two groups.
    In `Stage` mode, genes are grouped based on the stage (early or late) at which they exhibit differential expression (up or down).

    .. seealso::
        - See :doc:`../../../tutorial/step3_DE` for how to detect pseudotemporal
        differential genes.

    Parameters
    ----------
    mdata
        MuData object processed by `TrajDiff` process.
    mode
        Gene group strategy, either `Kmean` or `Stage`. If `Kmean` were selected, genes are grouped using Kmeans clustering. If `Stage` were selected
        genes are grouped based on the stage. (default: "Kmean")
    FDR
        False discovery rate to discover significant genes. (default: 0.05)
    kmean_cluster
        If 'Kmeans' is chosen in the mode parameters, you can specify the number of clusters for this parameter. (Default: 10)
    stage_threshold
        If 'Stage' is chosen in the mode parameters, you can specify the threshold of every stage (early/late) to select stage-specific genes. (Default: 10)
    feature_key
        Key to store the cell-level AnnData object in the MuData object. (Default: "rna")
    select_genes
        Custom gene sets are allowed. If not provided, we are using significant genes. (Default: None)


    Returns
    -----------------
    Gene category. Also update MuData in `MuData[feature_key].var`.
    """
    if mode not in ["Kmean", "Stage"]:
            raise ValueError("mode must be one of  'Kmean' 或 'Stage'")
    try:
        tdiff = mdata["tdiff"]
        pseudobulk = mdata["pseudobulk"]  
    except KeyError:
        print("tdiff_mdata should be a MuData object with three slots: feature_key, 'tdiff' and 'pseudobulk' - please run tdiff.count_nhoods(adata) first")
        raise
    try:
        if select_genes == None:
            sigGene = pseudobulk.var_names[pseudobulk.var["overall_gene_p"]<FDR]
            select_genes = sigGene.copy()
        exprMatrix=pseudobulk.varm["exprPoint"].loc[select_genes]
        fdr_matrix=pseudobulk.varm["gene_p_adj"].loc[select_genes]
    except KeyError:
        print("tdiff_mdata should have been run de pipeline. Please run tdiff.de first")
        raise
    if mode=="Kmean":
        exprMatrix=exprMatrix.fillna(0)
        kmeans = KMeans(n_clusters=kmean_cluster)
        kmeans.fit(exprMatrix)
        labels = kmeans.labels_
        labels=pd.DataFrame(labels)
        labels.index=exprMatrix.index
        labels.columns=["geneGroup"]
        mdata[feature_key].var["Kmeans"]=np.nan
        mdata[feature_key].var["Kmeans"].loc[labels.index]=np.array(labels["geneGroup"])
        labels=labels["geneGroup"]
    elif mode=="Stage":
        fdr_matrix=-np.log(fdr_matrix+0.000000001)
        # Assuming exprMatrix and fdrMatrix are numpy arrays
        fdrBinary = fdr_matrix > 1

        exprBinary = exprMatrix * fdrBinary

        start = exprBinary.iloc[:, 0:int(exprBinary.shape[1]/3)]
        end = exprBinary.iloc[:, int(2 * exprBinary.shape[1]/3):exprBinary.shape[1]]
        startSum = np.sum(start, axis=1)
        endSum = np.sum(end, axis=1)
        
        # Define different categories of genes
        geneWholeUp = np.array([name for idx, name in enumerate(startSum.index) if startSum[idx] > stage_threshold and endSum[idx] > stage_threshold])
        geneStartUp = np.array([name for idx, name in enumerate(startSum.index) if startSum[idx] > stage_threshold and abs(endSum[idx]) < stage_threshold])
        geneEndUp = np.array([name for idx, name in enumerate(startSum.index) if abs(startSum[idx]) < stage_threshold and endSum[idx] > stage_threshold])
        geneWholeDown = np.array([name for idx, name in enumerate(startSum.index) if startSum[idx] < -stage_threshold and endSum[idx] < -stage_threshold])
        geneStartdown = np.array([name for idx, name in enumerate(startSum.index) if startSum[idx] < -stage_threshold and abs(endSum[idx]) < stage_threshold])
        geneEndDown = np.array([name for idx, name in enumerate(startSum.index) if abs(startSum[idx]) < stage_threshold and endSum[idx] < -stage_threshold])
        geneUpDown = np.array([name for idx, name in enumerate(startSum.index) if startSum[idx] > stage_threshold and endSum[idx] < -stage_threshold])
        geneDownUp = np.array([name for idx, name in enumerate(startSum.index) if startSum[idx] < -stage_threshold and endSum[idx] > stage_threshold])
        geneGroup = np.concatenate((geneWholeUp, geneStartUp, geneEndUp, geneWholeDown, geneStartdown, geneEndDown, geneUpDown, geneDownUp))
        geneCat = np.concatenate((np.repeat("Up_Up", len(geneWholeUp)),
                                  np.repeat("Up_0", len(geneStartUp)),
                                  np.repeat("0_Up", len(geneEndUp)),
                                  np.repeat("Down_Down", len(geneWholeDown)),
                                  np.repeat("Down_0", len(geneStartdown)),
                                  np.repeat("0_Down", len(geneEndDown)),
                                  np.repeat("Up_Down", len(geneUpDown)),
                                  np.repeat("Down_Up", len(geneDownUp))))


        geneSplit = pd.DataFrame({"gene": geneGroup, "geneGroup": geneCat})
        geneSplit['geneGroup'] = pd.Categorical(geneSplit['geneGroup'])
        geneSplit.index=geneSplit["gene"]
        labels=geneSplit["geneGroup"]
        mdata[feature_key].var["Stage"]=np.nan
        mdata[feature_key].var["Stage"].loc[labels.index]=np.array(labels)
    return(labels)
    