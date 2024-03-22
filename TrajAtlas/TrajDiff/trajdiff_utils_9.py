from __future__ import annotations
from scipy.stats import binom
import numpy as np
from anndata import AnnData
import pandas as pd
import statsmodels.api as sm



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
    
def _test_gene_binom(mdata: Mudata):
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