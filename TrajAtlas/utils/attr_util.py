from __future__ import annotations
from joblib import Parallel, delayed
from functools import partial
from tqdm import tqdm
from scipy.stats import pearsonr, ttest_rel
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
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning

def _process_gene(chunk, geneMat, gene_agg, varName, timeVal):
    pearsonCorrDict = {}
    maxRowsDict = {}
    sumValuesDict = {}

    for k in chunk:
        geneArr = geneMat.iloc[:, k]
        geneAggArr = gene_agg.iloc[:, k]
        geneName = varName[k]

        if geneAggArr.sum() == 0:
            maxRowsDict[geneName] = 0
            sumValuesDict[geneName] = 0
            pearsonCorrDict[geneName] = 0
        else:
            pearson, _ = pearsonr(geneArr, np.array(timeVal))
            pearsonCorrDict[geneName] = pearson
            maxRowsDict[geneName] = geneAggArr.idxmax()
            sumValuesDict[geneName] = geneAggArr.sum()

    return [maxRowsDict, sumValuesDict, pearsonCorrDict]


def _process_subset(adata, timeDict, subsetCell: list = None, subsetGene: List = None, num_bins = 10, 
                    cell_threshold = 40, njobs = -1):
    if subsetGene is None:
        subsetGene = adata.var_names
    if subsetCell is None:
        subsetCell = adata.obs_names

    if len(subsetCell) < cell_threshold:
        return None

    subsetAdata = adata[subsetCell, subsetGene]
    dptValue = np.array(list(timeDict.values()))
    hist, bin_edges = np.histogram(dptValue, bins = num_bins)
    timeBin = np.digitize(dptValue, bin_edges)
    timeBin = pd.DataFrame(timeBin, index = np.array(list(timeDict.keys())))
    timeBin[timeBin == 11] = 10
    timeBin = timeBin.squeeze().to_dict()

    timeVal = [timeDict[val] for val in subsetCell]
    timeBinVal = [timeBin[val] for val in subsetCell]

    geneMat = pd.DataFrame(subsetAdata.X.toarray(), columns = subsetAdata.var_names, index = subsetAdata.obs_names)
    geneMat["dpt_bin"] = np.array(timeBinVal)
    gene_agg = geneMat.groupby("dpt_bin").agg({gene: "mean" for gene in subsetGene})
    bin_mask = geneMat.groupby("dpt_bin").size() < 5
    gene_agg[bin_mask] = np.nan
    geneMat = geneMat.loc[:, subsetAdata.var_names]

    chunk_size = geneMat.shape[1] // (100 if njobs == -1 else njobs)
    chunks = [range(i, min(i + chunk_size, geneMat.shape[1])) for i in range(0, geneMat.shape[1], chunk_size)]

    partial_process_gene = partial(_process_gene, geneMat = geneMat, gene_agg = gene_agg, varName = subsetAdata.var_names, timeVal = timeVal)
    results = Parallel(n_jobs = njobs)(delayed(partial_process_gene)(chunk) for chunk in chunks)

    res1 = [result[0] for result in results if result is not None]
    res2 = [result[1] for result in results if result is not None]
    res3 = [result[2] for result in results if result is not None]

    combined_dict = {}
    for d in res1:
        combined_dict.update(d)
    peak = pd.DataFrame.from_dict(combined_dict, orient = 'index', columns = ['peak'])

    combined_dict = {}
    for d in res2:
        combined_dict.update(d)
    expr = pd.DataFrame.from_dict(combined_dict, orient = 'index', columns = ['expr'])

    combined_dict = {}
    for d in res3:
        combined_dict.update(d)
    corr = pd.DataFrame.from_dict(combined_dict, orient = 'index', columns = ['corr'])

    return pd.concat([peak, expr, corr], axis = 1)

def _concat_results(resDict, key):
    if isinstance(next(iter(resDict.values())), list):
        # If resDict contains lists of DataFrames, concatenate each list separately
        concatenated = []
        for sample, dfs in resDict.items():
            concatenated.append(pd.concat([df[key] for df in dfs], axis=1, keys=range(len(dfs))))
        return pd.concat(concatenated, axis=1, keys=resDict.keys())
    else:
        return pd.concat([resDict[result][key] for result in resDict.keys()], axis=1, keys=resDict.keys())

def flatten_multiindex(df):
    df.columns = ['_'.join(map(str, col)) for col in df.columns]
    return df

def _dict2Mudata(resDict, bootstrap_iterations):
    peak = _concat_results(resDict, "peak")
    expr = _concat_results(resDict, "expr")
    corr = _concat_results(resDict, "corr")

    # Flatten MultiIndex
    peak = flatten_multiindex(peak)
    expr = flatten_multiindex(expr)
    corr = flatten_multiindex(corr)

    exprAdata = sc.AnnData(expr.T)
    peakAdata = sc.AnnData(peak.T)
    corrAdata = sc.AnnData(corr.T)

    exprAdata.layers["raw"] = exprAdata.X
    peakAdata.layers["raw"] = peakAdata.X
    corrAdata.layers["raw"] = corrAdata.X

    corr_mod = pd.DataFrame(np.where(corr >= 0, np.sqrt(corr), -np.sqrt(-corr)), columns=corr.columns, index=corr.index)
    expr_mod = expr.apply(lambda row: row / row.max(), axis=1)

    exprAdata.layers["mod"] = expr_mod.T
    peakAdata.layers["mod"] = peak.T
    corrAdata.layers["mod"] = corr_mod.T

    # Add two obs columns to all AnnData objects
    if bootstrap_iterations > 0:
        for adata in [peakAdata, corrAdata, exprAdata]:
            adata.obs['sample'] = [col.rsplit('_', 1)[0] for col in adata.obs_names]
            adata.obs['bootstrap'] = [col.rsplit('_', 1)[1] for col in adata.obs_names]
    else:
        for adata in [peakAdata, corrAdata, exprAdata]:
            adata.obs['sample'] = adata.obs_names

    return mu.MuData({"corr": corrAdata, "expr": exprAdata, "peak": peakAdata})


def getAttributeBase(
    adata: AnnData, 
    axis_key, 
    sampleKey: str = "sample",
    subsetCell: List = None,
    subsetGene: List = None, 
    cell_threshold: int = 40, 
    njobs: int = -1,
    bootstrap_iterations: int = 0,
    **kwargs
):
    """Get attribute (peak, expression, correlation) base on axis.

    .. seealso::
        - See :doc:`../../../tutorial/1_OPCST_projecting` for how to make trajecoty dotplot.

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
        bootstrap_iterations
            number of bootstrap iterations to perform. If 0, bootstrapping is not performed.

        
    Returns:
        MuData object which contains correlation, expression, peak modal.
    """
    if subsetCell is None:
        subsetCell = adata.obs_names

    adata.obs["Cell"] = adata.obs_names
    sampleDf = adata.obs.groupby(sampleKey)['Cell'].agg(list).reset_index()
    sampleDict = dict(zip(sampleDf[sampleKey], sampleDf['Cell']))
    timeDict = adata.obs[axis_key].to_dict()

    resDict = {}
    for sample in tqdm(sampleDict.keys(), desc = "Processing Samples"):
        selectCell = np.intersect1d(sampleDict[sample], subsetCell)
        
        if bootstrap_iterations == 0:
            loopDf = _process_subset(adata, subsetCell = selectCell, subsetGene = subsetGene,
                                     timeDict = timeDict, cell_threshold = cell_threshold,
                                      njobs = njobs, **kwargs)
            resDict[sample] = loopDf
        else:
            bootstrap_results = []
            for b in range(bootstrap_iterations):
                resampled_indices = np.random.choice(selectCell, size=len(selectCell), replace=True)
                resampled_adata = adata[resampled_indices].copy()
                resampled_adata.obs_names = [f"{name}_{j}" for j, name in enumerate(resampled_adata.obs_names)]
                timeDict = resampled_adata.obs[axis_key].to_dict()
                loopDf = _process_subset(resampled_adata, subsetCell = resampled_adata.obs_names, subsetGene = subsetGene,
                                         timeDict = timeDict, cell_threshold = cell_threshold, njobs = njobs, **kwargs)
                bootstrap_results.append(loopDf)
            
            resDict[sample] = bootstrap_results

    return _dict2Mudata(resDict, bootstrap_iterations = bootstrap_iterations)



@d.dedent
def getAttributeGEP(
    adata: AnnData,
    featureKey: str = "pattern",
    sampleKey: str = "sample",
    patternKey: List = None,
    cell_threshold: int = 5,
    njobs: int = -1,
    bootstrap_iterations: int = 0,
    **kwargs
):
    """Get attribute (peak, expression, correlation) based on gene expression pattern axis with optional bootstrapping.

    Parameters
    ----------
        %(adata)s
        featureKey
            The specific slot in :attr:`adata.var <anndata.AnnData.var>` that houses the gene-feature relationship information
        sampleKey
            The specific slot in :attr:`adata.obs <anndata.AnnData.obs>` that houses the sample information
        patternKey
            Pattern names. If not specific, we use all patterns in adata.obs.featureKey
        cell_threshold
            Minimal cells to keep 
        njobs
            Number of cores to use
        bootstrap_iterations
            Number of bootstrap iterations to perform. If 0, bootstrapping is not performed.

    Returns:
        MuData object. Contains correlation, expression, and peak modal.
    """
    varDf = pd.DataFrame(adata.var[featureKey].dropna())
    varDf["GENES"] = varDf.index
    featureDict = varDf.groupby(featureKey)['GENES'].apply(list).to_dict()

    if patternKey is None:
        patternKey = featureDict.keys()

    patternWhole = None
    for counter, pattern in enumerate(patternKey):
        #print("Processing pattern: ", pattern)
        patternLoop = getAttributeBase(
            adata,
            axis_key = pattern,
            sampleKey = sampleKey,
            subsetGene = featureDict[pattern],
            njobs = njobs,
            bootstrap_iterations = bootstrap_iterations,
            **kwargs
        )
        #print(patternLoop)
        if counter == 0:
            patternWhole = patternLoop.copy()
        else:
            patternWhole.mod["corr"] = sc.concat([patternWhole["corr"], patternLoop["corr"]], axis = 1,join="outer", merge="same")
            patternWhole.mod["expr"] = sc.concat([patternWhole["expr"], patternLoop["expr"]], axis = 1,join="outer", merge="same")
            patternWhole.mod["peak"] = sc.concat([patternWhole["peak"], patternLoop["peak"]], axis = 1,join="outer", merge="same")

    return patternWhole


@d.dedent
def attrTTest(
    adata: AnnData,
    group1Name: List,
    group2Name: List,
    alpha: float = 0.05,
    method: str = 'fdr_bh'  # Method for multiple testing correction
):
    """Perform t-test to attribute with FDR correction.

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
        alpha
            Significance level for FDR correction.
        method
            Method for multiple testing correction. Default is 'fdr_bh'.

    Returns:
        Dataframe of t-test statistics with FDR-corrected p-values.
    """
    corrDf = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs_names)
    corrDf = corrDf.loc[:, np.sum(corrDf == 0, axis=0) < len(corrDf)]

    group1 = corrDf.loc[group1Name]
    group2 = corrDf.loc[group2Name]

    results = []
    for col in group1.columns:
        statistic, p_value = ttest_rel(group1[col], group2[col])
        results.append({'Gene': col, 'Statistic': statistic, 'P-value': p_value})

    results_df = pd.DataFrame(results)
    results_df["logFC"] = np.log(group1.mean() / group2.mean())

    # Perform FDR correction
    p_values = results_df['P-value'].values
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method=method)
    results_df['P-value_corrected'] = pvals_corrected
    results_df['Significant'] = reject

    return results_df

@d.dedent
def attrANOVA(
    adata: AnnData,
    group_labels: List,
    alpha: float = 0.05,
    method: str = 'fdr_bh'  # Method for multiple testing correction
):
    """Perform one-way ANOVA to test for differences across multiple groups with FDR correction.

    Parameters
    ----------
        %(adata)s
        group_labels
            A list where each element corresponds to the group label for each sample in `adata.obs_names`.
        alpha
            Significance level for FDR correction.
        method
            Method for multiple testing correction. Default is 'fdr_bh'.

    Returns:
        DataFrame containing ANOVA F-statistics and FDR-corrected p-values for each feature.
    """
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=ValueWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    corr_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    corr_df['group'] = group_labels

    valid_feature_names = {}
    for feature in corr_df.columns[:-1]:  # Exclude 'group' column
        valid_name = feature.replace(' ', '_')
        if not valid_name[0].isalpha() and valid_name[0] != '_':
            valid_name = f'gene_{valid_name}'
        valid_name = valid_name.replace('-', '_').replace('.', '_')
        valid_feature_names[feature] = valid_name

    corr_df.rename(columns=valid_feature_names, inplace=True)

    results = []
    for feature in corr_df.columns[:-1]:  # Exclude the 'group' column
        try:
            formula = f'Q("{feature}") ~ C(group)'
            model = ols(formula, data=corr_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            f_stat = anova_table['F'].iloc[0]
            p_value = anova_table['PR(>F)'].iloc[0]
            results.append({'Feature': feature, 'F-statistic': f_stat, 'P-value': p_value})
        except Exception as e:
            print(f"Error processing feature {feature}: {e}")

    results_df = pd.DataFrame(results)

    # Perform FDR correction
    p_values = results_df['P-value'].values
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method=method)
    results_df['P-value_corrected'] = pvals_corrected
    results_df['Significant'] = reject

    return results_df

@d.dedent
def attrTwoWayANOVA(
    adata: AnnData,
    factor1_labels: List,
    factor2_labels: List,
    interaction: bool = True
):
    """Perform two-way ANOVA with optional interaction term.

    Parameters
    ----------
        %(adata)s
        factor1_labels
            Labels for the first factor.
        factor2_labels
            Labels for the second factor.
        interaction
            Include interaction term in the model.

    Returns:
        DataFrame containing ANOVA F-statistics and P-values for each feature.
    """
    warnings.filterwarnings("ignore", category=ValueWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    corr_df = pd.DataFrame(adata.X, index = adata.obs_names, columns = adata.var_names)
    corr_df['factor1'] = factor1_labels
    corr_df['factor2'] = factor2_labels

    valid_feature_names = {}
    for feature in corr_df.columns[:-2]:  # Exclude 'factor1' and 'factor2' columns
        valid_name = feature.replace(' ', '_')
        if not valid_name[0].isalpha() and valid_name[0] != '_':
            valid_name = f'gene_{valid_name}'
        valid_name = valid_name.replace('-', '_').replace('.', '_')
        valid_feature_names[feature] = valid_name

    corr_df.rename(columns = valid_feature_names, inplace = True)

    results = []
    for feature in corr_df.columns[:-2]:  # Exclude 'factor1' and 'factor2' columns
        formula = f'{feature} ~ C(factor1) {"* C(factor2)" if interaction else "+ C(factor2)"}'
        model = ols(formula, data = corr_df).fit()
        anova_table = sm.stats.anova_lm(model, typ = 2)

        f_stat1 = anova_table['F']['C(factor1)']
        p_value1 = anova_table['PR(>F)']['C(factor1)']
        f_stat2 = anova_table['F']['C(factor2)']
        p_value2 = anova_table['PR(>F)']['C(factor2)']
        f_stat_inter = anova_table['F'].get('C(factor1):C(factor2)', None)
        p_value_inter = anova_table['PR(>F)'].get('C(factor1):C(factor2)', None)

        results.append({
            'Feature': feature,
            'F-factor1': f_stat1,
            'P-factor1': p_value1,
            'F-factor2': f_stat2,
            'P-factor2': p_value2,
            'F-interaction': f_stat_inter,
            'P-interaction': p_value_inter
        })

    return pd.DataFrame(results)