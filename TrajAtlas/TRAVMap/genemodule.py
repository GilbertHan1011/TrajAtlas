from __future__ import annotations
import pandas as pd
import PyComplexHeatmap as pch
import numpy as np
from sklearn.decomposition import NMF
from mudata import MuData
from TrajAtlas.TrajDiff.trajdiff_utils import _row_scale

try:
    from rpy2.robjects import conversion, numpy2ri, pandas2ri
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import STAP, PackageNotInstalledError, importr
except ModuleNotFoundError:
    print(
        "[bold yellow]rpy2 is not installed. Install with [green]pip install rpy2 [yellow]to run tools with R support."
    )

def _setup_RcppML(
):
    """Set up rpy2 to run edgeR"""
    numpy2ri.activate()
    pandas2ri.activate()
    RcppML = _try_import_bioc_library("RcppML")

    return RcppML

def _try_import_bioc_library(
    name: str,
):
    """Import R packages.

    Args:
        name (str): R packages name
    """
    try:
        _r_lib = importr(name)
        return _r_lib
    except PackageNotInstalledError:
        print(f"Install Bioconductor library `{name!r}` first as `BiocManager::install({name!r}).`")
        raise
        
def _detect_RcppML():
    """Import R packages.

    Args:
        name (str): R packages name
    """
    try:
        _r_lib = importr("RcppML")
        return True
    except PackageNotInstalledError:
        return False

def find_gene_module(mdata: MuData,
                    varGene: str | None=None,
                    bin_threshold: int |None=30,
                    gene_threshold:int | None=1000,
                     n_factors: int | None=15
                    ):
    RcppML= _setup_RcppML()
    keys_to_delete = []
    if varGene==None:
        varGene=pd.read_csv("../../../../important_processed_data/NMF_varGene.csv",index_col=0)
        varGene=varGene["x"]
    
    factorDict={}
    RcppDetect=_detect_RcppML()
    cpmDict=mdata["tdiff"].uns["cpm_dict"].copy()
    for df_name, df in cpmDict.items():
        print(f"Detecting NMF factors in {df_name}....")
        # preprocessing
        intersectGene=np.intersect1d(df.index,varGene)
        df=df.loc[intersectGene]
        df = df.fillna(0)
        df = df.loc[:, (df != 0).any(axis=0)]  # Remove columns with all zeros
        df = df.loc[(df != 0).any(axis=1), :]
        cpmDict[df_name] = df.apply(_row_scale, axis=1)
        if (df.shape[1] < bin_threshold) | (df.shape[0]< gene_threshold) :
            print(f"{df_name} doesn't seem like a trajectory. Removing.....")
            keys_to_delete.append(df_name)
        else:
            # factorization
            if RcppDetect:
                print("Using RcppML for decomposition.....")
                model_test = RcppML.nmf(df.T, n_factors, verbose=False, seed=1234)
                # Extract h matrix from model_test
                h = model_test.rx2('h')
            else:
                print("RcppML was not detected. Using Sklearn's NMF for decomposition.....")
                model_test = NMF(n_components=n_comp, init='random', random_state=1234)
                h = model_test.components_
            h=pd.DataFrame(h)
            # Rename the columns of h to match the row names of x
            h_colnames = df.index.tolist()
            h.columns=h_colnames
            h_index= "NMF_"+h.index.astype("str")
            h.index=h_index
            # Filter rows in h where the row sum is not equal to 0
            filterH = h.sum(axis=1) != 0
            h_filtered = h[filterH]
            factorDict[df_name]=h_filtered.T

    # Delete items outside the loop
    for key in keys_to_delete:
        del cpmDict[key]
        del factorDict[key]
    mdata['tdiff'].uns["cpmDict"]=cpmDict
    mdata['tdiff'].uns["factor_dict"]=factorDict


def plotGeneModule(
                   mdata:Mudata,
                   sample:str, 
                   factor:str, 
                   gene_num: int| None = 20,
                   **kwargs):
    expDf=mdata["tdiff"].uns["cpmDict"][sample]
    factorDf=mdata["tdiff"].uns["factor_dict"][sample]
    # Sort the DataFrame based on column 'A'
    factorDf = factorDf.sort_values(by=factor,ascending=False)
    # Get the row names (index) of the sorted DataFrame
    geneModule = factorDf.index.tolist()[0:gene_num]
    pseudotimeCol=expDf.columns.astype("int")
    pseudotimeDf=pd.DataFrame(pseudotimeCol)
    pseudotimeDf.index=expDf.columns
    col_ha = pch.HeatmapAnnotation(Pseudotime=pch.anno_simple(pseudotimeDf[0],cmap='jet',
                                    add_text=False,text_kws={'color':'black','rotation':-90,'fontweight':'bold','fontsize':10,},
                                    legend=True),
                    verbose=0,label_side='left',label_kws={'horizontalalignment':'right'})
    pch.ClusterMapPlotter(expDf.loc[geneModule], row_cluster=False, 
                          col_cluster=False,cmap="RdBu_r",show_rownames=True,top_annotation=col_ha,**kwargs)