from __future__ import annotations
import pandas as pd
import PyComplexHeatmap as pch
import numpy as np
from sklearn.decomposition import NMF
from mudata import MuData
from TrajAtlas.TrajDiff.trajdiff_utils import _row_scale
from TrajAtlas.utils._env import _setup_RcppML, _try_import_bioc_library, _detect_RcppML
from anndata import AnnData
import os

try:
    from rpy2.robjects import conversion, numpy2ri, pandas2ri
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import STAP, PackageNotInstalledError, importr
except ModuleNotFoundError:
    print(
        "[bold yellow]rpy2 is not installed. Install with [green]pip install rpy2 [yellow]to run tools with R support."
    )

location = os.path.dirname(os.path.realpath(__file__))
NMFfilePath = os.path.join(location, '..','datasets', "NMF_varGene.csv")

from TrajAtlas.TrajDiff.trajDiff import Tdiff
tdiff = Tdiff()

def getTrajExpression(data: MuData | AnnData,
                      subsetLineage: str = None,
                      run_milo=True,
                      run_pseudobulk=True,
                      feature_key: str="rna",
                      n_interval: int=100,
                      milo_nhood_prop:float = 0.1,
                      sample_col: str = None,
                      group_col: str = None,
                      time_col: str = None,
                      njob: int = -1,
                      min_cell: int =4,
                     ):
    if isinstance(data, MuData):
        adata = data[feature_key]
        mdata = data
    if isinstance(data, AnnData):
        adata = data
        mdata = tdiff.load(adata)
    if subsetLineage != None:
        adata=adata[adata.obs[subsetLineage]]
        run_milo=True
        run_pseudobulk=True
    if sample_col == None:
        try:
            sample_col = mdata["tdiff"].uns["sample_col"]
        except KeyError:
            print('Please specify sample_col parameter first')
            raise
    if sample_col == None:
        try:
            sample_col = mdata["tdiff"].uns["sample_col"]
        except KeyError:
            print('Please specify sample_col parameter first')
            raise
    if group_col == None:
        try:
            group_col = mdata["tdiff"].uns["group_col"]
        except KeyError:
            print('Please specify group_col parameter first')
            raise
    if time_col == None:
        try:
            time_col = mdata["tdiff"].uns["time_col"]
        except KeyError:
            print('Please specify time_col parameter first')
            raise
    if run_milo == True:
        tdiff.make_nhoods(mdata['rna'], prop=milo_nhood_prop)
        mdata =  tdiff.count_nhoods(mdata, sample_col=sample_col)
    if run_pseudobulk == True:
        pseudobulk=tdiff.make_pseudobulk_parallel(mdata=mdata,sample_col=sample_col,
                                                  group_col=group_col,time_col=time_col,njob=njob,min_cell=min_cell)
    wholeCpm=tdiff.make_whole_cpm(mdata)
    tdiff._make_range(mdata,only_range=True)
    tdiff.permute_point_cpm_parallel(mdata)
    return(mdata)


def find_gene_module(mdata: MuData,
                    varGene: str | None=None,
                    bin_threshold: int |None=30,
                    gene_threshold:int | None=1000,
                    n_factors: int | None=15
                    ):
    RcppML= _setup_RcppML()
    keys_to_delete = []
    if varGene==None:
        varGene=pd.read_csv(NMFfilePath,index_col=0)
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