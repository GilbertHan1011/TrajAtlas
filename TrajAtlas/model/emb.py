import pandas as pd
import numpy as np
import muon as mu
import scanpy as sc



def MergeTRAVMap(tvmap1 : MuData,
             tvmap2 : MuData,
             include_trav : bool = False):
    corr1=tvmap1["corr"]
    expr1=tvmap1["expr"]
    peak1=tvmap1["peak"]
    corr2=tvmap2["corr"]
    expr2=tvmap2["expr"]
    peak2=tvmap2["peak"]
    
    corrConbined=sc.concat([corr1,corr2])
    exprConbined=sc.concat([expr1,expr2])
    peakConbined=sc.concat([peak1,peak2])
    if include_trav:
        trav1=tvmap1["TRAV"]
        trav2=tvmap2["TRAV"]
        travConbined=sc.concat([trav1,trav2])
        tvmapConbined=mu.MuData({"corr":corrConbined, "expr": exprConbined,
                                 "peak":peakConbined,"TRAV":travConbined})
    else:
        tvmapConbined=mu.MuData({"corr":corrConbined, "expr": exprConbined,
                                 "peak":peakConbined})
        
    obs1 = tvmap1.obs
    obs1["old_or_new"]="old"
    obs2 = tvmap2.obs
    obs2["old_or_new"]="new"
    obsConcat = pd.concat([obs1,obs2])
    tvmapConbined.obs=obsConcat

    
    return(tvmapConbined)

def processTRAV(
    TRAVMap: MuData,
    include_trav : bool = False,
    pca_comps: int = 10,
    n_multineighbors:int = 20
               ):
    corr=TRAVMap["corr"]
    corr.layers["raw"] = corr.X
    corr.X = corr.layers["mod"]
    expr=TRAVMap["expr"]
    expr.layers["raw"] = expr.X
    expr.X=np.log1p(expr.X)
    
    #sc.pp.normalize_total(expr, target_sum=1e6)
    
    peak=TRAVMap["peak"]
    peak.layers["raw"] = peak.X

    sc.pp.scale(corr)
    sc.pp.scale(expr)
    sc.pp.scale(peak)
    sc.tl.pca(corr, n_comps=pca_comps, svd_solver="auto")
    sc.tl.pca(expr, n_comps=pca_comps, svd_solver="auto")
    sc.tl.pca(peak, n_comps=pca_comps, svd_solver="auto")
    sc.pp.neighbors(corr)
    sc.pp.neighbors(expr)
    sc.pp.neighbors(peak)

    mu.pp.neighbors(TRAVMap,n_multineighbors=n_multineighbors)
    mu.tl.umap(TRAVMap)
    return(TRAVMap)
    
    