import scanpy as sc
import anndata
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import scvi
import scarches as sca
import tensorflow as tf
from scipy.stats import pearsonr
import anndata as ad
from scipy.stats import ttest_ind
from scipy.stats import norm

def formOsteoAdata(adata,missing_threshold, batchVal,variableFeature="Default"):
    print("Total number of genes needed for mapping:",len(variableFeature))
    print(
        "Number of genes found in query dataset:",
        adata.var_names.isin(variableFeature).sum(),
    )
    if isinstance(variableFeature,str):
        if variableFeature=="Default":
            variableFeature=pd.read_csv(highVarGeneFile,index_col=0)["0"].values
    if(len(variableFeature)-adata.var_names.isin(variableFeature).sum()>missing_threshold):
        raise ValueError("Too many missing gene! Please check data!")
    missing_genes = [
        gene_id
        for gene_id in variableFeature
        if gene_id not in adata.var_names
    ]
    missing_gene_adata = sc.AnnData(
        X=csr_matrix(np.zeros(shape=(adata.n_obs, len(missing_genes))), dtype="float32"),
        obs=adata.obs.iloc[:, :1],
        var=missing_genes,
    )
    missing_gene_adata.var_names=missing_genes
    missing_gene_adata.layers["counts"] = missing_gene_adata.X
    if "PCs" in adata.varm.keys():
        del adata.varm["PCs"]
        
    adata_merged = sc.concat(
        [adata, missing_gene_adata],
        axis=1,
        join="outer",
        index_unique=None,
        merge="unique",
    )
    adata_merged = adata_merged[
        :, variableFeature
    ].copy()
    adata_merged.obs["batch"]=adata_merged.obs[batchVal]
    return(adata_merged)


def lineagePredict(adata,chondroPath="Default",mesPath="Default",leprPath="Default",max_epoch=100):
    if isinstance(mesPath,str):
        if mesPath=="Default":
            mesPath=scanviMesFile
    if isinstance(chondroPath,str):
        if chondroPath=="Default":
            chondroPath=scanviChondroFile
            
    if isinstance(leprPath,str):
        if leprPath=="Default":
            leprPath=scanviLeprFile
    #= predict chondro lineage
    print("predicting chondro path")
    model_chondro = sca.models.SCANVI.load_query_data(
        adata,
        chondroPath,
        freeze_dropout = True,
    )
    model_chondro.train(
        max_epochs=max_epoch,
        plan_kwargs=dict(weight_decay=0.0),
        check_val_every_n_epoch=10
    )
    adata.obs["chondro_prediction"]=model_chondro.predict()
    print("predicting lepr path")
    model_lepr = sca.models.SCANVI.load_query_data(
        adata,
        leprPath,
        freeze_dropout = True,
    )
    model_lepr.train(
        max_epochs=max_epoch,
        plan_kwargs=dict(weight_decay=0.0),
        check_val_every_n_epoch=10
    )
    adata.obs["lepr_prediction"]=model_lepr.predict()
    print("predicting mes/fibro path")
    model_mes = sca.models.SCANVI.load_query_data(
        adata,
        mesPath,
        freeze_dropout = True,
    )
    model_mes.train(
        max_epochs=max_epoch,
        plan_kwargs=dict(weight_decay=0.0),
        check_val_every_n_epoch=10
    )
    adata.obs["mes_prediction"]=model_mes.predict()
    return(adata)


def pseduo_predict(adata,modelPath="Default"):
    if isinstance(variableFeature,str):
        if modelPath=="Default":
            modelPath=pseduoPredFile
    #==load model================
    model = tf.keras.models.load_model(modelPath)
    adataDf=adata.layers["counts"].toarray()
    adata.obs["pseduotime_predcit"]=model.predict(adataDf)
    return(adata)


def pseduo_traj(adata,cell_threshold=20, p_threshold=0.05,sample="orig.ident",organ="Limb adult",
               chondro_bool="chondro_bool",mes_bool="mes_bool",lepr_bool="lepr_bool",
                pseduotime_predcit="pseduotime_predcit",meta=None):
    select_lineage=[]
    sample_col=adata.obs[sample].cat.categories
    if (organ=="Limb adult"):
        lineageDict={"Chondro":adata.obs[chondro_bool],
             "Fibro":adata.obs[mes_bool],
             "Lepr":adata.obs[lepr_bool]}
        
    else:
        lineageDict={"Chondro":adata.obs[chondro_bool],
             "Mesenchyme":adata.obs[mes_bool]}
    for j in sample_col:
        dfLogic=adata.obs[sample]==j
        for i in  lineageDict.keys():
        # We belelieve that cells in transition state is most important to check whether a lineage is existed.
        # So we consider the pseudotime in the interval [0.2,0.8] to be meaningful for a spectrum of more than [threshold] cells
            logic=dfLogic&lineageDict[i]&(adata.obs[pseduotime_predcit]<0.8)&(adata.obs[pseduotime_predcit]>0.2)
            if(logic.sum()>threshold):
                select_lineage.append(i+("_")+j)
                
    geneMatrix=pd.DataFrame(adata.X.toarray())
    geneMatrix.index=adata.obs_names
    geneMatrix.columns=adata.var_names
    correlationMatrix=pd.DataFrame(columns=select_lineage,index=geneMatrix.columns)
    pvalueMatrix=pd.DataFrame(columns=select_lineage,index=geneMatrix.columns)

    for j in sample_col:
        dfLogic=adata.obs[sample]==j
        for i in  lineageDict.keys():
            # We belelieve that cells in transition state is most important to check whether a lineage is existed.
            # So we consider the pseudotime in the interval [0.2,0.8] to be meaningful for a spectrum of more than [threshold] cells
            logic=dfLogic&lineageDict[i]&(adata.obs[pseduotime_predcit]<0.8)&(adata.obs[pseduotime_predcit]>0.2)
            if(logic.sum()>threshold):
                colname=i+("_")+j
                print(colname)
                df=geneMatrix.loc[logic.values,:]
                pseduo_tmp=adata.obs[pseduotime_predcit].values[logic.values]
                for k in correlationMatrix.index:
                    gene_df=df.loc[:,k]
                    corr_coef, p_value = pearsonr(gene_df, pseduo_tmp)
                    correlationMatrix.loc[k,colname]=corr_coef
                    pvalueMatrix.loc[k,colname]=p_value
                    
    pvalueMask=pvalueMatrix<p_threshold
    pvalueMask=~pvalueMask
    correlationMatrixMasked = correlationMatrix.mask(pvalueMask, 0)
    trajAdata=sc.AnnData(correlationMatrixMasked.transpose())
    
    sample = [string.split('_', 1)[1] if '_' in string else string for string in select_lineage]
    traj_method=[string.split('_', 1)[0]if '_' in string else string for string in select_lineage]
    traj_meta={"traj":select_lineage,"Sample":sample,"Methods":traj_method}
    traj_df=pd.DataFrame(traj_meta)
    traj_df.index=traj_df["traj"]

    if not(isinstance(trajMeta,pd.DataFrame)):
        raise ValueError("Please input metadata as a Pandas' DataFrame")
    if meta is not None:
        if ("Sample" in meta.columns):
            traj_df= pd.merge(traj_df, meta, on='Sample')
        else:
          raise ValueError("Please ensure Sample is in the column names")  
    trajAdata.obs=traj_df        
    return(trajAdata)


#== integrate adata-----------------
def integrateTrajMap(adata,mapPath="Default"):
    if isinstance(mapPath,str):
        if modelPath=="Default":
            modelPath=trajMapFile
    trajMap=sc.read(mapPath)
    trajMap.obs["current"]="Old"
    pseduotimeAdata.obs["current"]="new"
    adata_concat = ad.concat([trajMap, pseduotimeAdata],join="outer")
    return(adata_concat)


#== the input sample must be a list or array
def calculate_posterior(sample, method, adata):
    sample=list(set(sample).intersection(set(adata.obs["Sample"].values)))
    sample=np.array(sample,dtype=object)
    sample=method+"_"+sample
    adataMatrix=pd.DataFrame(adata.X)
    adataMatrix.columns=adata.var_names
    adataMatrix.index=adata.obs_names
    geneList = adataMatrix.loc[sample]
    geneList = geneList.loc[:, np.array(geneList.sum() != 0)].columns
    dfTest = pd.DataFrame(columns=['gene', 'probility'])
    for gene in geneList:
        point = adataMatrix[gene][sample]
        point = np.array(point[point != 0])
        geneTest = adataMatrix[gene].loc[np.array(adata.obs["Methods"] == method)]
        geneTest = geneTest[geneTest != 0]
        if (len(point) == 1):
            mu = np.mean(geneTest)
            sigma = np.std(geneTest)
            # Calculate the posterior probability of the points belonging to the group
            prob = norm.cdf(point, mu, sigma)
            gene_dict = {"gene": gene, "probility": prob[0]}
            dfTest.loc[len(dfTest)] = gene_dict
        else:
            statistic, p_value = ttest_ind(geneTest, point)
            gene_dict = {"gene": gene, "probility": p_value}
            dfTest.loc[len(dfTest)] = gene_dict
        dfTest=dfTest.sort_values(by='probility')
    return dfTest
