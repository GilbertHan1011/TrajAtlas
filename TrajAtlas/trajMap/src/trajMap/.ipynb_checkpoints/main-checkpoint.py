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
import os

location = os.path.dirname(os.path.realpath(__file__))
highVarGeneFile = os.path.join(location, 'datasets', 'variable_2000.csv')
trajMapFile=os.path.join(location, 'datasets', 'trajMap_reference_1.h5ad')
scanviMesFile=os.path.join(location, 'datasets', "scanvi_mes")
scanviLeprFile=os.path.join(location, 'datasets', "scanvi_lepr")
scanviChondroFile=os.path.join(location, 'datasets', "scanvi_chondro")
pseduoPredFile=os.path.join(location, 'datasets', "pseduoPred")

def formOsteoAdata(adata, batchVal,missing_threshold=500,variableFeature="Default"):
    
    if isinstance(variableFeature,str):
        if variableFeature=="Default":
            variableFeature=pd.read_csv(highVarGeneFile,index_col=0)["0"].values
    if(len(variableFeature)-adata.var_names.isin(variableFeature).sum()>missing_threshold):
        raise ValueError("Too many missing gene! Please check data!")
        
    print("Total number of genes needed for mapping:",len(variableFeature))
    print(
        "Number of genes found in query dataset:",
        adata.var_names.isin(variableFeature).sum(),
    )
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
    adata_merged.obs["batch"]=adata_merged.obs[batchVal].astype(str)
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
    if isinstance(modelPath,str):
        if modelPath=="Default":
            modelPath=pseduoPredFile
    #==load model================
    model = tf.keras.models.load_model(modelPath)
    adataDf=adata.layers["counts"].toarray()
    adata.obs["pseduotime_predcit"]=model.predict(adataDf)
    return(adata)


def substractLineageAdata(adata,lepr_pred="lepr_prediction",mes_pred='mes_prediction',chondro_pred="chondro_prediction"):
    lepr_bool=adata.obs[lepr_pred]=="True"
    mes_bool=adata.obs[mes_pred]=="True"
    chondro_bool=adata.obs[chondro_pred]=="True"
    lineageBool=chondro_bool | mes_bool|lepr_bool
    adata=adata[lineageBool,:]
    adata.obs["lepr_bool"]=lepr_bool
    adata.obs["mes_bool"]=mes_bool
    adata.obs["chondro_bool"]=chondro_bool
    return(adata)

def pseduo_traj(adata,cell_threshold=20, p_threshold=0.05,sample="batch",organ="Limb adult",
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
            if(logic.sum()>cell_threshold):
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
            if(logic.sum()>cell_threshold):
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
    

    #if not(isinstance(meta,pd.DataFrame)):
    #    raise ValueError("Please input metadata as a Pandas' DataFrame")
    if meta is not None:
        if not(isinstance(meta,pd.DataFrame)):
            raise ValueError("Please input metadata as a Pandas' DataFrame")
        if ("Sample" in meta.columns):
            traj_df= pd.merge(traj_df, meta, on='Sample')
        else:
          raise ValueError("Please ensure Sample is in the column names")  
    traj_df.index=traj_df["traj"]
    trajAdata.obs=traj_df        
    return(trajAdata)


#== integrate adata-----------------
def integrateTrajMap(adata,mapPath="Default"):
    if isinstance(mapPath,str):
        if mapPath=="Default":
            mapPath=trajMapFile
    trajMap=sc.read(mapPath)
    trajMap.obs["current"]="Old"
    adata.obs["current"]="new"
    adata_concat = ad.concat([trajMap, adata],join="outer")
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
