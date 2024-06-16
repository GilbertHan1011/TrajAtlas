from __future__ import annotations
import scanpy as sc
import pandas as pd
import numpy as np
from rich import print
from scipy.sparse import csr_matrix
from anndata import AnnData
#import scarches as sca
#from scipy.stats import pearsonr
#import anndata as ad
#from scipy.stats import norm
from sklearn.neighbors import KNeighborsTransformer
import joblib
from TrajAtlas.utils._docs import d
#import lightgbm
import os
from mudata import MuData


try:
    import scarches as sca
except ModuleNotFoundError:
    print(
        "[bold yellow]scarches is not installed. Install with [green]pip install sca [yellow]to project your datasets to our Differetiation Atlas."
    )
except Exception as e:
    print(f"[bold yellow]You should check whether scarches have properly installed")


location = os.path.dirname(os.path.realpath(__file__))
highVarGeneFile = os.path.join(location,'..','datasets', 'varGene_1500.csv')
trajMapFile=os.path.join(location, '..','datasets', 'trajMap_reference_1.h5ad')
refObs=os.path.join(location, '..','datasets', 'pred_obs.csv')
k_neighbor_model=os.path.join(location, '..','datasets', 'knn_transformer_model.joblib')
rfGeneFile=os.path.join(location,'..',"datasets","rf_genes.csv")
scanviModel=os.path.join(location, '..','datasets',"scanvi_model")
pseduoPredFile=os.path.join(location, '..','datasets', "pseduoPred","lightGBM_pred.pkl")


def _weighted_knn_transfer(
    query_adata,
    query_adata_emb,
    ref_adata_obs,
    label_keys,
    knn_model,
    threshold=1,
    pred_unknown=False,
    mode="package",
):
    """Annotates ``query_adata`` cells with an input trained weighted KNN classifier.
    Parameters
    ----------
    query_adata: :class:`~anndata.AnnData`
        Annotated dataset to be used to queryate KNN classifier. Embedding to be used
    query_adata_emb: str
        Name of the obsm layer to be used for label transfer. If set to "X",
        query_adata.X will be used
    ref_adata_obs: :class:`pd.DataFrame`
        obs of ref Anndata
    label_keys: str
        Names of the columns to be used as target variables (e.g. cell_type) in ``query_adata``.
    knn_model: :class:`~sklearn.neighbors._graph.KNeighborsTransformer`
        knn model trained on reference adata with weighted_knn_trainer function
    threshold: float
        Threshold of uncertainty used to annotating cells as "Unknown". cells with
        uncertainties higher than this value will be annotated as "Unknown".
        Set to 1 to keep all predictions. This enables one to later on play
        with thresholds.
    pred_unknown: bool
        ``False`` by default. Whether to annotate any cell as "unknown" or not.
        If `False`, ``threshold`` will not be used and each cell will be annotated
        with the label which is the most common in its ``n_neighbors`` nearest cells.
    mode: str
        Has to be one of "paper" or "package". If mode is set to "package",
        uncertainties will be 1 - P(pred_label), otherwise it will be 1 - P(true_label).
    """
    if not type(knn_model) == KNeighborsTransformer:
        raise ValueError(
            "knn_model should be of type sklearn.neighbors._graph.KNeighborsTransformer!"
        )

    if query_adata_emb == "X":
        query_emb = query_adata.X
    elif query_adata_emb in query_adata.obsm.keys():
        query_emb = query_adata.obsm[query_adata_emb]
    else:
        raise ValueError(
            "query_adata_emb should be set to either 'X' or the name of the obsm layer to be used!"
        )
    top_k_distances, top_k_indices = knn_model.kneighbors(X=query_emb)

    stds = np.std(top_k_distances, axis=1)
    stds = (2.0 / stds) ** 2
    stds = stds.reshape(-1, 1)

    top_k_distances_tilda = np.exp(-np.true_divide(top_k_distances, stds))

    weights = top_k_distances_tilda / np.sum(
        top_k_distances_tilda, axis=1, keepdims=True
    )
    cols = ref_adata_obs.columns[ref_adata_obs.columns.str.startswith(label_keys)]
    uncertainties = pd.DataFrame(columns=cols, index=query_adata.obs_names)
    pred_labels = pd.DataFrame(columns=cols, index=query_adata.obs_names)
    for i in range(len(weights)):
        for j in cols:
            y_train_labels = ref_adata_obs[j].values
            unique_labels = np.unique(y_train_labels[top_k_indices[i]])
            best_label, best_prob = None, 0.0
            for candidate_label in unique_labels:
                candidate_prob = weights[
                    i, y_train_labels[top_k_indices[i]] == candidate_label
                ].sum()
                if best_prob < candidate_prob:
                    best_prob = candidate_prob
                    best_label = candidate_label

            if pred_unknown:
                if best_prob >= threshold:
                    pred_label = best_label
                else:
                    pred_label = "Unknown"
            else:
                pred_label = best_label

            if mode == "package":
                uncertainties.iloc[i][j] = (max(1 - best_prob, 0))

            else:
                raise Exception("Inquery Mode!")

            pred_labels.iloc[i][j] = (pred_label)

    print("finished!")

    return pred_labels, uncertainties

def formOsteoAdata(adata, batchVal="sample",missing_threshold=500,variableFeature="Default"):
    
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

@d.dedent
def ProjectData(
        adata:AnnData,
        modelPath:str  = None,
        max_epoch:int = 100):
    """Projected query datasets (osteogenesis-related) to scANVI latent space :cite:`xuProbabilisticHarmonizationAnnotation2021` which 
    trained with Differentiation Atlas by scArches :cite:`lotfollahiMappingSinglecellData2022`.

    .. seealso::
        - See :doc:`../../../tutorial/1_OPCST_projecting` for how to
        projecting OPCST model to your datasets.
    
    Parameters
    ----------
    %(adata)s
    modelPath
        scANVI model. The default model loaded with scANVI is typically trained using the Differentiation Atlas dataset.
    max_epoch
        scANVI training epoch.
    

    
    Returns
    ----------------------
    :class:`adata <anndata.AnnData>` object. Updates :attr:`adata.obsm <anndata.AnnData.obsm>` with the following:

    - ``scANVI`` scANVI latent trained by scANVI models.
    """

    adata_immediate=formOsteoAdata(adata)
    if modelPath==None:
        modelPath=scanviModel
#    if isinstance(modelPath,str):
#        if modelPath=="Default":
#            modelPath=scanviModel
#            print("projecting....")
    print("projecting....")
    model = sca.models.SCANVI.load_query_data(
        adata_immediate,
        modelPath,
        freeze_dropout = True,
    )
    model.train(
        max_epochs=max_epoch,
        plan_kwargs=dict(weight_decay=0.0),
        check_val_every_n_epoch=10
    )
    query_latent = sc.AnnData(model.get_latent_representation())
    adata.obsm["scANVI"]=query_latent.X
    return(adata)

@d.dedent
def label_transfer(
        adata:AnnData
        ):
    """Transfer seven-level annotation system and lineage path to adata.

    .. seealso::
        - See :doc:`../../../tutorial/1_OPCST_projecting` for how to
        projecting OPCST model to your datasets.

    Parameters
    ----------
        %(adata)s

    Returns:
        :class:`adata <anndata.AnnData>` object. Also updates :attr:`adata.obs <anndata.AnnData.obs>` with the following:

        ``pred_level[1-7]_anno`` predicted seven-level annotation.

        ``predict_lineage_*`` predicted OPCST lineage path.


    """
    k_neighbors_transformer=joblib.load(k_neighbor_model)
    refTable=pd.read_csv(refObs,index_col=0)
    labels, _ = _weighted_knn_transfer(
        query_adata=adata,
        query_adata_emb="scANVI",
        label_keys="pred",
        knn_model=k_neighbors_transformer,
        ref_adata_obs = refTable
    )
    adata.obs[labels.columns]=labels
    adata.obs[['pred_lineage_lepr', 'pred_lineage_msc', 'pred_lineage_chondro',"pred_lineage_fibro"]]=adata.obs[['pred_lineage_lepr', 'pred_lineage_msc', 'pred_lineage_chondro',"pred_lineage_fibro"]].astype("str")
    return(adata)


@d.dedent
def pseduo_predict(adata:AnnData,
                   modelPath:str="Default"):
    """Predict common pseudotime.

    .. seealso::
        - See :doc:`../../../tutorial/1_OPCST_projecting` for how to
        projecting OPCST model to your datasets.


    Parameters
    ----------
    %(adata)s
    modelPath
        Path of model to predict pseudotime. The default model loaded was trained with Differentiation Atlas by LightGBMRegressor.
    
    Returns
    ----------------------
    :class:`adata <anndata.AnnData>` object. Also updates :attr:`adata.obs <anndata.AnnData.obs>` with the following:
        
    """

    rfGene=pd.read_csv(rfGeneFile ,index_col=0)
    gene=rfGene["gene"][rfGene["importance"]>0.000008]
    adata_immediate=formOsteoAdata(adata, variableFeature=gene,batchVal="sample")
    if isinstance(modelPath,str):
        if modelPath=="Default":
            modelPath=pseduoPredFile
    #==load model================
    model=joblib.load(modelPath)
    adata.obs["pseduoPred"]=model.predict(adata_immediate.layers["counts"])
    return(adata)


def substractLineageAdata(adata, lineage: list or None = ["Fibroblast", "LepR_BMSC", "MSC", "Chondro"]):
    lineageDict = {
        "Fibroblast": "pred_lineage_fibro",
        "LepR_BMSC": "pred_lineage_lepr",
        "MSC": "pred_lineage_msc",
        "Chondro": "pred_lineage_chondro"
    }
    if not isinstance(lineage, list):
        raise TypeError("Lineage argument must contain only the valid lineages: 'Fibroblast', 'LepR_BMSC', 'MSC', 'Chondro'.")
    if lineage is None:
        lineage = ["Fibroblast", "LepR_BMSC", "MSC", "Chondro"]

    #values = [lineageDict[key] for key in lineage if key in lineageDict]
    values = []
    for key in lineage:
        if key not in lineageDict:
            raise ValueError(f"Invalid lineage '{key}' provided. Lineage argument must contain only the valid lineages: 'Fibroblast', 'LepR_BMSC', 'MSC', 'Chondro'.")
        values.append(lineageDict[key])
    adata.obs[values] = adata.obs[values].astype("bool")
    boolVal = adata.obs[values].apply(lambda row: row.any(), axis=1)
    adata.obs["lineageSum"] = boolVal
    adata=adata[boolVal,:]

    return adata


