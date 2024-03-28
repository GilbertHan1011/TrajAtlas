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

    Each lineage is defined via its lineage weights. This function accepts any model based off
    :class:`~cellrank.models.BaseModel` to fit gene expression, where we take the lineage weights
    into account in the loss function.

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




@d.dedent
def gene_trends(
    adata: AnnData,
    model: _input_model_type,
    genes: Union[str, Sequence[str]],
    time_key: str,
    lineages: Optional[Union[str, Sequence[str]]] = None,
    backward: bool = False,
    data_key: str = "X",
    time_range: Optional[Union[_time_range_type, List[_time_range_type]]] = None,
    transpose: bool = False,
    callback: _callback_type = None,
    conf_int: Union[bool, float] = True,
    same_plot: bool = False,
    hide_cells: bool = False,
    perc: Optional[Union[Tuple[float, float], Sequence[Tuple[float, float]]]] = None,
    lineage_cmap: Optional[matplotlib.colors.ListedColormap] = None,
    cell_color: Optional[str] = None,
    cell_alpha: float = 0.6,
    lineage_alpha: float = 0.2,
    size: float = 15,
    lw: float = 2,
    cbar: bool = True,
    margins: float = 0.015,
    sharex: Optional[Union[str, bool]] = None,
    sharey: Optional[Union[str, bool]] = None,
    gene_as_title: Optional[bool] = None,
    legend_loc: Optional[str] = "best",
    obs_legend_loc: Optional[str] = "best",
    ncols: int = 2,
    suptitle: Optional[str] = None,
    return_models: bool = False,
    njobs: Optional[int] = 1,
    show_progress_bar: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[Union[str, pathlib.Path]] = None,
    return_figure: bool = False,
    **kwargs: Any,
) -> Optional[_return_model_type]:
    """Plot gene expression trends along lineages.

    .. seealso::
        - See :doc:`../../../notebooks/tutorials/estimators/800_gene_trends` on how to
          visualize the gene trends.

    Each lineage is defined via its lineage weights. This function accepts any model based off
    :class:`~cellrank.models.BaseModel` to fit gene expression, where we take the lineage weights
    into account in the loss function.

    Parameters
    ----------
    %(adata)s
    %(model)s
    %(genes)s
    time_key
        Key in :attr:`~anndata.AnnData.obs` where the pseudotime is stored.
    lineages
        Names of the lineages to plot. If :obj:`None`, plot all lineages.
    %(backward)s
    data_key
        Key in :attr:`~anndata.AnnData.layers` or ``'X'`` for :attr:`~anndata.AnnData.X` where the data is stored.
    %(time_range)s
        This can also be specified on per-lineage basis.
    %(gene_symbols)s
    transpose
        If ``same_plot = True``, group the trends by ``lineages`` instead of ``genes``.
        This forces ``hide_cells = True``.
        If ``same_plot = False``, show ``lineages`` in rows and ``genes`` in columns.
    %(model_callback)s
    conf_int
        Whether to compute and show confidence interval. If the ``model`` is :class:`~cellrank.models.GAMR`,
        it can also specify the confidence level, the default is :math:`0.95`.
    same_plot
        Whether to plot all lineages for each gene in the same plot.
    hide_cells
        If :obj:`True`, hide all cells.
    perc
        Percentile for colors. Valid values are in :math:`[0, 100]`.
        This can improve visualization. Can be specified individually for each lineage.
    lineage_cmap
        Categorical colormap to use when coloring in the lineages. If :obj:`None` and ``same_plot = True``,
        use the corresponding colors in :attr:`~anndata.AnnData.uns`, otherwise use ``'black'``.
    fate_prob_cmap
        Continuous colormap to use when visualizing the fate probabilities for each lineage.
        Only used when ``same_plot = False``.
    cell_color
        Key in :attr:`~anndata.AnnData.obs` or :attr:`~anndata.AnnData.var_names` used for coloring the cells.
    cell_alpha
        Alpha channel for cells.
    lineage_alpha
        Alpha channel for lineage confidence intervals.
    size
        Size of the points.
    lw
        Line width of the smoothed values.
    cbar
        Whether to show colorbar. Always shown when percentiles for lineages differ.
        Only used when ``same_plot = False``.
    margins
        Margins around the plot.
    sharex
        Whether to share x-axis. Valid options are ``'row'``, ``'col'`` or ``'none'``.
    sharey
        Whether to share y-axis. Valid options are ``'row'`, ``'col'`` or ``'none'``.
    gene_as_title
        Whether to show gene names as titles instead on y-axis.
    legend_loc
        Location of the legend displaying lineages. Only used when ``same_plot = True``.
    obs_legend_loc
        Location of the legend when ``cell_color`` corresponds to a categorical variable.
    ncols
        Number of columns of the plot when plotting multiple genes. Only used when ``same_plot = True``.
    suptitle
        Suptitle of the figure.
    return_figure
        Whether to return the figure object.
    %(return_models)s
    %(parallel)s
    %(plotting)s
    plot_kwargs
        Keyword arguments for the :meth:`~cellrank.models.BaseModel.plot`.
    kwargs
        Keyword arguments for :meth:`~cellrank.models.BaseModel.prepare`.

    Returns
    -------
    %(plots_or_returns_models)s
    """
    if isinstance(genes, str):
        genes = [genes]
    genes = _unique_order_preserving(genes)

    _check_collection(
        adata,
        genes,
        "obs" if data_key == "obs" else "var_names",
        use_raw=kwargs.get("use_raw", False),
    )

    probs = Lineage.from_adata(adata, backward=backward)
    if lineages is None:
        lineages = probs.names
    elif isinstance(lineages, str):
        lineages = [lineages]
    elif all(ln is None for ln in lineages):  # no lineage, all the weights are 1
        lineages = [None]
        cbar = False
        logg.debug("All lineages are `None`, setting the weights to `1`")
    lineages = _unique_order_preserving(lineages)

    if isinstance(time_range, (tuple, float, int, type(None))):
        time_range = [time_range] * len(lineages)
    elif len(time_range) != len(lineages):
        raise ValueError(f"Expected time ranges to be of length `{len(lineages)}`, found `{len(time_range)}`.")

    kwargs["time_key"] = time_key
    kwargs["data_key"] = data_key
    kwargs["backward"] = backward
    kwargs["conf_int"] = conf_int  # prepare doesnt take or need this
    models = _create_models(model, genes, lineages)

    all_models, models, genes, lineages = _fit_bulk(
        models,
        _create_callbacks(adata, callback, genes, lineages, **kwargs),
        genes,
        lineages,
        time_range,
        return_models=True,
        filter_all_failed=False,
        parallel_kwargs={
            "show_progress_bar": show_progress_bar,
            "njobs": _get_n_cores(njobs, len(genes)),
            "backend": _get_backend(models, backend),
        },
        **kwargs,
    )

    lineages = sorted(lineages)
    probs = probs[lineages]
    if lineage_cmap is None and not transpose:
        lineage_cmap = probs.colors

    plot_kwargs = dict(plot_kwargs)
    plot_kwargs["obs_legend_loc"] = obs_legend_loc
    if transpose:
        all_models = pd.DataFrame(all_models).T.to_dict()
        models = pd.DataFrame(models).T.to_dict()
        genes, lineages = lineages, genes
        hide_cells = same_plot or hide_cells
    else:
        # information overload otherwise
        plot_kwargs["lineage_probability"] = False
        plot_kwargs["lineage_probability_conf_int"] = False

    tmp = pd.DataFrame(models).T.astype(bool)
    start_rows = np.argmax(tmp.values, axis=0)
    end_rows = tmp.shape[0] - np.argmax(tmp[::-1].values, axis=0) - 1

    if same_plot:
        gene_as_title = True if gene_as_title is None else gene_as_title
        sharex = "all" if sharex is None else sharex
        if sharey is None:
            sharey = "row" if plot_kwargs.get("lineage_probability", False) else "none"
        ncols = len(genes) if ncols >= len(genes) else ncols
        nrows = int(np.ceil(len(genes) / ncols))
    else:
        gene_as_title = False if gene_as_title is None else gene_as_title
        sharex = "col" if sharex is None else sharex
        if sharey is None:
            sharey = "row" if not hide_cells or plot_kwargs.get("lineage_probability", False) else "none"
        nrows = len(genes)
        ncols = len(lineages)

    plot_kwargs = dict(plot_kwargs)
    if plot_kwargs.get("xlabel", None) is None:
        plot_kwargs["xlabel"] = time_key

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        figsize=(6 * ncols, 4 * nrows) if figsize is None else figsize,
        tight_layout=True,
        dpi=dpi,
    )
    axes = np.reshape(axes, (nrows, ncols))

    cnt = 0
    plot_kwargs["obs_legend_loc"] = None if same_plot else obs_legend_loc

    logg.info("Plotting trends")
    for row in range(len(axes)):
        for col in range(len(axes[row])):
            if cnt >= len(genes):
                break
            gene = genes[cnt]
            if same_plot and plot_kwargs.get("lineage_probability", False) and transpose:
                lpc = probs[gene].colors[0]
            else:
                lpc = None

            if same_plot:
                plot_kwargs["obs_legend_loc"] = obs_legend_loc if row == 0 and col == len(axes[0]) - 1 else None

            _trends_helper(
                models,
                gene=gene,
                lineage_names=lineages,
                transpose=transpose,
                same_plot=same_plot,
                hide_cells=hide_cells,
                perc=perc,
                lineage_cmap=lineage_cmap,
                fate_prob_cmap=fate_prob_cmap,
                lineage_probability_color=lpc,
                cell_color=cell_color,
                alpha=cell_alpha,
                lineage_alpha=lineage_alpha,
                size=size,
                lw=lw,
                cbar=cbar,
                margins=margins,
                sharey=sharey,
                gene_as_title=gene_as_title,
                legend_loc=legend_loc,
                figsize=figsize,
                fig=fig,
                axes=axes[row, col] if same_plot else axes[cnt],
                show_ylabel=col == 0,
                show_lineage=same_plot or (cnt == start_rows),
                show_xticks_and_label=((row + 1) * ncols + col >= len(genes)) if same_plot else (cnt == end_rows),
                **plot_kwargs,
            )
            cnt += 1  # plot legend on the 1st plot

            if not same_plot:
                plot_kwargs["obs_legend_loc"] = None

    if same_plot and (col != ncols):
        for ax in np.ravel(axes)[cnt:]:
            ax.remove()

    fig.suptitle(suptitle, y=1.05)

    if return_figure:
        return fig

    if save is not None:
        save_fig(fig, save)

    if return_models:
        return all_models

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
    expr=TRAVMap["expr"]
    peak=TRAVMap["peak"]
    trav=TRAVMap["TRAV"]
    sc.pp.scale(corr)
    sc.pp.scale(expr)
    sc.pp.scale(peak)
    sc.pp.scale(trav)
    sc.tl.pca(corr, n_comps=pca_comps, svd_solver="auto")
    sc.tl.pca(expr, n_comps=pca_comps, svd_solver="auto")
    sc.tl.pca(peak, n_comps=pca_comps, svd_solver="auto")
    sc.tl.pca(trav, n_comps=pca_comps, svd_solver="auto")
    sc.pp.neighbors(corr)
    sc.pp.neighbors(expr)
    sc.pp.neighbors(peak)
    sc.pp.neighbors(trav)
    mu.pp.neighbors(TRAVMap,n_multineighbors=20)
    mu.tl.umap(TRAVMap)
    return(TRAVMap)
    
    