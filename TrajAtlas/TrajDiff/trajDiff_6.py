from __future__ import annotations
import random
import logging
import re
from itertools import compress
import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData
from rich import print
from scipy.stats import binom
from joblib import Parallel, delayed
from tqdm import tqdm
import statsmodels.api as sm
import scanpy as sc

from TrajAtlas.TrajDiff.trajdiff_utils import _test_binom

pd.DataFrame.iteritems = pd.DataFrame.items
try:
    from rpy2.robjects import conversion, numpy2ri, pandas2ri
    from rpy2.robjects.packages import STAP, PackageNotInstalledError, importr
except ModuleNotFoundError:
    print(
        "[bold yellow]rpy2 is not installed. Install with [green]pip install rpy2 [yellow]to run tools with R support."
    )
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances

class Tdiff:
    """Python implementation of tdiff."""

    def __init__(self):
        pass

    def load(
        self,
        input: AnnData,
        feature_key: str | None = "rna"
    ) -> MuData:
        """Prepare a MuData object for subsequent processing.

        Args:
            input: AnnData
            feature_key: Key to store the cell-level AnnData object in the MuData object
        Returns:
            MuData: MuData object with original AnnData (default is `mudata[feature_key]`).
        """
        mdata = MuData({feature_key: input, "tdiff": AnnData(),"pseudobulk":AnnData()})

        return mdata

    def make_nhoods(
        self,
        data: AnnData | MuData,
        neighbors_key: str | None = None,
        feature_key: str | None = "rna",
        prop: float = 0.1,
        seed: int = 0,
        copy: bool = False,
    ):
        """Randomly sample vertices on a KNN graph to define neighbourhoods of cells.

        The set of neighborhoods get refined by computing the median profile for the neighbourhood in reduced dimensional space
        and by selecting the nearest vertex to this position.
        Thus, multiple neighbourhoods may be collapsed to prevent over-sampling the graph space.

        Args:
            data: AnnData object with KNN graph defined in `obsp` or MuData object with a modality with KNN graph defined in `obsp`
            neighbors_key: The key in `adata.obsp` or `mdata[feature_key].obsp` to use as KNN graph.
                           If not specified, `make_nhoods` looks .obsp[‘connectivities’] for connectivities (default storage places for `scanpy.pp.neighbors`).
                           If specified, it looks at .obsp[.uns[neighbors_key][‘connectivities_key’]] for connectivities.
                           (default: None)
            feature_key: If input data is MuData, specify key to cell-level AnnData object. (default: 'rna')
            prop: Fraction of cells to sample for neighbourhood index search. (default: 0.1)
            seed: Random seed for cell sampling. (default: 0)
            copy: Determines whether a copy of the `adata` is returned. (default: False)

        Returns:
            If `copy=True`, returns the copy of `adata` with the result in `.obs`, `.obsm`, and `.uns`.
            Otherwise:

            nhoods: scipy.sparse._csr.csr_matrix in `adata.obsm['nhoods']`.
            A binary matrix of cell to neighbourhood assignments. Neighbourhoods in the columns are ordered by the order of the index cell in adata.obs_names

            nhood_ixs_refined: pandas.Series in `adata.obs['nhood_ixs_refined']`.
            A boolean indicating whether a cell is an index for a neighbourhood

            nhood_kth_distance: pandas.Series in `adata.obs['nhood_kth_distance']`.
            The distance to the kth nearest neighbour for each index cell (used for SpatialFDR correction)

            nhood_neighbors_key: `adata.uns["nhood_neighbors_key"]`
            KNN graph key, used for neighbourhood construction
        """
        if isinstance(data, MuData):
            adata = data[feature_key]
        if isinstance(data, AnnData):
            adata = data
        if copy:
            adata = adata.copy()

        # Get reduced dim used for KNN graph
        if neighbors_key is None:
            try:
                use_rep = adata.uns["neighbors"]["params"]["use_rep"]
            except KeyError:
                logging.warning("Using X_pca as default embedding")
                use_rep = "X_pca"
            try:
                knn_graph = adata.obsp["connectivities"].copy()
            except KeyError:
                print('No "connectivities" slot in adata.obsp -- please run scanpy.pp.neighbors(adata) first')
                raise
        else:
            try:
                use_rep = adata.uns["neighbors"]["params"]["use_rep"]
            except KeyError:
                logging.warning("Using X_pca as default embedding")
                use_rep = "X_pca"
            knn_graph = adata.obsp[neighbors_key + "_connectivities"].copy()

        X_dimred = adata.obsm[use_rep]
        n_ixs = int(np.round(adata.n_obs * prop))
        knn_graph[knn_graph != 0] = 1
        random.seed(seed)
        random_vertices = random.sample(range(adata.n_obs), k=n_ixs)
        random_vertices.sort()
        ixs_nn = knn_graph[random_vertices, :]
        non_zero_rows = ixs_nn.nonzero()[0]
        non_zero_cols = ixs_nn.nonzero()[1]
        refined_vertices = np.empty(
            shape=[
                len(random_vertices),
            ]
        )

        for i in range(len(random_vertices)):
            nh_pos = np.median(X_dimred[non_zero_cols[non_zero_rows == i], :], 0).reshape(-1, 1)
            nn_ixs = non_zero_cols[non_zero_rows == i]
            # Find closest real point (amongst nearest neighbors)
            dists = euclidean_distances(X_dimred[non_zero_cols[non_zero_rows == i], :], nh_pos.T)
            # Update vertex index
            refined_vertices[i] = nn_ixs[dists.argmin()]

        refined_vertices = np.unique(refined_vertices.astype("int"))
        refined_vertices.sort()

        nhoods = knn_graph[:, refined_vertices]
        adata.obsm["nhoods"] = nhoods

        # Add ixs to adata
        adata.obs["nhood_ixs_random"] = adata.obs_names.isin(adata.obs_names[random_vertices])
        adata.obs["nhood_ixs_refined"] = adata.obs_names.isin(adata.obs_names[refined_vertices])
        adata.obs["nhood_ixs_refined"] = adata.obs["nhood_ixs_refined"].astype("int")
        adata.obs["nhood_ixs_random"] = adata.obs["nhood_ixs_random"].astype("int")
        adata.uns["nhood_neighbors_key"] = neighbors_key
        # Store distance to K-th nearest neighbor (used for spatial FDR correction)
        if neighbors_key is None:
            knn_dists = adata.obsp["distances"]
        else:
            knn_dists = adata.obsp[neighbors_key + "_distances"]

        nhood_ixs = adata.obs["nhood_ixs_refined"] == 1
        dist_mat = knn_dists[nhood_ixs, :]
        k_distances = dist_mat.max(1).toarray().ravel()
        adata.obs["nhood_kth_distance"] = 0
        adata.obs.loc[adata.obs["nhood_ixs_refined"] == 1, "nhood_kth_distance"] = k_distances

        if copy:
            return adata

    def count_nhoods(
        self,
        data: AnnData | MuData,
        sample_col: str,
        feature_key: str | None = "rna",
    ):
        """Builds a sample-level AnnData object storing the matrix of cell counts per sample per neighbourhood.

        Args:
            data: AnnData object with neighbourhoods defined in `obsm['nhoods']` or MuData object with a modality with neighbourhoods defined in `obsm['nhoods']`
            sample_col: Column in adata.obs that contains sample information
            feature_key: If input data is MuData, specify key to cell-level AnnData object. (default: 'rna')

        Returns:
            MuData object storing the original (i.e. rna) AnnData in `mudata[feature_key]`
            and the compositional anndata storing the neighbourhood cell counts in `mudata['tdiff']`.
            Here:
            - `mudata['tdiff'].obs_names` are samples (defined from `adata.obs['sample_col']`)
            - `mudata['tdiff'].var_names` are neighbourhoods
            - `mudata['tdiff'].X` is the matrix counting the number of cells from each
            sample in each neighbourhood
        """
        if isinstance(data, MuData):
            adata = data[feature_key]
            is_MuData = True
        if isinstance(data, AnnData):
            adata = data
            is_MuData = False
        if isinstance(adata, AnnData):
            try:
                nhoods = adata.obsm["nhoods"]
            except KeyError:
                print('Cannot find "nhoods" slot in adata.obsm -- please run tdiffpy.make_nhoods(adata)')
                raise
        # Make nhood abundance matrix
        sample_dummies = pd.get_dummies(adata.obs[sample_col])
        all_samples = sample_dummies.columns
        sample_dummies = csr_matrix(sample_dummies.values)
        nhood_count_mat = nhoods.T.dot(sample_dummies)
        sample_obs = pd.DataFrame(index=all_samples)
        sample_adata = AnnData(X=nhood_count_mat.T, obs=sample_obs, dtype=np.float32)
        sample_adata.uns["sample_col"] = sample_col
        # Save nhood index info
        sample_adata.var["index_cell"] = adata.obs_names[adata.obs["nhood_ixs_refined"] == 1]
        sample_adata.var["kth_distance"] = adata.obs.loc[
            adata.obs["nhood_ixs_refined"] == 1, "nhood_kth_distance"
        ].values

        if is_MuData is True:
            data.mod["tdiff"] = sample_adata
            return data
        else:
            tdiff_mdata = MuData({feature_key: adata, "tdiff": sample_adata})
            return tdiff_mdata
    def make_null(
        self,
        mdata: MuData,
        design: str,
        model_contrasts: str | None = None,
        subset_samples: list[str] | None = None,
        add_intercept: bool = True,
        feature_key: str | None = "rna",
        FDR:int = 0.05,
        times=5
    ):
        null_dict={}
        for i in range(times):
            fdr_=self.da_nhoods(mdata,design=design,model_contrasts=model_contrasts,subset_samples=subset_samples,
                               add_intercept=add_intercept,feature_key=feature_key,shuffle=True,return_fdr=True)

            null_dict[i]=fdr_

        null_df=pd.DataFrame(null_dict)
        null_df=null_df<FDR
        null_df["RowMean"]=null_df.mean(axis=1)
        sample_adata = mdata["tdiff"]
        sample_adata.var["null"]=null_df["RowMean"]

    
    def da_nhoods(
        self,
        mdata: MuData,
        design: str,
        model_contrasts: str | None = None,
        subset_samples: list[str] | None = None,
        add_intercept: bool = True,
        feature_key: str | None = "rna",
        shuffle: bool = False,
        return_fdr= False
    ):
        """Performs differential abundance testing on neighbourhoods using QLF test implementation as implemented in edgeR.

        Args:
            mdata: MuData object
            design: Formula for the test, following glm syntax from R (e.g. '~ condition').
                    Terms should be columns in `tdiff_mdata[feature_key].obs`.
            model_contrasts: A string vector that defines the contrasts used to perform DA testing, following glm syntax from R (e.g. "conditionDisease - conditionControl").
                             If no contrast is specified (default), then the last categorical level in condition of interest is used as the test group. Defaults to None.
            subset_samples: subset of samples (obs in `tdiff_mdata['tdiff']`) to use for the test. Defaults to None.
            add_intercept: whether to include an intercept in the model. If False, this is equivalent to adding + 0 in the design formula. When model_contrasts is specified, this is set to False by default. Defaults to True.
            feature_key: If input data is MuData, specify key to cell-level AnnData object. Defaults to 'rna'.

        Returns:
            None, modifies `tdiff_mdata['tdiff']` in place, adding the results of the DA test to `.var`:
            - `logFC` stores the log fold change in cell abundance (coefficient from the GLM)
            - `PValue` stores the p-value for the QLF test before multiple testing correction
            - `SpatialFDR` stores the the p-value adjusted for multiple testing to limit the false discovery rate,
                calculated with weighted Benjamini-Hochberg procedure
        """
        try:
            if shuffle:
                sample_adata = mdata["tdiff"].copy()
            else:
                sample_adata = mdata["tdiff"]
        except KeyError:
            print(
                "[bold red]tdiff_mdata should be a MuData object with two slots:"
                " feature_key and 'tdiff' - please run tdiffpy.count_nhoods() first"
            )
            raise
        adata = mdata[feature_key]

        covariates = [x.strip(" ") for x in set(re.split("\\+|\\*", design.lstrip("~ ")))]

        # Add covariates used for testing to sample_adata.var
        sample_col = sample_adata.uns["sample_col"]
        try:
            sample_obs = adata.obs[covariates + [sample_col]].drop_duplicates()
        except KeyError:
            missing_cov = [x for x in covariates if x not in sample_adata.obs.columns]
            print("Covariates {c} are not columns in adata.obs".format(c=" ".join(missing_cov)))
            raise
        sample_obs = sample_obs[covariates + [sample_col]]
        sample_obs.index = sample_obs[sample_col].astype("str")
    
        if shuffle:
            seed = np.random.randint(0, 100000)  
            np.random.seed(seed)  
            shuffled_array = np.random.permutation(sample_obs[covariates].values)
            sample_obs[covariates]=shuffled_array

        try:
            assert sample_obs.loc[sample_adata.obs_names].shape[0] == len(sample_adata.obs_names)
        except AssertionError:
            print(
                f"Values in mdata[{feature_key}].obs[{covariates}] cannot be unambiguously assigned to each sample"
                f" -- each sample value should match a single covariate value"
            )
            raise
        sample_adata.obs = sample_obs.loc[sample_adata.obs_names]

        # Get design dataframe
        try:
            design_df = sample_adata.obs[covariates]
        except KeyError:
            missing_cov = [x for x in covariates if x not in sample_adata.obs.columns]
            print('Covariates {c} are not columns in adata.uns["sample_adata"].obs'.format(c=" ".join(missing_cov)))
            raise
        # Get count matrix
        count_mat = sample_adata.X.T.toarray()
        lib_size = count_mat.sum(0)

        # Filter out samples with zero counts
        keep_smp = lib_size > 0

        # Subset samples
        if subset_samples is not None:
            keep_smp = keep_smp & sample_adata.obs_names.isin(subset_samples)
            design_df = design_df[keep_smp]
            for i, e in enumerate(design_df.columns):
                if design_df.dtypes[i].name == "category":
                    design_df[e] = design_df[e].cat.remove_unused_categories()

        # Filter out nhoods with zero counts (they can appear after sample filtering)
        keep_nhoods = count_mat[:, keep_smp].sum(1) > 0


        # Set up rpy2 to run edgeR
        edgeR, limma, stats, base = self._setup_rpy2()

        # Define model matrix
        if not add_intercept or model_contrasts is not None:
            design = design + " + 0"
        model = stats.model_matrix(object=stats.formula(design), data=design_df)

        # Fit NB-GLM
        dge = edgeR.DGEList(counts=count_mat[keep_nhoods, :][:, keep_smp], lib_size=lib_size[keep_smp])
        dge = edgeR.calcNormFactors(dge, method="TMM")
        dge = edgeR.estimateDisp(dge, model)
        fit = edgeR.glmQLFit(dge, model, robust=True)

        # Test
        n_coef = model.shape[1]
        if model_contrasts is not None:
            r_str = """
            get_model_cols <- function(design_df, design){
                m = model.matrix(object=formula(design), data=design_df)
                return(colnames(m))
            }
            """
            get_model_cols = STAP(r_str, "get_model_cols")
            model_mat_cols = get_model_cols.get_model_cols(design_df, design)
            model_df = pd.DataFrame(model)
            model_df.columns = model_mat_cols
            try:
                mod_contrast = limma.makeContrasts(contrasts=model_contrasts, levels=model_df)
            except ValueError:
                print("Model contrasts must be in the form 'A-B' or 'A+B'")
                raise
            res = base.as_data_frame(
                edgeR.topTags(edgeR.glmQLFTest(fit, contrast=mod_contrast), sort_by="none", n=np.inf)
            )
        else:
            res = base.as_data_frame(edgeR.topTags(edgeR.glmQLFTest(fit, coef=n_coef), sort_by="none", n=np.inf))
        res = conversion.rpy2py(res)
        if not isinstance(res, pd.DataFrame):
            res = pd.DataFrame(res)
        # add cpm
        cpmList=edgeR.cpm(dge)
        cpmList=pd.DataFrame(cpmList)
        #print(model_df)
        group1=np.mean(cpmList.loc[:,model_df.iloc[:,0]==1],axis=1)
        group2=np.mean(cpmList.loc[:,model_df.iloc[:,1]==1],axis=1)
        mean_df = pd.DataFrame({model_df.columns[0]: group1,
                               model_df.columns[1]: group2})
        mean_df.index = sample_adata.var_names[keep_nhoods]
        sample_adata.var["group1_cpm"]=mean_df.iloc[:,0]
        sample_adata.var["group2_cpm"]=mean_df.iloc[:,1]


        
        # Save outputs
        res.index = sample_adata.var_names[keep_nhoods]  # type: ignore
        if any(col in sample_adata.var.columns for col in res.columns):
            sample_adata.var = sample_adata.var.drop(res.columns, axis=1)
        
        sample_adata.var = pd.concat([sample_adata.var, res], axis=1)

        # Run Graph spatial FDR correction
        self._graph_spatial_fdr(sample_adata, neighbors_key=adata.uns["nhood_neighbors_key"])
        if return_fdr:
            return(sample_adata.var["SpatialFDR"])

    def annotate_nhoods(
        self,
        mdata: MuData,
        anno_col: str,
        feature_key: str | None = "rna",
    ):
        """Assigns a categorical label to neighbourhoods, based on the most frequent label among cells in each neighbourhood. This can be useful to stratify DA testing results by cell types or samples.

        Args:
            mdata: MuData object
            anno_col: Column in adata.obs containing the cell annotations to use for nhood labelling
            feature_key: If input data is MuData, specify key to cell-level AnnData object. Defaults to 'rna'.

        Returns:
            None. Adds in place:
            - `tdiff_mdata['tdiff'].var["nhood_annotation"]`: assigning a label to each nhood
            - `tdiff_mdata['tdiff'].var["nhood_annotation_frac"]` stores the fraciton of cells in the neighbourhood with the assigned label
            - `tdiff_mdata['tdiff'].varm['frac_annotation']`: stores the fraction of cells from each label in each nhood
            - `tdiff_mdata['tdiff'].uns["annotation_labels"]`: stores the column names for `tdiff_mdata['tdiff'].varm['frac_annotation']`
        """
        try:
            sample_adata = mdata["tdiff"]
        except KeyError:
            print(
                "tdiff_mdata should be a MuData object with two slots: feature_key and 'tdiff' - please run tdiffpy.count_nhoods(adata) first"
            )
            raise
        adata = mdata[feature_key]

        # Check value is not numeric
        if pd.api.types.is_numeric_dtype(adata.obs[anno_col]):
            raise ValueError(
                "adata.obs[anno_col] is not of categorical type - please use tdiffpy.utils.annotate_nhoods_continuous for continuous variables"
            )

        anno_dummies = pd.get_dummies(adata.obs[anno_col])
        anno_count = adata.obsm["nhoods"].T.dot(csr_matrix(anno_dummies.values))
        anno_count_dense = anno_count.toarray()
        anno_sum = anno_count_dense.sum(1)
        anno_frac = np.divide(anno_count_dense, anno_sum[:, np.newaxis])

        anno_frac_dataframe = pd.DataFrame(anno_frac, columns=anno_dummies.columns, index=sample_adata.var_names)
        sample_adata.varm["frac_annotation"] = anno_frac_dataframe.values
        sample_adata.uns["annotation_labels"] = anno_frac_dataframe.columns
        sample_adata.uns["annotation_obs"] = anno_col
        sample_adata.var["nhood_annotation"] = anno_frac_dataframe.idxmax(1)
        sample_adata.var["nhood_annotation_frac"] = anno_frac_dataframe.max(1)

    def annotate_nhoods_continuous(self, mdata: MuData, anno_col: str, feature_key: str | None = "rna"):
        """Assigns a continuous value to neighbourhoods, based on mean cell level covariate stored in adata.obs. This can be useful to correlate DA log-foldChanges with continuous covariates such as pseudotime, gene expression scores etc...

        Args:
            mdata: MuData object
            anno_col: Column in adata.obs containing the cell annotations to use for nhood labelling
            feature_key: If input data is MuData, specify key to cell-level AnnData object. Defaults to 'rna'.

        Returns:
            None. Adds in place:
            - `tdiff_mdata['tdiff'].var["nhood_{anno_col}"]`: assigning a continuous value to each nhood
        """
        if "tdiff" not in mdata.mod:
            raise ValueError(
                "tdiff_mdata should be a MuData object with two slots: feature_key and 'tdiff' - please run tdiffpy.count_nhoods(adata) first"
            )
        adata = mdata[feature_key]

        # Check value is not categorical
        if not pd.api.types.is_numeric_dtype(adata.obs[anno_col]):
            raise ValueError(
                "adata.obs[anno_col] is not of continuous type - please use tdiffpy.utils.annotate_nhoods for categorical variables"
            )

        anno_val = adata.obsm["nhoods"].T.dot(csr_matrix(adata.obs[anno_col]).T)

        mean_anno_val = anno_val.toarray() / np.array(adata.obsm["nhoods"].T.sum(1))

        mdata["tdiff"].var[f"nhood_{anno_col}"] = mean_anno_val

    def add_covariate_to_nhoods_var(self, mdata: MuData, new_covariates: list[str], feature_key: str | None = "rna"):
        """Add covariate from cell-level obs to sample-level obs. These should be covariates for which a single value can be assigned to each sample.

        Args:
            mdata: MuData object
            new_covariates: columns in `tdiff_mdata[feature_key].obs` to add to `tdiff_mdata['tdiff'].obs`.
            feature_key: If input data is MuData, specify key to cell-level AnnData object. Defaults to 'rna'.

        Returns:
            None, adds columns to `tdiff_mdata['tdiff']` in place
        """
        try:
            sample_adata = mdata["tdiff"]
        except KeyError:
            print(
                "tdiff_mdata should be a MuData object with two slots: feature_key and 'tdiff' - please run tdiffpy.count_nhoods(adata) first"
            )
            raise
        adata = mdata[feature_key]

        sample_col = sample_adata.uns["sample_col"]
        covariates = list(
            set(sample_adata.obs.columns[sample_adata.obs.columns != sample_col].tolist() + new_covariates)
        )
        try:
            sample_obs = adata.obs[covariates + [sample_col]].drop_duplicates()
        except KeyError:
            missing_cov = [covar for covar in covariates if covar not in sample_adata.obs.columns]
            print("Covariates {c} are not columns in adata.obs".format(c=" ".join(missing_cov)))
            raise
        sample_obs = sample_obs[covariates + [sample_col]].astype("str")
        sample_obs.index = sample_obs[sample_col]
        try:
            assert sample_obs.loc[sample_adata.obs_names].shape[0] == len(sample_adata.obs_names)
        except ValueError:
            print(
                "Covariates cannot be unambiguously assigned to each sample -- each sample value should match a single covariate value"
            )
            raise
        sample_adata.obs = sample_obs.loc[sample_adata.obs_names]

    def build_nhood_graph(self, mdata: MuData, basis: str = "X_umap", feature_key: str | None = "rna"):
        """Build graph of neighbourhoods used for visualization of DA results

        Args:
            mdata: MuData object
            basis: Name of the obsm basis to use for layout of neighbourhoods (key in `adata.obsm`). Defaults to "X_umap".
            feature_key: If input data is MuData, specify key to cell-level AnnData object. Defaults to 'rna'.

        Returns:
            - `tdiff_mdata['tdiff'].varp['nhood_connectivities']`: graph of overlap between neighbourhoods (i.e. no of shared cells)
            - `tdiff_mdata['tdiff'].var["Nhood_size"]`: number of cells in neighbourhoods
        """
        adata = mdata[feature_key]
        # # Add embedding positions
        mdata["tdiff"].varm["X_tdiff_graph"] = adata[adata.obs["nhood_ixs_refined"] == 1].obsm[basis]
        # Add nhood size
        mdata["tdiff"].var["Nhood_size"] = np.array(adata.obsm["nhoods"].sum(0)).flatten()
        # Add adjacency graph
        mdata["tdiff"].varp["nhood_connectivities"] = adata.obsm["nhoods"].T.dot(adata.obsm["nhoods"])
        mdata["tdiff"].varp["nhood_connectivities"].setdiag(0)
        mdata["tdiff"].varp["nhood_connectivities"].eliminate_zeros()
        mdata["tdiff"].uns["nhood"] = {
            "connectivities_key": "nhood_connectivities",
            "distances_key": "",
        }

    def add_nhood_expression(self, mdata: MuData, layer: str | None = None, feature_key: str | None = "rna"):
        """Calculates the mean expression in neighbourhoods of each feature.

        Args:
            mdata: MuData object
            layer: If provided, use `tdiff_mdata[feature_key][layer]` as expression matrix instead of `tdiff_mdata[feature_key].X`. Defaults to None.
            feature_key: If input data is MuData, specify key to cell-level AnnData object. Defaults to 'rna'.

        Returns:
            Updates adata in place to store the matrix of average expression in each neighbourhood in `tdiff_mdata['tdiff'].varm['expr']`
        """
        try:
            sample_adata = mdata["tdiff"]
        except KeyError:
            print(
                "tdiff_mdata should be a MuData object with two slots:"
                " feature_key and 'tdiff' - please run tdiffpy.count_nhoods(adata) first"
            )
            raise
        adata = mdata[feature_key]

        # Get gene expression matrix
        if layer is None:
            X = adata.X
            expr_id = "expr"
        else:
            X = adata.layers[layer]
            expr_id = "expr_" + layer

        # Aggregate over nhoods -- taking the mean
        nhoods_X = X.T.dot(adata.obsm["nhoods"])
        nhoods_X = csr_matrix(nhoods_X / adata.obsm["nhoods"].toarray().sum(0))
        sample_adata.varm[expr_id] = nhoods_X.T

    def _setup_rpy2(
        self,
    ):
        """Set up rpy2 to run edgeR"""
        numpy2ri.activate()
        pandas2ri.activate()
        edgeR = self._try_import_bioc_library("edgeR")
        limma = self._try_import_bioc_library("limma")
        stats = importr("stats")
        base = importr("base")

        return edgeR, limma, stats, base

    def _try_import_bioc_library(
        self,
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

    def _graph_spatial_fdr(
        self,
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



    def permute_test_window(self,
                            range_df, 
                            n=100, 
                            window_size=10, 
                            step_size=5):
        range_data = range_df.values
    
        permute_test_true_window = {}
        permute_test_false_window = {}
    
        for i in range(0, n, step_size):
            true_list = []
            false_list = []
    
            start_point = i / n
            end_point = (i + window_size) / n
    
            # Vectorized condition check
            mask = (end_point >= range_data[:, 0]) & (start_point <= range_data[:, 1])
    
            for j, (up, down, fdr) in enumerate(range_data[mask]):
                if fdr:
                    true_list.append(j)
                else:
                    false_list.append(j)
    
            permute_test_true_window[i] = len(true_list)
            permute_test_false_window[i] = len(false_list)
            length_df = pd.DataFrame([permute_test_true_window, permute_test_false_window])
            length_df=length_df.T
            length_df.columns=["true","false"]
            length_df["rate"]=length_df["true"]/(length_df["false"]+length_df["true"])
        
        return length_df



    def permute_test_point(self,
                           mdata: MuData,
                            n:int = 100,
                           include_null:bool = True,
                           times:int = 20
                          ):
        try:
            sample_adata = mdata["tdiff"]
        except KeyError:
            print(
                "tdiff_mdata should be a MuData object with two slots: feature_key and 'tdiff' - please run tdiffpy.count_nhoods(adata) first"
            )
            raise

        varTable=sample_adata.var
        if include_null:
            range_data = varTable[["range_down","range_up","Accept","logChange","null"]].values
        else:
            range_data = varTable[["range_down","range_up","Accept","logChange"]].values
    
        permute_point_value = {} # logchange
        permute_test_true_window={} # true
        permute_test_false_window={} # false 
        if include_null:
            permute_null={} # null
        for i in range(n):
            true_list = []
            false_list = []
            null_true_list=[]
            null_false_list=[]
            point=(i+1)/(n+1)
            permute_point_list=[]
            # Vectorized condition check

            mask = (point >= range_data[:, 0]) & (point <= range_data[:, 1])
            if include_null:
                for j, (up, down, fdr,logChange,null) in enumerate(range_data[mask]):
                    if fdr:
                        true_list.append(j)
                    else:
                        false_list.append(j)
                    if null>0:
                        null_true_list.append(null)
                    else:
                        null_false_list.append(j)
                    permute_point_list.append(logChange)

            else:
                for j, (up, down, fdr,logChange) in enumerate(range_data[mask]):
                    if fdr:
                        true_list.append(j)
                    else:
                        false_list.append(j)
        
                    permute_point_list.append(logChange)

    
            permute_point_list=np.array(permute_point_list)
            
            permute_test_true_window[i] = len(true_list)
            permute_test_false_window[i] = len(false_list)
            permute_point_value[i] = np.mean(permute_point_list)
            if include_null:
                permute_null[i]=np.sum(null_true_list)/(len(null_false_list)+len(null_true_list))
                length_df = pd.DataFrame([permute_test_true_window, permute_test_false_window,permute_point_value,permute_null])
                length_df=length_df.T
                length_df.columns=["true","false","meanLogChange","null"]
            else:
                length_df = pd.DataFrame([permute_test_true_window, permute_test_false_window,permute_point_value])
                length_df=length_df.T
                length_df.columns=["true","false","meanLogChange"]
            length_df["rate"]=length_df["true"]/(length_df["false"]+length_df["true"])
        length_df=self._test_binom(length_df,times=times)
        return length_df

    def make_range(self,
        mdata: MuData,
        time_col: str|None=None,
        FDR:int=0.05,
        only_range:bool =False,
        feature_key: str | None = "rna",
    ):
        try:
            sample_adata = mdata["tdiff"]
            pseudobulk=mdata["pseudobulk"]
        except KeyError:
            print(
                "tdiff_mdata should be a MuData object with two slots: feature_key and 'tdiff' - please run tdiffpy.count_nhoods(adata) first"
            )
            raise
        adata = mdata[feature_key]
        if isinstance(adata, AnnData):
            try:
                nhoods = adata.obsm["nhoods"]
            except KeyError:
                print('Cannot find "nhoods" slot in adata.obsm -- please run tdiffpy.make_nhoods(adata)')
                raise
        if time_col is None:
            time_col=pseudobulk.uns["time_col"]
        timeVal=adata.obs[time_col]
        range_df_1_list=list()
        range_df_2_list=list()
        for i in range(nhoods.shape[1]):
            filter=nhoods[:,i].toarray().flatten().astype("bool")
            dptFilter=timeVal[filter]
            bin_max=np.percentile(dptFilter,80)
            bin_min=np.percentile(dptFilter,20)
            range_df_1_list.append(bin_min)
            range_df_2_list.append(bin_max)
        sample_adata.var.index=sample_adata.var["index_cell"]
        sample_adata.var["time"]=timeVal[sample_adata.var.index]
        sample_adata.var["range_down"]=range_df_1_list
        sample_adata.var["range_up"]=range_df_2_list
        if not only_range:
            sample_adata.var["Accept"]=sample_adata.var["SpatialFDR"]<FDR
            sample_adata.var["logChange"]=sample_adata.var["logCPM"]*sample_adata.var["logFC"]


    def make_da_cpm(
        self,
        mdata: MuData,
        fix_libsize=False,
        sample_column:str|None=None,
        n_jobs : int =-1
    ):
        """perform CPM in all sample
    
        Args:
            mdata: MuData object
        Returns:
            None, modifies `tdiff_mdata['tdiff']` in place, adding the results of the DA test to `.var`:
            - `logFC` stores the log fold change in cell abundance (coefficient from the GLM)
            - `PValue` stores the p-value for the QLF test before multiple testing correction
            - `SpatialFDR` stores the the p-value adjusted for multiple testing to limit the false discovery rate,
                calculated with weighted Benjamini-Hochberg procedure
        """
        try:
            tdiff=mdata["tdiff"]
        except KeyError:
            print(
                "[bold red]tdiff_mdata should be a MuData object with three slots:"
                " feature_key and 'tdiff' - please make_pseudobulk_parallel() first"
            )
            raise
        

        indexCell=tdiff.var["index_cell"]
          

        count_mat = tdiff.X.T.toarray()

        lib_size_raw = count_mat.sum(0)
        keep_smp = lib_size_raw > 0
        keep_nhoods = count_mat[:, keep_smp].sum(1) > 0
        if fix_libsize:
            lib_size=np.full_like(lib_size_raw, 1)
        else:
            lib_size=lib_size_raw.copy()    
        
        edgeR, limma, stats, base = self._setup_rpy2()

            # Fit NB-GLM
        dge = edgeR.DGEList(counts=count_mat[keep_nhoods, :][:, keep_smp], lib_size=lib_size[keep_smp])
        if fix_libsize:
            dge = edgeR.calcNormFactors(dge, method="none")
        else:
            dge = edgeR.calcNormFactors(dge, method="TMM")
        cpmList=edgeR.cpm(dge)
        cpmList=pd.DataFrame(cpmList)
        if sample_column is None:
            sample_col=tdiff.uns["sample_col"]
        else:
            sample_col=sample_column
        sampleVal=tdiff.obs[sample_col][keep_smp]
        #group=np.mean(cpmList,axis=1)
        #mean_df = pd.DataFrame({"CPM_single": group})
        cpmList.index = tdiff.var_names[keep_nhoods]
        sampleVal=sampleVal.astype("str")
        #print(sampleVal)
        cpmList.columns= sampleVal
        tdiff.varm["whole_cpm"]=cpmList
        return(cpmList)


    def test_whole(self,
                   mdata: MuData,
                     times:int = 10):
        try:
            sample_adata = mdata["tdiff"]
        except KeyError:
            print(
                "tdiff_mdata should be a MuData object with two slots: feature_key and 'tdiff' - please run tdiffpy.count_nhoods(adata) first"
            )
            raise
        varTable=sample_adata.var
        sumVal=varTable.shape[0]
        trueVal=sum(varTable["Accept"])
        nullVal=sum(varTable["null"])/sumVal
        if nullVal==0:
            nullVal=1/(sumVal*times)
        pval=1- binom.cdf(trueVal, sumVal, nullVal)
        if trueVal == 0:
            pval=1
        sample_adata.uns["overall_p_val"]=pval

    def make_pseudobulk_parallel(self,
                       mdata: MuData,
                       min_cell:int = 3,
                       feature_key: str | None = "rna",
                       sample_col: str | None = "Sample",
                       group_col: str | None = "Group",
                       time_col:str | None = "Time",
                        other_col=[],
                        shuffle:bool = False,
                        njob: int = -1):
        adata = mdata[feature_key]
        if isinstance(adata, AnnData):
            try:
                nhoods = adata.obsm["nhoods"]
            except KeyError:
                print('Cannot find "nhoods" slot in adata.obsm -- please run tdiffpy.make_nhoods(adata)')
                raise
        df_res_dict={}
        indexCell=adata.obs_names[adata.obs["nhood_ixs_refined"] == 1]
        obs_to_keep=[sample_col,group_col,time_col]
        obs_to_keep=obs_to_keep+other_col
        def process_iteration(i):
            filterBool_1=nhoods[:,i].toarray().astype(bool)
            filterObs=adata.obs[filterBool_1]
            # find cell's which sample more than NUM_OF_CELL_PER_DONOR
            subsetCell=filterObs.index[filterObs[sample_col].isin(filterObs.groupby([sample_col]).size().index[
                                                                  filterObs.groupby([sample_col]).size()>min_cell])]
            filterObj=adata[subsetCell]
            #filterObj=adata
            if isinstance(filterObj.X, np.ndarray):
                df_filter = pd.DataFrame(filterObj.X)
            else:
                df_filter = pd.DataFrame(filterObj.X.A)
            df_filter.index = filterObj.obs_names
            df_filter.columns = filterObj.var_names
            df_filter = df_filter.join(filterObj.obs[obs_to_keep])
            agg_dict = {gene: "sum" for gene in filterObj.var_names}
            for obs in obs_to_keep:
                    agg_dict[obs] = "first"
            df_filter = df_filter.groupby(sample_col).agg(agg_dict)
            df_filter.index=df_filter.index.astype(str)+"_sep_"+indexCell[i]
            df_res_dict[i]=df_filter
            return(df_filter.copy())
        
        results = Parallel(n_jobs=njob)(delayed(process_iteration)(i) for i in tqdm(range(nhoods.shape[1])))
        df_res = pd.concat(results)
        #df_res = results[0].copy()
        #for result in results[1:]:
        #    df_res = df_res._append(result)
        df_res["nhoods_index"]=[s.split("_sep_")[-1] for s in df_res.index]
        adata_res = sc.AnnData(
            df_res[adata.var_names], obs=df_res.drop(columns=adata.var_names)
        )
        adata_res.uns["sample_col"]= sample_col
        adata_res.uns["group_col"]= group_col
        adata_res.uns["time_col"]= time_col
        mdata.mod["pseudobulk"] = adata_res
        
        return(adata_res)
        
    def da_expression(
        self,
        mdata: MuData,
        design: str,
        model_contrasts: str | None = None,
        subset_samples: list[str] | None = None,
        add_intercept: bool = True,
        feature_key: str | None = "rna",
        shuffle: bool = False,
        fix_libsize=False,
        njob : int =-1
    ):
        """Performs differential abundance testing on neighbourhoods using QLF test implementation as implemented in edgeR.

        Args:
            mdata: MuData object
            design: Formula for the test, following glm syntax from R (e.g. '~ condition').
                    Terms should be columns in `tdiff_mdata[feature_key].obs`.
            model_contrasts: A string vector that defines the contrasts used to perform DA testing, following glm syntax from R (e.g. "conditionDisease - conditionControl").
                             If no contrast is specified (default), then the last categorical level in condition of interest is used as the test group. Defaults to None.
            subset_samples: subset of samples (obs in `tdiff_mdata['tdiff']`) to use for the test. Defaults to None.
            add_intercept: whether to include an intercept in the model. If False, this is equivalent to adding + 0 in the design formula. When model_contrasts is specified, this is set to False by default. Defaults to True.
            feature_key: If input data is MuData, specify key to cell-level AnnData object. Defaults to 'rna'.

        Returns:
            None, modifies `tdiff_mdata['tdiff']` in place, adding the results of the DA test to `.var`:
            - `logFC` stores the log fold change in cell abundance (coefficient from the GLM)
            - `PValue` stores the p-value for the QLF test before multiple testing correction
            - `SpatialFDR` stores the the p-value adjusted for multiple testing to limit the false discovery rate,
                calculated with weighted Benjamini-Hochberg procedure
        """
        try:
            if shuffle:
                sample_adata = mdata["pseudobulk"].copy()
            else:
                sample_adata = mdata["pseudobulk"]
        except KeyError:
            print(
                "[bold red]tdiff_mdata should be a MuData object with three slots:"
                " feature_key and 'tdiff' - please make_pseudobulk_parallel() first"
            )
            raise
        adata = mdata[feature_key]
        tdiff=mdata["tdiff"]
        if isinstance(adata, AnnData):
            try:
                nhoods = adata.obsm["nhoods"]
            except KeyError:
                print('Cannot find "nhoods" slot in adata.obsm -- please run tdiffpy.make_nhoods(adata)')
                raise
        
        indexCell=adata.obs_names[adata.obs["nhood_ixs_refined"] == 1]
        covariates = [x.strip(" ") for x in set(re.split("\\+|\\*", design.lstrip("~ ")))]
        if shuffle:
            seed = np.random.randint(0, 100000)  
            np.random.seed(seed)  
            shuffled_array = np.random.permutation(sample_adata.obs[covariates].values)
            sample_adata.obs[covariates]=shuffled_array
            
        def da(i):
            #print(f"the first i {i}")
            subadata=sample_adata[sample_adata.obs["nhoods_index"]==indexCell[i]]
            try:
                design_df = subadata.obs[covariates]
            except KeyError:
                missing_cov = [x for x in covariates if x not in subadata.obs.columns]
                print('Covariates {c} are not columns in adata.uns["sample_adata"].obs'.format(c=" ".join(missing_cov)))
                raise
            for j in range(len(covariates)):
                if (len(design_df.iloc[:,j].unique()))==1:
                    return()
                if (len(design_df.iloc[:,j])<3):
                    return()
            if len(covariates)>1:
                crossTab=pd.crosstab(design_df[covariates[0]],design_df[covariates[1]])
                if (crossTab==0).any().any():
                    return()
            count_mat = subadata.X.T.toarray()
            lib_size_raw = count_mat.sum(0)
            keep_smp = lib_size_raw > 0
            keep_nhoods = count_mat[:, keep_smp].sum(1) > 0
            if fix_libsize:
                lib_size=np.full_like(lib_size_raw, 1)
            else:
                lib_size=lib_size_raw.copy()
            for k, e in enumerate(design_df.columns):
                if design_df.dtypes[k].name == "category":
                    design_df.loc[:, e] = design_df[e].cat.remove_unused_categories()
        # Subset samples
            if subset_samples is not None:
                keep_smp = keep_smp & sample_adata.obs_names.isin(subset_samples)
                design_df = design_df[keep_smp]
                for k, e in enumerate(design_df.columns):
                    if design_df.dtypes[k].name == "category":
                        design_df.loc[:, e]  = design_df[e].cat.remove_unused_categories()
            #print(design_df)
            design_df = design_df[keep_smp]
            #print(design_df.columns)
        # Filter out nhoods with zero counts (they can appear after sample filtering)
            #keep_nhoods = count_mat[:, keep_smp].sum(1) > 0

                    # Set up rpy2 to run edgeR
            edgeR, limma, stats, base = self._setup_rpy2()
    
                # Define model matrix
            if not add_intercept or model_contrasts is not None:
                designVal = design + " + 0"
            else:
                designVal=design
            model = stats.model_matrix(object=stats.formula(designVal), data=design_df)
            #print(model.shape[0]==count_mat[keep_nhoods, :][:, keep_smp].shape[1])
                # Fit NB-GLM
            dge = edgeR.DGEList(counts=count_mat[keep_nhoods, :][:, keep_smp], lib_size=lib_size[keep_smp])
            if fix_libsize:
                dge = edgeR.calcNormFactors(dge, method="none")
            else:
                dge = edgeR.calcNormFactors(dge, method="TMM")
            dge = edgeR.estimateDisp(dge, model)
            fit = edgeR.glmQLFit(dge, model, robust=True)
    
            # Test
            n_coef = model.shape[1]
            if model_contrasts is not None:
                r_str = """
                get_model_cols <- function(design_df, designVal){
                    m = model.matrix(object=formula(designVal), data=design_df)
                    return(colnames(m))
                }
                """
                get_model_cols = STAP(r_str, "get_model_cols")
                model_mat_cols = get_model_cols.get_model_cols(design_df, designVal)
                model_df = pd.DataFrame(model)
                model_df.columns = model_mat_cols
                try:
                    mod_contrast = limma.makeContrasts(contrasts=model_contrasts, levels=model_df)
                except ValueError:
                    print("Model contrasts must be in the form 'A-B' or 'A+B'")
                    raise
                #print(mod_contrast) # test1
                res = base.as_data_frame(
                    edgeR.topTags(edgeR.glmQLFTest(fit, contrast=mod_contrast), sort_by="none", n=np.inf)
                )
            else:
                res = base.as_data_frame(edgeR.topTags(edgeR.glmQLFTest(fit, coef=n_coef), sort_by="none", n=np.inf))
            res = conversion.rpy2py(res)
            if not isinstance(res, pd.DataFrame):
                res = pd.DataFrame(res)
            
        # Save outputs
            res.index = sample_adata.var_names[keep_nhoods]  # type: ignore
            #print(f"index{indexCell[i]}and i{i}")
            res.columns=res.columns+"_sep_"+indexCell[i]

            cpmList=edgeR.cpm(dge)
            cpmList=pd.DataFrame(cpmList)
            #print(model_df)
            group1=np.mean(cpmList.loc[:,model_df.iloc[:,0]==1],axis=1)
            group2=np.mean(cpmList.loc[:,model_df.iloc[:,1]==1],axis=1)
            mean_df = pd.DataFrame({model_df.columns[0]: group1,
                                   model_df.columns[1]: group2})
            mean_df.index = sample_adata.var_names[keep_nhoods]

            mean_df.columns=mean_df.columns+"_sep_"+indexCell[i]                
            return(res, mean_df)
        print("Using edgeR to find DEG......")
        #results = joblib.Parallel(n_jobs=njob)(joblib.delayed(da)(i) for i in tqdm(range(10)))
        results = Parallel(n_jobs=njob)(delayed(da)(i) for i in tqdm(range(nhoods.shape[1])))
        res_df = pd.DataFrame()
        res_df_cpm=pd.DataFrame()
        # Merge DataFrames from the dictionary one by one, handling None values

        for df in results:
            if (len(df)==2):
                if isinstance(df[0], pd.DataFrame):
                    res_df = res_df.merge(df[0], left_index=True, right_index=True, how="outer")
                    res_df_cpm = res_df_cpm.merge(df[1], left_index=True, right_index=True, how="outer")
              
        if shuffle:
            varDf=pd.DataFrame(index=sample_adata.var_names)
            res_df=pd.merge(varDf,res_df, left_index=True, right_index=True, how='left')
            colName=res_df.columns
            attr1=[s.split("_sep_")[0] for s in colName]
            attr2=[s.split("_sep_")[1] for s in colName]
            logic=[s=="PValue" for s in attr1]
            p_df=res_df.iloc[:,logic]
            sample_adata.varm["null_PValue"]=p_df
            indexCell = list(compress(attr2, logic))
            sample_adata.varm["null_PValue"].columns=indexCell
            return(p_df)
        else:
            varDf=pd.DataFrame(index=sample_adata.var_names)
            res_df=pd.merge(varDf,res_df, left_index=True, right_index=True, how='left')
            colName=res_df.columns
            attr1=[s.split("_sep_")[0] for s in colName]
            attr2=[s.split("_sep_")[1] for s in colName]
            logic=[s=="PValue" for s in attr1]
            p_df=res_df.iloc[:,logic]
            sample_adata.varm["PValue"]=p_df
            logFC_logic=[s=="logFC" for s in attr1]
            logFC_df=res_df.iloc[:,logFC_logic]
            sample_adata.varm["logFC"]=logFC_df
            # CPM
            logCPM_logic=[s=="logCPM" for s in attr1]
            logCPM_df=res_df.iloc[:,logCPM_logic]
            sample_adata.varm["logCPM"]=logCPM_df
            # FDR
            fdr_logic=[s=="FDR" for s in attr1]
            fdr_df=res_df.iloc[:,fdr_logic]
            sample_adata.varm["FDR"]=fdr_df
            # F
            f_logic=[s=="F" for s in attr1]
            f_df=res_df.iloc[:,f_logic]
            sample_adata.varm["F"]=f_df
            
            indexCell = list(compress(attr2, logic))
            sample_adata.varm["F"].columns=indexCell
            sample_adata.varm["PValue"].columns=indexCell
            sample_adata.varm["logFC"].columns=indexCell
            sample_adata.varm["FDR"].columns=indexCell
            sample_adata.varm["logCPM"].columns=indexCell
            # add CPM
            varDf=pd.DataFrame(index=sample_adata.var_names)
            res_df_cpm = pd.merge(varDf,res_df_cpm, left_index=True, right_index=True, how='left')
            colName=res_df_cpm.columns
            attr1=[s.split("_sep_")[0] for s in colName]
            attr2=[s.split("_sep_")[1] for s in colName]
            
            var1=np.unique(attr1)[0]
            logic1=[s==var1 for s in attr1]
            group1=res_df_cpm.iloc[:,logic1]
            var2=np.unique(attr1)[1]
            logic2=[s==var2 for s in attr1]
            group2=res_df_cpm.iloc[:,logic2]
            sample_adata.varm[var1]=group1
            sample_adata.varm[var2]=group2
            
    
            indexCell = list(compress(attr2, logic2))
            tdiff.var.index=tdiff.var["index_cell"]
            sample_adata.varm[var1].columns=indexCell
            sample_adata.varm[var2].columns=indexCell
            sample_adata.uns["var1"]=var1
            sample_adata.uns["var2"]=var2
            varDf=pd.DataFrame(index=tdiff.var_names)
            tdiff.varm[var1]=self.mergeVar(varDf,group1.T)
            tdiff.varm[var2]=self.mergeVar(varDf,group2.T)
            
            
    def makeSPFDR(self,
                  mdata: MuData,
                 p_df= None,
                  njob: int = -1,
                 shuffle: bool=False):
        pseudobulk = mdata["pseudobulk"]
        if p_df is None:
            p_df=pseudobulk.varm["PValue"].T
        else:
            p_df=p_df.T

        indexCell = p_df.index
        def process_column(i):
            pvalues = p_df.iloc[:, i]
            pvalues.index=indexCell
            keep_nhoods = ~pvalues.isna()
            o = pvalues[keep_nhoods].argsort()
            pvalues = pvalues[keep_nhoods][o]
            keep_nhoods.index=indexCell
            w_ = w[keep_nhoods][o]
            adjp = np.zeros(shape=len(o))
            adjp[o] = (sum(w_) * pvalues / np.cumsum(w_))[::-1].cummin()[::-1]
            adjp = np.array([x if x < 1 else 1 for x in adjp])
            varIndex = p_df.columns[i]
            return keep_nhoods, varIndex, adjp
        # Create an empty DataFrame with the desired index and columns

        spFDRDf = pd.DataFrame(index=indexCell, columns=p_df.columns)

        # retrive kth_distance from mdata
        tdiff = mdata["tdiff"]
        tdiff.var.index=tdiff.var["index_cell"]
        w = 1 / tdiff.var["kth_distance"]
        # to avoid missing index cell
        w=w.loc[list(indexCell)]
        w.index=indexCell
        w[np.isinf(w)] = 0
        # Define the number of parallel jobs to use
        num_jobs = njob  # Use all available cores, adjust as needed
        print("add spatial FDR......")
        # Use Parallel from joblib to parallelize the processing
        results = Parallel(n_jobs=njob)(delayed(process_column)(i) for i in tqdm(range(p_df.shape[1])))
        
        # Extract results and fill the DataFrame
        for keep_nhoods, varIndex, adjp in results:
            spFDRDf.loc[keep_nhoods, varIndex] = adjp
        
        # Convert the DataFrame entries to numeric values
        spFDRDf = spFDRDf.apply(pd.to_numeric, errors='coerce')
        spFDRDf.index= "SpatialFDR_sep_" + spFDRDf.index
        spFDRDf=spFDRDf.T
        spFDRDf.columns=indexCell
        if shuffle:
            return(spFDRDf)
        else:
            pseudobulk.varm["SPFDR"]=spFDRDf
        return(spFDRDf)

    
    def makeShuffleDA(self,
                     mdata:Mudata,
                     design: str,
                     times: int = 3,
                     model_contrasts: str | None = None,
                    subset_samples: list[str] | None = None,
                    add_intercept: bool = True,
                    feature_key: str | None = "rna",
                    FDR_threshold:int= 0.05,
                    fix_libsize=False,
                    njob : int =-1
                     ):
        res_null_Dict={}
        try:
            pseudobulk = mdata["pseudobulk"]
        except KeyError:
            print(
                "tdiff_mdata should be a MuData object with three slots: feature_key and 'pseudobulk' - please run tdiffpy.count_nhoods(adata) first"
            )
            raise
        for i in range(times):
            print(f"working on {i} times")
            res_null=self.da_expression(mdata,design=design,
                                        model_contrasts=model_contrasts,
                                        add_intercept=add_intercept,
                                       feature_key=feature_key,
                                       shuffle=True,
                                       fix_libsize= fix_libsize,
                                       njob=njob)
            print(f"Making FDR")
            res_null_Dict[i]= self.makeSPFDR(mdata=mdata,
                 p_df= res_null,
                 njob= njob,
                 shuffle= True) 
        filtered_dfs = [df < FDR_threshold for df in res_null_Dict.values()]
        df_mean = pd.concat(filtered_dfs).groupby(level=0).mean()
        varDf=pd.DataFrame(index=pseudobulk.var_names)
        #df_mean = pd.concat(res_null_Dict.values()).groupby(level=0).mean()
        pseudobulk.varm["null_mean"]=self.mergeVar(varDf,df_mean)
        pseudobulk.uns["shuffle_times"]=times
        #pseudobulk.var=pd.merge(pseudobulk.var,df_mean,left_index=True,right_index=True,how="left")
        return(df_mean)
    def mergeVar(self,
                 varTable,
                table):
        return(pd.merge(varTable,table,left_index=True,right_index=True,how="left"))

    def make_range_gene(self,
        mdata: MuData,
        time_col: str,
        FDR_threshold:int=0.05,
        feature_key: str | None = "rna",
    ):
        try:
            sample_adata = mdata["tdiff"]
        except KeyError:
            print(
                "tdiff_mdata should be a MuData object with two slots: feature_key and 'tdiff' - please run tdiffpy.count_nhoods(adata) first"
            )
            raise
        pseudobulk=mdata["pseudobulk"]
        adata = mdata[feature_key]
        if isinstance(adata, AnnData):
            try:
                nhoods = adata.obsm["nhoods"]
            except KeyError:
                print('Cannot find "nhoods" slot in adata.obsm -- please run tdiffpy.make_nhoods(adata)')
                raise
        timeVal=adata.obs[time_col]
        range_df_1_list=list()
        range_df_2_list=list()
        for i in range(nhoods.shape[1]):
            filter=nhoods[:,i].toarray().flatten().astype("bool")
            dptFilter=timeVal[filter]
            bin_max=np.percentile(dptFilter,80)
            bin_min=np.percentile(dptFilter,20)
            range_df_1_list.append(bin_min)
            range_df_2_list.append(bin_max)

        sample_adata.var.index=sample_adata.var["index_cell"]
        sample_adata.var["time"]=timeVal[sample_adata.var.index]
        sample_adata.var["range_down"]=range_df_1_list
        sample_adata.var["range_up"]=range_df_2_list
        varDf=pd.DataFrame(index=sample_adata.var_names)
        spFDR=self.mergeVar(varDf,pseudobulk.varm["SPFDR"].T)
        sample_adata.varm["Accept"]=spFDR<FDR_threshold
        logExp=(pseudobulk.varm["logCPM"]*pseudobulk.varm["logFC"]).T
        sample_adata.varm["logChange"]=self.mergeVar(varDf,logExp)
        sample_adata.varm["null_mean"]=self.mergeVar(varDf,pseudobulk.varm["null_mean"].T)
    def test_whole_gene(self,
                   mdata: MuData):
        try:
            tdiff = mdata["tdiff"]
        except KeyError:
            print(
                "tdiff_mdata should be a MuData object with two slots: feature_key and 'tdiff' - please run tdiffpy.count_nhoods(adata) first"
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


    def permute_point_gene(self,
                           mdata: MuData,
                            n:int = 100,
                          ):
        try:
            sample_adata = mdata["tdiff"]
            pseudobulk = mdata["pseudobulk"]
        except KeyError:
            print(
                "tdiff_mdata should be a MuData object with three slots: feature_key and 'tdiff' and 'pseudobulk' - please run tdiffpy.count_nhoods(adata) first"
            )
            raise
        varTable=sample_adata.var
        range_data=varTable[["range_down","range_up"]].values
        acceptTable=sample_adata.varm["Accept"]
        nullTable=sample_adata.varm["null_mean"]
        logExpTable=sample_adata.varm["logChange"]
        permute_point_value = {} # logchange
        permute_test_true_window={} # true
        permute_null={} # null
        permute_exp={} # exp
        length_dict={}
        # add cpm
        var1=pseudobulk.uns["var1"]
        group1=sample_adata.varm[var1]
        var2=pseudobulk.uns["var2"]
        group2=sample_adata.varm[var2]   
        permute_point_group1 = {}
        permute_point_group2 = {}
        
        for i in range(n):
            point=(i+1)/(n+1)

            mask = (point >= range_data[:, 0]) & (point <= range_data[:, 1])
            acceptArray=acceptTable.loc[mask,:]
            acceptSum=np.sum(acceptArray,axis=0)
            nullArray=nullTable.loc[mask,:]
            nullSum=np.sum(nullArray,axis=0)
            expArray=logExpTable.loc[mask,:]
            expMean=np.mean(expArray,axis=0)
            permute_point_value[i]=expMean
            permute_test_true_window[i]=acceptSum
            permute_null[i]=nullSum
            length_dict[i]=sum(mask)
            ## CPM
            group1Array=group1.loc[mask,:]
            group1Mean=np.mean(group1Array,axis=0)
            group2Array=group2.loc[mask,:]
            group2Mean=np.mean(group2Array,axis=0)
            permute_point_group1[i]=group1Mean
            permute_point_group2[i]=group2Mean

        nullDf=pd.DataFrame(permute_null)
        nullDf.columns=nullDf.columns.astype("str")
        trueDf=pd.DataFrame(permute_test_true_window)
        trueDf.columns=trueDf.columns.astype("str")
        expDf=pd.DataFrame(permute_point_value)
        expDf.columns=expDf.columns.astype("str")
        sumDf=pd.DataFrame.from_dict(length_dict, orient='index', columns=['Sum'])
        sumDf.index=sumDf.index.astype("str")
        sumCol=sumDf.index[sumDf["Sum"]!=0].astype("str")
        sumCol=np.array(sumCol)
        pseudobulk.varm["nullPoint"]=nullDf[sumCol]
        pseudobulk.varm["truePoint"]=trueDf[sumCol]
        pseudobulk.varm["exprPoint"]=expDf[sumCol]
        pseudobulk.uns["sum"]=sumDf.loc[sumCol]
        # CPM
        group1Df=pd.DataFrame(permute_point_group1)
        group2Df=pd.DataFrame(permute_point_group2)
        group1Df.columns=group1Df.columns.astype("str")
        group2Df.columns=group2Df.columns.astype("str")
        pseudobulk.varm["group1_cpm"]=group1Df[pseudobulk.uns["sum"].index]
        pseudobulk.varm["group2_cpm"]=group2Df[pseudobulk.uns["sum"].index]
    
    def test_gene_binom(self,
                     mdata: Mudata):
        try:
            sample_adata = mdata["tdiff"]
            pseudobulk=mdata["pseudobulk"]
        except KeyError:
            print(
                "tdiff_mdata should be a MuData object with three slots: feature_key and 'tdiff' - please run tdiffpy.count_nhoods(adata) first"
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
        '''
        p_val_df=pd.DataFrame(index=trueTable.index,columns=trueTable.columns)
        for i in range(trueTable.shape[0]):
            for j in range(trueTable.shape[1]):
                trueVal=trueTable.iloc[i,j]
                nullVal=nullTable.iloc[i,j]
                sumVal=sumTable.iloc[j]
                if nullVal==0:
                    nullVal=1/(sumVal*times) # minimal
                p_val= 1- binom.cdf(trueVal, sumVal, nullVal)
                if trueVal == 0:
                    p_val=1
                p_val_df.iloc[i,j]=p_val
                '''
    def da_expression_overall(
        self,
        mdata: MuData,
        design: str,
        model_contrasts: str | None = None,
        subset_samples: list[str] | None = None,
        add_intercept: bool = True,
        feature_key: str | None = "rna",
        fix_libsize: bool = False
    ):
        """Performs differential expression testing on neighbourhoods using QLF test implementation as implemented in edgeR.

        Args:
            mdata: MuData object
            design: Formula for the test, following glm syntax from R (e.g. '~ condition').
                    Terms should be columns in `tdiff_mdata[feature_key].obs`.
            model_contrasts: A string vector that defines the contrasts used to perform DA testing, following glm syntax from R (e.g. "conditionDisease - conditionControl").
                             If no contrast is specified (default), then the last categorical level in condition of interest is used as the test group. Defaults to None.
            subset_samples: subset of samples (obs in `tdiff_mdata['tdiff']`) to use for the test. Defaults to None.
            add_intercept: whether to include an intercept in the model. If False, this is equivalent to adding + 0 in the design formula. When model_contrasts is specified, this is set to False by default. Defaults to True.
            feature_key: If input data is MuData, specify key to cell-level AnnData object. Defaults to 'rna'.

        Returns:
            None, modifies `tdiff_mdata['tdiff']` in place, adding the results of the DA test to `.var`:
            - `logFC` stores the log fold change in cell abundance (coefficient from the GLM)
            - `PValue` stores the p-value for the QLF test before multiple testing correction
            - `SpatialFDR` stores the the p-value adjusted for multiple testing to limit the false discovery rate,
                calculated with weighted Benjamini-Hochberg procedure
        """
        try:
            pseudobulk = mdata["pseudobulk"]
        except KeyError:
            print(
                "[bold red]tdiff_mdata should be a MuData object with two slots:"
                " feature_key and 'tdiff' - please run tdiffpy.count_nhoods() first"
            )
            raise
        adata = mdata[feature_key]

        covariates = [x.strip(" ") for x in set(re.split("\\+|\\*", design.lstrip("~ ")))]

        # Add covariates used for testing to sample_adata.var
        sample_col = pseudobulk.uns["sample_col"]
        time_col = pseudobulk.uns["time_col"]
        try:
            sample_obs = pseudobulk.obs[covariates + [sample_col] + [time_col]]
        except KeyError:
            missing_cov = [x for x in covariates if x not in sample_adata.obs.columns]
            print("Covariates {c} are not columns in pseudobulk.obs".format(c=" ".join(missing_cov)))
            raise

        # Get design dataframe
        try:
            design_df = pseudobulk.obs[covariates+[time_col]]
        except KeyError:
            missing_cov = [x for x in covariates if x not in pseudobulk.obs.columns]
            print('Covariates {c} are not columns in adata.uns["sample_adata"].obs'.format(c=" ".join(missing_cov)))
            raise
        # Get count matrix
        if isinstance(pseudobulk.X, np.ndarray):
                count_mat = pseudobulk.X.T
        else:
            count_mat = pseudobulk.X.T.toarray()
        lib_size_raw = count_mat.sum(0)
        keep_smp = lib_size_raw > 0
        if fix_libsize:
            lib_size=np.full_like(lib_size_raw, 1)
        else:
            lib_size=lib_size_raw.copy()
        # Filter out samples with zero counts
        keep_smp = lib_size > 0

        # Subset samples
        if subset_samples is not None:
            keep_smp = keep_smp & pseudobulk.obs_names.isin(subset_samples)
            design_df = design_df[keep_smp]
            for i, e in enumerate(design_df.columns):
                if design_df.dtypes[i].name == "category":
                    design_df[e] = design_df[e].cat.remove_unused_categories()

        # Filter out nhoods with zero counts (they can appear after sample filtering)
        keep_nhoods = count_mat[:, keep_smp].sum(1) > 0

        # Set up rpy2 to run edgeR
        edgeR, limma, stats, base = self._setup_rpy2()

        # Define model matrix
        if not add_intercept or model_contrasts is not None:
            design = design + " + " +  time_col +" + 0"
        else:
            design = design + " + " + time_col
        model = stats.model_matrix(object=stats.formula(design), data=design_df)

        # Fit NB-GLM
        dge = edgeR.DGEList(counts=count_mat[keep_nhoods, :][:, keep_smp], lib_size=lib_size[keep_smp])
        if fix_libsize:
            dge = edgeR.calcNormFactors(dge, method="none")
        else:
            dge = edgeR.calcNormFactors(dge, method="TMM")
        dge = edgeR.estimateDisp(dge, model)
        fit = edgeR.glmQLFit(dge, model, robust=True)

        # Test
        n_coef = model.shape[1] - 1
        if model_contrasts is not None:
            r_str = """
            get_model_cols <- function(design_df, design){
                m = model.matrix(object=formula(design), data=design_df)
                return(colnames(m))
            }
            """
            get_model_cols = STAP(r_str, "get_model_cols")
            model_mat_cols = get_model_cols.get_model_cols(design_df, design)
            model_df = pd.DataFrame(model)
            model_df.columns = model_mat_cols
            try:
                mod_contrast = limma.makeContrasts(contrasts=model_contrasts, levels=model_df)
            except ValueError:
                print("Model contrasts must be in the form 'A-B' or 'A+B'")
                raise
            res = base.as_data_frame(
                edgeR.topTags(edgeR.glmQLFTest(fit, contrast=mod_contrast), sort_by="none", n=np.inf)
            )
        else:
            res = base.as_data_frame(edgeR.topTags(edgeR.glmQLFTest(fit, coef=n_coef), sort_by="none", n=np.inf))
        res = conversion.rpy2py(res)
        if not isinstance(res, pd.DataFrame):
            res = pd.DataFrame(res)
        # Save outputs
        res.index = pseudobulk.var_names[keep_nhoods]  # type: ignore
        varDf=pd.DataFrame(index=pseudobulk.var_names)
        #df_mean = pd.concat(res_null_Dict.values()).groupby(level=0).mean()
        pseudobulk.varm["edgeR_overall"]=self.mergeVar(varDf,res)


    def make_single_cpm(
        self,
        mdata: MuData,
        feature_key: str | None = "rna",
        fix_libsize=False,
        njob : int =-1
    ):
        """permute expression matrix
    
        Args:
            mdata: MuData object
            design: Formula for the test, following glm syntax from R (e.g. '~ condition').
                    Terms should be columns in `tdiff_mdata[feature_key].obs`.
            model_contrasts: A string vector that defines the contrasts used to perform DA testing, following glm syntax from R (e.g. "conditionDisease - conditionControl").
                             If no contrast is specified (default), then the last categorical level in condition of interest is used as the test group. Defaults to None.
            subset_samples: subset of samples (obs in `tdiff_mdata['tdiff']`) to use for the test. Defaults to None.
            add_intercept: whether to include an intercept in the model. If False, this is equivalent to adding + 0 in the design formula. When model_contrasts is specified, this is set to False by default. Defaults to True.
            feature_key: If input data is MuData, specify key to cell-level AnnData object. Defaults to 'rna'.
    
        Returns:
            None, modifies `tdiff_mdata['tdiff']` in place, adding the results of the DA test to `.var`:
            - `logFC` stores the log fold change in cell abundance (coefficient from the GLM)
            - `PValue` stores the p-value for the QLF test before multiple testing correction
            - `SpatialFDR` stores the the p-value adjusted for multiple testing to limit the false discovery rate,
                calculated with weighted Benjamini-Hochberg procedure
        """
        try:
            sample_adata = mdata["pseudobulk"]
            adata = mdata[feature_key]
            tdiff=mdata["tdiff"]
        except KeyError:
            print(
                "[bold red]tdiff_mdata should be a MuData object with three slots:"
                " feature_key and 'tdiff' - please make_pseudobulk_parallel() first"
            )
            raise
        
        if isinstance(adata, AnnData):
            try:
                nhoods = adata.obsm["nhoods"]
            except KeyError:
                print('Cannot find "nhoods" slot in adata.obsm -- please run tdiffpy.make_nhoods(adata)')
                raise
        
        indexCell=adata.obs_names[adata.obs["nhood_ixs_refined"] == 1]

            
        def da(i):
            #print(f"the first i {i}")
            subadata=sample_adata[sample_adata.obs["nhoods_index"]==indexCell[i]]
            count_mat = subadata.X.T.toarray()
            lib_size_raw = count_mat.sum(0)
            keep_smp = lib_size_raw > 0
            keep_nhoods = count_mat[:, keep_smp].sum(1) > 0
            if fix_libsize:
                lib_size=np.full_like(lib_size_raw, 1)
            else:
                lib_size=lib_size_raw.copy()    
            
            edgeR, limma, stats, base = self._setup_rpy2()
    
                # Fit NB-GLM
            dge = edgeR.DGEList(counts=count_mat[keep_nhoods, :][:, keep_smp], lib_size=lib_size[keep_smp])
            if fix_libsize:
                dge = edgeR.calcNormFactors(dge, method="none")
            else:
                dge = edgeR.calcNormFactors(dge, method="TMM")
            cpmList=edgeR.cpm(dge)
            cpmList=pd.DataFrame(cpmList)
            group=np.mean(cpmList,axis=1)
            mean_df = pd.DataFrame({"CPM_single": group})
            mean_df.index = sample_adata.var_names[keep_nhoods]
            return(mean_df)
        resDict={}
        print("Using edgeR to find CPM......")
        #results = joblib.Parallel(n_jobs=njob)(joblib.delayed(da)(i) for i in tqdm(range(10)))
        results = Parallel(n_jobs=njob)(delayed(da)(i) for i in tqdm(range(nhoods.shape[1])))

        res_df = pd.DataFrame()
        
        # Merge DataFrames from the dictionary one by one, handling None values
        for df in results:
            if isinstance(df, pd.DataFrame):
                res_df = res_df.merge(df, left_index=True, right_index=True, how="outer")
        return(res_df)
        varDf=pd.DataFrame(index=sample_adata.var_names)
        res_df=pd.merge(varDf,res_df, left_index=True, right_index=True, how='left')
        sample_adata.varm["single_cpm"]=res_df

        return(res_df)


    def permute_point_cpm(self,
                           mdata: MuData,
                           n:int = 100,
                          ):
        try:
            tdiff = mdata["tdiff"]
            pseudobulk = mdata["pseudobulk"]
        except KeyError:
            print(
                "tdiff_mdata should be a MuData object with three slots: feature_key and 'pseudobulk' - please run tdiffpy.count_nhoods(adata) first"
            )
            raise
        
        varTable=tdiff.var
        range_data=varTable[["range_down","range_up"]].values
        whole_cpm=pseudobulk.varm["whole_cpm"]
        permute_point_dict={}
        colName=whole_cpm.columns
        attr1=[s.split("_sep_")[0] for s in colName]
        attr2=[s.split("_sep_")[1] for s in colName]
        for i in set(attr2):
            logic=[s==i for s in attr2]
            cpm_i=whole_cpm.iloc[:,logic]
            cpm_i.columns=list(compress(attr1, logic))
            varDf=pd.DataFrame(index=tdiff.var_names)
            cpm_update=self.mergeVar(varDf,cpm_i.T)
            permute_point_group = {}
            for j in range(n):
                point=(j+1)/(n+1)
                mask = (point >= range_data[:, 0]) & (point <= range_data[:, 1])
                group1Array=cpm_update.loc[mask,:]
                group1Mean=np.mean(group1Array,axis=0)
                permute_point_group[j]=group1Mean
            groupDf=pd.DataFrame(permute_point_group)
            groupDf.columns=groupDf.columns.astype("str")
            permute_point_dict[i]=groupDf
        tdiff.uns["cpm_dict"]=permute_point_dict

        
    def process_attr2_value(self,attr2_value, attr1, attr2, whole_cpm, var_names, range_data, n):
        logic = [s == attr2_value for s in attr2]
        cpm_i = whole_cpm.iloc[:, logic]
        cpm_i.columns = list(compress(attr1, logic))
        varDf = pd.DataFrame(index=var_names)
        cpm_update = self.mergeVar(varDf, cpm_i.T)
    
        permute_point_group = {}
        for j in range(n):
            point = (j + 1) / (n + 1)
            mask = (point >= range_data[:, 0]) & (point <= range_data[:, 1])
            group1Array = cpm_update.loc[mask, :]
            group1Mean = np.mean(group1Array, axis=0)
            permute_point_group[j] = group1Mean
    
        groupDf = pd.DataFrame(permute_point_group)
        groupDf.columns = groupDf.columns.astype("str")
        
        return attr2_value, groupDf
    
    def permute_point_cpm_parallel(self,mdata, mode:str="DE",
                                   n: int = 100, n_jobs: int = -1):
        if mode == "DA":
            try:
                tdiff = mdata["tdiff"]
            except KeyError:
                print("tdiff_mdata should be a MuData object with three slots: feature_key and 'pseudobulk' - please run tdiffpy.count_nhoods(adata) first")
                raise
        else: 
            try:
                tdiff = mdata["tdiff"]
                pseudobulk = mdata["pseudobulk"]
            except KeyError:
                print("tdiff_mdata should be a MuData object with three slots: feature_key and 'pseudobulk' - please run tdiffpy.count_nhoods(adata) first")
                raise
    
        varTable = tdiff.var
        range_data = varTable[["range_down", "range_up"]].values
        if mode == "DA":
            whole_cpm=tdiff.varm["whole_cpm"]
            colName = whole_cpm.columns
            var_names = tdiff.var_names
            



            
        whole_cpm = pseudobulk.varm["whole_cpm"]
        colName = whole_cpm.columns
        attr1 = [s.split("_sep_")[0] for s in colName]
        attr2 = [s.split("_sep_")[1] for s in colName]
        var_names = tdiff.var_names
    
        permute_point_dict = {}
    
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.process_attr2_value)(attr2_value, attr1, attr2, whole_cpm, var_names, range_data, n)
            for attr2_value in tqdm(set(attr2))
        )
    
        for attr2_value, groupDf in results:
            permute_point_dict[attr2_value] = groupDf
    
        tdiff.uns["cpm_dict"] = permute_point_dict
    
    def make_whole_cpm(
        self,
        mdata: MuData,
        feature_key: str | None = "rna",
        fix_libsize=False,
        sample_column:str|None=None,
        n_jobs : int =-1
    ):
        """perform CPM in all sample
    
        Args:
            mdata: MuData object
            design: Formula for the test, following glm syntax from R (e.g. '~ condition').
                    Terms should be columns in `tdiff_mdata[feature_key].obs`.
            model_contrasts: A string vector that defines the contrasts used to perform DA testing, following glm syntax from R (e.g. "conditionDisease - conditionControl").
                             If no contrast is specified (default), then the last categorical level in condition of interest is used as the test group. Defaults to None.
            feature_key: If input data is MuData, specify key to cell-level AnnData object. Defaults to 'rna'.
    
        Returns:
            None, modifies `tdiff_mdata['tdiff']` in place, adding the results of the DA test to `.var`:
            - `logFC` stores the log fold change in cell abundance (coefficient from the GLM)
            - `PValue` stores the p-value for the QLF test before multiple testing correction
            - `SpatialFDR` stores the the p-value adjusted for multiple testing to limit the false discovery rate,
                calculated with weighted Benjamini-Hochberg procedure
        """
        try:
            sample_adata = mdata["pseudobulk"]
            tdiff=mdata["tdiff"]
        except KeyError:
            print(
                "[bold red]tdiff_mdata should be a MuData object with three slots:"
                " feature_key and 'tdiff' - please make_pseudobulk_parallel() first"
            )
            raise
        

        indexCell=tdiff.var["index_cell"]
      
        def da(i):
            #print(f"the first i {i}")
            subadata=sample_adata[sample_adata.obs["nhoods_index"]==indexCell[i]]
            count_mat = subadata.X.T.toarray()
            lib_size_raw = count_mat.sum(0)
            keep_smp = lib_size_raw > 0
            keep_nhoods = count_mat[:, keep_smp].sum(1) > 0
            if fix_libsize:
                lib_size=np.full_like(lib_size_raw, 1)
            else:
                lib_size=lib_size_raw.copy()    
            
            edgeR, limma, stats, base = self._setup_rpy2()
    
                # Fit NB-GLM
            dge = edgeR.DGEList(counts=count_mat[keep_nhoods, :][:, keep_smp], lib_size=lib_size[keep_smp])
            if fix_libsize:
                dge = edgeR.calcNormFactors(dge, method="none")
            else:
                dge = edgeR.calcNormFactors(dge, method="TMM")
            cpmList=edgeR.cpm(dge)
            cpmList=pd.DataFrame(cpmList)
            if sample_column is None:
                sample_col=sample_adata.uns["sample_col"]
            else:
                sample_col=sample_column
            sampleVal=subadata.obs[sample_col][keep_smp]
            #group=np.mean(cpmList,axis=1)
            #mean_df = pd.DataFrame({"CPM_single": group})
            cpmList.index = sample_adata.var_names[keep_nhoods]
            sampleVal=sampleVal.astype("str")
            #print(sampleVal)
            cpmList.columns=indexCell[i] + "_sep_"+ sampleVal
            return(cpmList)
        resDict={}
        print("Using edgeR to find CPM......")
        #results = joblib.Parallel(n_jobs=njob)(joblib.delayed(da)(i) for i in tqdm(range(10)))
        results = Parallel(n_jobs=n_jobs)(delayed(da)(i) for i in tqdm(range(len(indexCell))))
        
        res_df = pd.DataFrame()
        
        # Merge DataFrames from the dictionary one by one, handling None values
        for df in results:
            if isinstance(df, pd.DataFrame):
                res_df = res_df.merge(df, left_index=True, right_index=True, how="outer")

        varDf=pd.DataFrame(index=sample_adata.var_names)
        res_df=pd.merge(varDf,res_df, left_index=True, right_index=True, how='left')
        sample_adata.varm["whole_cpm"]=res_df

        return(res_df)
