try:
    from rpy2.robjects import conversion, numpy2ri, pandas2ri
    from rpy2.robjects.packages import STAP, PackageNotInstalledError, importr
except ModuleNotFoundError:
    print(
        "[bold yellow]rpy2 is not installed. Install with [green]pip install rpy2 [yellow]to run tools with R support."
    )

def _setup_rpy2(
):
    """Set up rpy2 to run edgeR"""
    numpy2ri.activate()
    pandas2ri.activate()
    edgeR = _try_import_bioc_library("edgeR")
    limma = _try_import_bioc_library("limma")
    stats = importr("stats")
    base = importr("base")

    return edgeR, limma, stats, base

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



def _setup_RcppML(
):
    """Set up rpy2 to run edgeR"""
    numpy2ri.activate()
    pandas2ri.activate()
    RcppML = _try_import_bioc_library("RcppML")

    return RcppML

        
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