#from .utils import getAttribute, makeDotTable, trajDotplot, split_umap
from ._env import _setup_rpy2,_try_import_bioc_library, _setup_RcppML, _detect_RcppML
#from .GEP import getAttributeGEP,attrTTest
from .attr_util import getAttributeGEP,attrTTest,getAttributeBase, getAttributeGEP_Bootstrap,attrANOVA
from .utils import makeDotTable, trajDotplot, split_umap