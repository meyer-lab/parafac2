import anndata
import scipy as sp
from numpy.random import choice
from scipy import sparse as sps

from ..normalize import prepare_dataset


def test_normalize():
    rvs = sp.stats.poisson(25, loc=10).rvs
    random_matrix = sps.random(4000, 2000, density=0.01, format="csr", data_rvs=rvs)
    adata = anndata.AnnData(random_matrix)
    adata.obs["condition"] = choice(["A", "B", "C"], size=adata.shape[0])  # noqa: NPY002

    prepare_dataset(adata, "condition", 0.1)
