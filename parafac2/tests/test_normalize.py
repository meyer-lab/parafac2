import anndata
from numpy.random import choice
from scipy import sparse as sp

from ..normalize import prepare_dataset


def test_normalize():
    random_matrix = sp.random(4000, 2000, density=0.03, format="csr")
    adata = anndata.AnnData(random_matrix)
    adata.obs["condition"] = choice(["A", "B", "C"], size=adata.shape[0])  # noqa: NPY002

    prepare_dataset(adata, "condition", 0.1)
