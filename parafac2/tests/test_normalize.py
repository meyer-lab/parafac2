import anndata
import numpy as np
import scipy as sp
from scipy import sparse as sps

from ..normalize import prepare_dataset


def test_normalize():
    rng = np.random.default_rng(42)
    rvs = sp.stats.poisson(25, loc=10).rvs
    random_matrix = sps.random(
        4000, 2000, density=0.01, format="csr", data_rvs=rvs, random_state=42
    )
    adata = anndata.AnnData(random_matrix)
    adata.obs["condition"] = rng.choice(["A", "B", "C"], size=adata.shape[0])

    out = prepare_dataset(adata, "condition", 0.001)

    # Assert expected metadata keys added
    assert "condition_unique_idxs" in out.obs
    assert "means" in out.var
    assert out.X.dtype == np.float32

    # Assert category codes and gene means validity
    assert set(out.obs["condition_unique_idxs"].unique()).issubset({0, 1, 2})
    assert len(out.var["means"]) == out.shape[1]
    assert np.all(out.var["means"] >= 0.0)

    # Assert filtering occurred as expected
    assert out.shape[0] <= 4000
    assert out.shape[1] <= 2000


def test_normalize_filtering():
    """Test that prepare_dataset filters out low-count cells and low-expressed genes."""
    # Matrix with 5 cells and 4 genes
    # Cell 0 has 0 counts (sum=0 <= 10 -> filtered)
    # Gene 3 has 0 counts across all cells -> filtered
    data = np.array(
        [
            [0, 0, 0, 0],
            [100, 200, 300, 0],
            [150, 250, 350, 0],
            [200, 300, 400, 0],
            [250, 350, 450, 0],
        ],
        dtype=np.float32,
    )
    adata = anndata.AnnData(sps.csr_matrix(data))
    adata.obs["condition"] = ["C1", "C1", "C2", "C2", "C2"]

    out = prepare_dataset(adata, "condition", geneThreshold=0.1)

    # 1 cell filtered out (cell 0), 1 gene filtered out (gene 3)
    assert out.shape == (4, 3)
    assert "condition_unique_idxs" in out.obs
    assert "means" in out.var
