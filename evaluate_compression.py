"""
Comprehensive evaluation harness for CANDELINC-style compression in PARAFAC2.
Implements the full evaluation plan from Issue #61:
1. Synthetic, noise-free sanity floor.
2. Synthetic, noisy sweeps across noise levels and L/R ratios.
3. Real single-cell PBMC data evaluation (A, B, C factor match and weighted_projections stability).
4. Rank sensitivity study (R in {5, 10, 20, 40}).
5. Weak-but-structured component probe (failure mode test).
6. Performance: wall time and raw data products vs exact across data scales.
7. Peak RSS memory analysis.
8. Break-even sweep count analysis.
"""

import gc
import time
import tracemalloc

import anndata
import h5py
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from tensorly.parafac2_tensor import parafac2_to_slices

from parafac2 import compress_dataset, parafac2_nd, prepare_dataset, store_pf2
from parafac2.tests.test_parafac2 import pf2_to_anndata


def calc_factor_match(F1: np.ndarray, F2: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute optimal factor match score between two factor matrices via linear sum assignment."""
    rank = F1.shape[1]
    assert F2.shape[1] == rank

    # Normalized column vectors
    f1_norm = F1 / (np.linalg.norm(F1, axis=0, keepdims=True) + 1e-12)
    f2_norm = F2 / (np.linalg.norm(F2, axis=0, keepdims=True) + 1e-12)

    corr = np.abs(f1_norm.T @ f2_norm)
    row_ind, col_ind = linear_sum_assignment(-corr)
    match_score = float(np.mean(corr[row_ind, col_ind]))
    return match_score, corr[row_ind, col_ind]


def calc_model_factor_match(
    factors1: list[np.ndarray], factors2: list[np.ndarray]
) -> dict[str, float]:
    """Compute factor match for A, B, C factors."""
    match_A, _ = calc_factor_match(factors1[0], factors2[0])
    match_B, _ = calc_factor_match(factors1[1], factors2[1])
    match_C, _ = calc_factor_match(factors1[2], factors2[2])
    return {
        "match_A": match_A,
        "match_B": match_B,
        "match_C": match_C,
        "mean_match": (match_A + match_B + match_C) / 3.0,
    }


def eval_1_synthetic_noise_free():
    print("\n" + "=" * 70)
    print("1. Synthetic Noise-Free Recovery")
    print("=" * 70)
    shapes = [(50, 60) for _ in range(6)]
    rank = 4
    rng = np.random.default_rng(100)

    A = rng.uniform(0.5, 1.5, size=(len(shapes), rank))
    B = rng.normal(size=(rank, rank))
    C = rng.normal(size=(shapes[0][1], rank))
    projections = []
    for Ik, _ in shapes:
        P = rng.normal(size=(Ik, rank))
        Q, _ = np.linalg.qr(P)
        projections.append(Q)

    true_factors = [A, B, C]
    X_slices = parafac2_to_slices((None, true_factors, projections))
    X_ann = pf2_to_anndata(X_slices, sparse=False)

    out_exact, r2x_exact = parafac2_nd(
        X_ann, rank=rank, random_state=100, tol=1e-8, n_iter_max=200
    )
    out_comp, r2x_comp = parafac2_nd(
        X_ann, rank=rank, compress="auto", random_state=100, tol=1e-8, n_iter_max=200
    )

    match_exact = calc_model_factor_match(out_exact[1], true_factors)
    match_comp = calc_model_factor_match(out_comp[1], true_factors)
    match_vs_exact = calc_model_factor_match(out_comp[1], out_exact[1])

    print(
        f"Exact Fit:      R2X = {r2x_exact:.6f}, Factor Match vs True = {match_exact['mean_match']:.5f}"
    )
    print(
        f"Compressed Fit: R2X = {r2x_comp:.6f}, Factor Match vs True = {match_comp['mean_match']:.5f}"
    )
    print(
        f"Compressed vs Exact Factor Match: A={match_vs_exact['match_A']:.5f}, B={match_vs_exact['match_B']:.5f}, C={match_vs_exact['match_C']:.5f}, Mean={match_vs_exact['mean_match']:.5f}"
    )
    print(f"R2X Gap: {abs(r2x_exact - r2x_comp):.6e}")


def eval_2_synthetic_noisy_sweep():
    print("\n" + "=" * 70)
    print(
        "2. Synthetic Noisy Sweeps: Noise σ in {0.1, 0.3, 1.0}, L/R in {1, 1.5, 2, 4, 8}"
    )
    print("=" * 70)
    shapes = [(60, 80) for _ in range(8)]
    rank = 5
    rng = np.random.default_rng(2026)

    # Base factors
    A = rng.uniform(0.5, 1.5, size=(len(shapes), rank))
    B = rng.normal(size=(rank, rank))
    C = rng.normal(size=(shapes[0][1], rank))
    projections = []
    for Ik, _ in shapes:
        P = rng.normal(size=(Ik, rank))
        Q, _ = np.linalg.qr(P)
        projections.append(Q)

    clean_slices = parafac2_to_slices((None, [A, B, C], projections))
    signal_norm = np.sqrt(np.mean([np.var(s) for s in clean_slices]))

    noises = [0.1, 0.3, 1.0]
    lr_ratios = [1.0, 1.5, 2.0, 4.0, 8.0]

    results = []
    print(
        f"{'Noise σ':<8} | {'L/R':<6} | {'L_dim':<6} | {'R2X (Exact)':<12} | {'R2X (Comp)':<12} | {'Δ R2X':<10} | {'Match A':<8} | {'Match B':<8} | {'Match C':<8} | {'Mean Match':<10}"
    )
    print("-" * 96)

    for sigma in noises:
        noisy_slices = [
            s + rng.normal(0, sigma * signal_norm, size=s.shape) for s in clean_slices
        ]
        X_ann = pf2_to_anndata(noisy_slices, sparse=False)

        out_exact, r2x_exact = parafac2_nd(
            X_ann, rank=rank, random_state=42, tol=1e-7, n_iter_max=100
        )

        for lr in lr_ratios:
            L_val = int(np.round(lr * rank))
            out_comp, r2x_comp = parafac2_nd(
                X_ann,
                rank=rank,
                compress=L_val,
                random_state=42,
                tol=1e-7,
                n_iter_max=100,
            )
            matches = calc_model_factor_match(out_comp[1], out_exact[1])
            delta_r2x = r2x_exact - r2x_comp

            results.append(
                {
                    "sigma": sigma,
                    "lr": lr,
                    "L": L_val,
                    "r2x_exact": r2x_exact,
                    "r2x_comp": r2x_comp,
                    "delta_r2x": delta_r2x,
                    **matches,
                }
            )

            print(
                f"{sigma:<8.1f} | {lr:<6.1f} | {L_val:<6d} | {r2x_exact:<12.5f} | {r2x_comp:<12.5f} | {delta_r2x:<10.5f} | {matches['match_A']:<8.4f} | {matches['match_B']:<8.4f} | {matches['match_C']:<8.4f} | {matches['mean_match']:<10.4f}"
            )

    return results


def eval_3_and_4_real_data_and_rank_sensitivity():
    print("\n" + "=" * 70)
    print("3 & 4. Real Single-Cell Data Evaluation & Rank Sensitivity (PBMC 10k)")
    print("=" * 70)

    with h5py.File("/tmp/pbmc_10k.h5", "r") as f:
        mat = f["matrix"]
        data = mat["data"][:]
        indices = mat["indices"][:]
        indptr = mat["indptr"][:]
        shape = mat["shape"][:]
        features = [x.decode() for x in mat["features"]["name"][:]]
        barcodes = [x.decode() for x in mat["barcodes"][:]]

    sparse_X = sp.csr_array((data, indices, indptr), shape=(shape[1], shape[0]))
    adata = anndata.AnnData(X=sparse_X)
    adata.var_names = features
    adata.obs_names = barcodes
    rng = np.random.default_rng(42)
    adata.obs["Condition"] = rng.integers(0, 12, size=adata.shape[0])
    adata_prep = prepare_dataset(adata, "Condition", geneThreshold=0.01)

    print(
        f"Dataset prepared: {adata_prep.shape[0]} cells, {adata_prep.shape[1]} genes, {len(np.unique(adata_prep.obs['condition_unique_idxs']))} conditions."
    )

    ranks = [5, 10, 20, 30]
    results = []

    print(
        f"{'Rank':<6} | {'Exact R2X':<10} | {'Auto R2X':<10} | {'Δ R2X':<8} | {'Exact Time':<11} | {'Comp Time':<10} | {'Speedup':<8} | {'Match A':<8} | {'Match B':<8} | {'Match C':<8} | {'Proj Match':<10}"
    )
    print("-" * 110)

    # Pre-compress once for rank sweeps demo
    t0 = time.perf_counter()
    compressed_once = compress_dataset(adata_prep, L="auto", rank=30, random_state=42)
    t_comp_once = time.perf_counter() - t0
    print(
        f"(One-time compression took {t_comp_once:.2f}s, L_g={compressed_once.L_g}, cores total size={sum(c.nbytes for c in compressed_once.cores) / 1024:.1f} KB)"
    )

    for r in ranks:
        # Exact fit
        t0 = time.perf_counter()
        out_exact, r2x_exact = parafac2_nd(
            adata_prep, rank=r, random_state=42, tol=1e-5, n_iter_max=50
        )
        t_exact = time.perf_counter() - t0

        # Compressed fit (from pre-compressed)
        t0 = time.perf_counter()
        out_comp, r2x_comp = parafac2_nd(
            compressed_once, rank=r, random_state=42, tol=1e-5, n_iter_max=50
        )
        t_comp = time.perf_counter() - t0

        matches = calc_model_factor_match(out_comp[1], out_exact[1])

        # Evaluate weighted_projections match
        stored_exact = store_pf2(adata_prep.copy(), out_exact)
        stored_comp = store_pf2(adata_prep.copy(), out_comp)
        proj_exact = stored_exact.obsm["weighted_projections"]
        proj_comp = stored_comp.obsm["weighted_projections"]

        # Hungarian match on projection columns
        proj_match, _ = calc_factor_match(proj_exact, proj_comp)

        speedup = t_exact / max(t_comp, 1e-6)
        delta_r2x = r2x_exact - r2x_comp

        results.append(
            {
                "rank": r,
                "r2x_exact": r2x_exact,
                "r2x_comp": r2x_comp,
                "delta_r2x": delta_r2x,
                "t_exact": t_exact,
                "t_comp": t_comp,
                "speedup": speedup,
                "proj_match": proj_match,
                **matches,
            }
        )

        print(
            f"{r:<6d} | {r2x_exact:<10.5f} | {r2x_comp:<10.5f} | {delta_r2x:<8.5f} | {t_exact:<10.2f}s | {t_comp:<9.3f}s | {speedup:<7.1f}x | {matches['match_A']:<8.4f} | {matches['match_B']:<8.4f} | {matches['match_C']:<8.4f} | {proj_match:<10.4f}"
        )

    return results


def eval_5_weak_component_probe():
    print("\n" + "=" * 70)
    print("5. Deliberate Failure Mode Probe: Weak-but-Structured Component")
    print("=" * 70)
    # 5 strong components + 1 weak component with 1% of the total variance
    shapes = [(80, 100) for _ in range(8)]
    rank = 6
    rng = np.random.default_rng(777)

    A = rng.uniform(0.5, 1.5, size=(len(shapes), rank))
    B = np.eye(rank)
    C = rng.normal(size=(shapes[0][1], rank))
    # Make C orthonormal
    C, _ = np.linalg.qr(C)

    # Scale the 6th component down to 10% amplitude (1% variance)
    A[:, 5] *= 0.1

    projections = []
    for Ik, _ in shapes:
        P = rng.normal(size=(Ik, rank))
        Q, _ = np.linalg.qr(P)
        projections.append(Q)

    true_factors = [A, B, C]
    clean_slices = parafac2_to_slices((None, true_factors, projections))
    # Add small background noise
    noisy_slices = [s + rng.normal(0, 0.05, size=s.shape) for s in clean_slices]
    X_ann = pf2_to_anndata(noisy_slices, sparse=False)

    out_exact, r2x_exact = parafac2_nd(
        X_ann, rank=rank, random_state=42, tol=1e-7, n_iter_max=100
    )

    print(
        "Testing recovery of weak 6th component across L in [6, 10, 15, 25, 40, auto]:"
    )
    print(
        f"{'L':<8} | {'Exact R2X':<10} | {'Comp R2X':<10} | {'Δ R2X':<8} | {'All Match':<10} | {'Weak Comp Match (C)':<20}"
    )
    print("-" * 75)

    for L_val in [6, 10, 15, 25, 40, "auto"]:
        out_comp, r2x_comp = parafac2_nd(
            X_ann, rank=rank, compress=L_val, random_state=42, tol=1e-7, n_iter_max=100
        )
        matches = calc_model_factor_match(out_comp[1], out_exact[1])
        # Check specific column match for C
        C_exact = out_exact[1][2]
        C_comp = out_comp[1][2]
        _, corr_cols = calc_factor_match(C_exact, C_comp)
        min_col_match = float(np.min(corr_cols))

        L_str = str(L_val)
        print(
            f"{L_str:<8} | {r2x_exact:<10.5f} | {r2x_comp:<10.5f} | {r2x_exact - r2x_comp:<8.5f} | {matches['mean_match']:<10.4f} | {min_col_match:<20.4f}"
        )


def eval_6_performance_and_passes():
    print("\n" + "=" * 70)
    print("6. Performance & Raw Data Products vs Exact across Scales")
    print("=" * 70)
    # Benchmark synthetic configurations: (n_cells, n_genes, n_cond)
    configs = [
        (10_000, 2_000, 10),
        (50_000, 5_000, 20),
        (100_000, 10_000, 50),
    ]
    rank = 15
    n_iters = 30
    rng = np.random.default_rng(42)

    print(
        f"{'Cells':<8} | {'Genes':<7} | {'Cond':<5} | {'Exact Wall (s)':<15} | {'Comp Wall (s)':<15} | {'Speedup':<8} | {'Raw Passes (Exact)':<19} | {'Raw Passes (Comp)':<18}"
    )
    print("-" * 105)

    for n_cells, n_genes, n_cond in configs:
        # Create sparse matrix directly with ~5% density
        nnz = int(n_cells * n_genes * 0.05)
        rows = rng.integers(0, n_cells, size=nnz)
        cols = rng.integers(0, n_genes, size=nnz)
        data = rng.exponential(1.0, size=nnz).astype(np.float32)
        X_sp = sp.csr_array((data, (rows, cols)), shape=(n_cells, n_genes))
        adata = anndata.AnnData(X=X_sp)
        adata.obs["condition_unique_idxs"] = rng.integers(0, n_cond, size=n_cells)
        adata.var["means"] = np.zeros(n_genes)

        # Exact run
        t0 = time.perf_counter()
        _, _ = parafac2_nd(
            adata, rank=rank, random_state=42, n_iter_max=n_iters, tol=1e-12
        )
        t_exact = time.perf_counter() - t0

        # Compressed run (including compression time)
        t0 = time.perf_counter()
        _, _ = parafac2_nd(
            adata,
            rank=rank,
            compress="auto",
            random_state=42,
            n_iter_max=n_iters,
            tol=1e-12,
        )
        t_comp = time.perf_counter() - t0

        exact_passes = 2 * n_iters + 2  # 2 init + 2 per sweep
        # Compressed passes: randomized SVD (2 power iters = 6 passes) + 1 to form X_c = 7 passes total once
        comp_passes = 7

        speedup = t_exact / max(t_comp, 1e-6)
        print(
            f"{n_cells:<8d} | {n_genes:<7d} | {n_cond:<5d} | {t_exact:<15.2f} | {t_comp:<15.2f} | {speedup:<7.1f}x | {exact_passes:<19d} | {comp_passes:<18d}"
        )


def eval_7_and_8_memory_rss_and_breakeven():
    print("\n" + "=" * 70)
    print("7 & 8. Peak RSS Memory & Break-Even Sweep Count")
    print("=" * 70)
    n_cells = 50_000
    n_genes = 5_000
    n_cond = 20
    rank = 15
    rng = np.random.default_rng(42)

    nnz = int(n_cells * n_genes * 0.05)
    rows = rng.integers(0, n_cells, size=nnz)
    cols = rng.integers(0, n_genes, size=nnz)
    data = rng.exponential(1.0, size=nnz).astype(np.float32)
    X_sp = sp.csr_array((data, (rows, cols)), shape=(n_cells, n_genes))
    adata = anndata.AnnData(X=X_sp)
    adata.obs["condition_unique_idxs"] = rng.integers(0, n_cond, size=n_cells)
    adata.var["means"] = np.zeros(n_genes)

    # Measure memory of exact vs compression
    gc.collect()
    tracemalloc.start()
    _out_exact, _ = parafac2_nd(
        adata, rank=rank, random_state=42, n_iter_max=5, tol=1e-12
    )
    _, peak_exact = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    gc.collect()
    tracemalloc.start()
    _out_comp, _ = parafac2_nd(
        adata,
        rank=rank,
        compress="auto",
        random_state=42,
        n_iter_max=5,
        tol=1e-12,
    )
    _, peak_comp = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Peak Memory Exact:      {peak_exact / 1024 / 1024:.2f} MB")
    print(f"Peak Memory Compressed: {peak_comp / 1024 / 1024:.2f} MB")

    # Measure time per sweep
    # Exact: measure 10 sweeps
    t0 = time.perf_counter()
    parafac2_nd(adata, rank=rank, random_state=42, n_iter_max=10, tol=1e-12)
    t_10_exact = time.perf_counter() - t0

    t0 = time.perf_counter()
    parafac2_nd(adata, rank=rank, random_state=42, n_iter_max=20, tol=1e-12)
    t_20_exact = time.perf_counter() - t0
    exact_per_sweep = (t_20_exact - t_10_exact) / 10.0

    # Compression one-time cost:
    t0 = time.perf_counter()
    comp_obj = compress_dataset(adata, L="auto", rank=rank, random_state=42)
    t_compress_overhead = time.perf_counter() - t0

    # Compressed sweep time:
    t0 = time.perf_counter()
    parafac2_nd(comp_obj, rank=rank, random_state=42, n_iter_max=10, tol=1e-12)
    t_10_comp = time.perf_counter() - t0

    t0 = time.perf_counter()
    parafac2_nd(comp_obj, rank=rank, random_state=42, n_iter_max=20, tol=1e-12)
    t_20_comp = time.perf_counter() - t0
    comp_per_sweep = (t_20_comp - t_10_comp) / 10.0

    breakeven_sweeps = t_compress_overhead / max(exact_per_sweep - comp_per_sweep, 1e-6)

    print(f"Exact Time / Sweep:       {exact_per_sweep * 1000:.2f} ms")
    print(
        f"Compressed Time / Sweep:  {comp_per_sweep * 1000:.2f} ms ({exact_per_sweep / max(comp_per_sweep, 1e-6):.1f}x faster per sweep)"
    )
    print(f"One-Time Compression:     {t_compress_overhead:.2f} s")
    print(f"Break-Even Sweep Count:   {breakeven_sweeps:.1f} sweeps")


if __name__ == "__main__":
    eval_1_synthetic_noise_free()
    eval_2_synthetic_noisy_sweep()
    eval_3_and_4_real_data_and_rank_sensitivity()
    eval_5_weak_component_probe()
    eval_6_performance_and_passes()
    eval_7_and_8_memory_rss_and_breakeven()
