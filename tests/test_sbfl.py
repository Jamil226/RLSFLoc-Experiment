import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp
import time

from src.utils.sbfl import compute_sbfl_scores

def test_sbfl_correctness_dense():
    """
    Test SBFL scores correctness on a small, hand-calculated dense dataset.
    
    Test setup:
      - 4 test cases (2 failed, 2 passed):
        test_results = [1, 1, 0, 0]  (indices 0, 1 are failed; 2, 3 are passed)
      - 3 statements:
        s0: executed by all tests (coverage = [1, 1, 1, 1]) -> ef=2, ep=2, nf=0, np=0
        s1: executed only by failed tests (coverage = [1, 1, 0, 0]) -> ef=2, ep=0, nf=0, np=2
        s2: executed only by passed tests (coverage = [0, 0, 1, 1]) -> ef=0, ep=2, nf=2, np=0
    """
    coverage = np.array([
        [1, 1, 0],  # Test 0 (failed)
        [1, 1, 0],  # Test 1 (failed)
        [1, 0, 1],  # Test 2 (passed)
        [1, 0, 1],  # Test 3 (passed)
    ])
    results = np.array([1, 1, 0, 0])
    
    # Run computation with epsilon=1e-12
    res = compute_sbfl_scores(coverage, results, dstar_alpha=2, epsilon=1e-12)
    
    # 1. Statement 0 values:
    # ef = 2, ep = 2, F = 2, P = 2
    # Ochiai = 2 / sqrt(2 * (2 + 2)) = 2 / sqrt(8) = 2 / 2.8284271 = 0.70710678
    # Tarantula = (2/2) / ((2/2) + (2/2)) = 1 / 2 = 0.5
    # DStar = 2^2 / (2 + 0 + 1e-12) = 4 / (2 + 1e-12) = 1.999999999999
    assert np.isclose(res.loc[0, 'Ochiai'], 0.70710678, atol=1e-6)
    assert np.isclose(res.loc[0, 'Tarantula'], 0.5, atol=1e-6)
    assert np.isclose(res.loc[0, 'DStar'], 2.0, atol=1e-6)
    
    # 2. Statement 1 values (perfect predictor of failure):
    # ef = 2, ep = 0, F = 2, P = 2
    # Ochiai = 2 / sqrt(2 * (2 + 0)) = 2 / 2 = 1.0
    # Tarantula = (2/2) / ((2/2) + (0/2)) = 1 / 1 = 1.0
    # DStar = 2^2 / (0 + 0 + 1e-12) = 4e12
    assert np.isclose(res.loc[1, 'Ochiai'], 1.0, atol=1e-6)
    assert np.isclose(res.loc[1, 'Tarantula'], 1.0, atol=1e-6)
    assert np.isclose(res.loc[1, 'DStar'], 4.0e12, rtol=1e-3)
    
    # 3. Statement 2 values (perfect predictor of success/clean):
    # ef = 0, ep = 2, F = 2, P = 2
    # Ochiai = 0 / sqrt(2 * (0 + 2)) = 0.0
    # Tarantula = (0/2) / ((0/2) + (2/2)) = 0.0
    # DStar = 0^2 / (2 + 2 + 1e-12) = 0.0
    assert np.isclose(res.loc[2, 'Ochiai'], 0.0, atol=1e-6)
    assert np.isclose(res.loc[2, 'Tarantula'], 0.0, atol=1e-6)
    assert np.isclose(res.loc[2, 'DStar'], 0.0, atol=1e-6)


def test_sbfl_pandas_dataframe():
    """
    Test that pandas DataFrame input works correctly and preserves column names as statement_ids.
    """
    coverage_df = pd.DataFrame(
        [
            [1, 0],
            [1, 1],
            [0, 1]
        ],
        columns=['line_42', 'line_99']
    )
    results = [1, 0, 0]
    
    res = compute_sbfl_scores(coverage_df, results, epsilon=1e-12)
    
    assert list(res['statement_id']) == ['line_42', 'line_99']
    assert len(res) == 2
    # Column line_42: ef = 1, ep = 1, F = 1, P = 2
    # Ochiai = 1 / sqrt(1 * (1 + 1)) = 1 / sqrt(2) = 0.70710678
    assert np.isclose(res.loc[res['statement_id'] == 'line_42', 'Ochiai'].values[0], 0.70710678)


def test_sbfl_sparse_matrix():
    """
    Test that scipy sparse matrix input matches dense input exactly.
    """
    np.random.seed(42)
    num_tests = 50
    num_statements = 100
    
    # Generate random binary coverage matrix
    coverage_dense = (np.random.rand(num_tests, num_statements) > 0.7).astype(int)
    results = (np.random.rand(num_tests) > 0.6).astype(int)
    
    # Sparse versions
    coverage_sparse_csr = sp.csr_matrix(coverage_dense)
    coverage_sparse_csc = sp.csc_matrix(coverage_dense)
    
    res_dense = compute_sbfl_scores(coverage_dense, results, dstar_alpha=3, epsilon=1e-5)
    res_csr = compute_sbfl_scores(coverage_sparse_csr, results, dstar_alpha=3, epsilon=1e-5)
    res_csc = compute_sbfl_scores(coverage_sparse_csc, results, dstar_alpha=3, epsilon=1e-5)
    
    # All must match perfectly
    pd.testing.assert_frame_equal(res_dense, res_csr)
    pd.testing.assert_frame_equal(res_dense, res_csc)


def test_sbfl_strict_division():
    """
    Test that setting epsilon=0 enables strict mathematical division
    resulting in np.inf for perfect suspiciousness.
    """
    coverage = np.array([
        [1],  # failed test executions
        [0],  # passed test executions
    ])
    results = np.array([1, 0])
    
    # ef = 1, ep = 0, F = 1, P = 1, nf = 0
    # DStar denom = ep + nf = 0 + 0 = 0.
    # ef^2 / 0 should return infinity when epsilon = 0.
    res = compute_sbfl_scores(coverage, results, epsilon=0)
    
    assert res.loc[0, 'DStar'] == np.inf


def test_sbfl_corner_cases():
    """
    Test edge cases:
      - F = 0 (no failing tests)
      - P = 0 (no passing tests)
      - ef = ep = 0 (unexecuted statements)
    """
    coverage = np.array([
        [1, 0],
        [1, 0],
    ])
    
    # 1. No failed tests (all passed)
    results_all_passed = np.array([0, 0])
    res1 = compute_sbfl_scores(coverage, results_all_passed)
    assert (res1['Ochiai'] == 0.0).all()
    assert (res1['Tarantula'] == 0.0).all()
    assert (res1['DStar'] == 0.0).all()
    
    # 2. No passed tests (all failed)
    results_all_failed = np.array([1, 1])
    res2 = compute_sbfl_scores(coverage, results_all_failed)
    # For statement 0: ef = 2, ep = 0, F = 2, P = 0
    # Ochiai: 2 / sqrt(2 * (2 + 0)) = 1.0
    # Tarantula: (2/2) / ((2/2) + 0) = 1.0
    # DStar (epsilon=1e-12): 4 / (0 + 0 + 1e-12) = 4e12
    assert res2.loc[0, 'Ochiai'] == 1.0
    assert res2.loc[0, 'Tarantula'] == 1.0
    assert np.isclose(res2.loc[0, 'DStar'], 4.0e12, rtol=1e-3)
    
    # For statement 1 (unexecuted statement): ef = 0, ep = 0, F = 2, P = 0
    # Ochiai: 0.0
    # Tarantula: 0.0
    # DStar: 0.0
    assert res2.loc[1, 'Ochiai'] == 0.0
    assert res2.loc[1, 'Tarantula'] == 0.0
    assert res2.loc[1, 'DStar'] == 0.0


def test_sbfl_dimension_validation():
    """
    Verify that compute_sbfl_scores raises appropriate ValueError when dimensions mismatch.
    """
    coverage = np.ones((5, 10))
    results_bad_len = np.ones(4)
    results_bad_dim = np.ones((5, 2))
    coverage_bad_dim = np.ones((5, 10, 2))
    
    with pytest.raises(ValueError, match="Dimension mismatch"):
        compute_sbfl_scores(coverage, results_bad_len)
        
    with pytest.raises(ValueError, match="1D array-like"):
        compute_sbfl_scores(coverage, results_bad_dim)
        
    with pytest.raises(ValueError, match="2D array-like"):
        compute_sbfl_scores(coverage_bad_dim, np.ones(5))


def test_sbfl_large_project_performance():
    """
    Sanity check for optimization on large projects.
    Generates a large-scale sparse system representing:
      - 2,000 test cases
      - 10,000 statements
      - 2% density (highly sparse)
    Asserts that the execution takes less than 150 milliseconds.
    """
    num_tests = 2000
    num_statements = 10000
    density = 0.02
    
    # Generate large sparse random matrix
    np.random.seed(123)
    coverage_sparse = sp.random(num_tests, num_statements, density=density, format='csr', dtype=bool)
    results = (np.random.rand(num_tests) > 0.7).astype(int)
    
    start_time = time.perf_counter()
    res = compute_sbfl_scores(coverage_sparse, results, dstar_alpha=2, epsilon=1e-12)
    end_time = time.perf_counter()
    
    elapsed_ms = (end_time - start_time) * 1000
    print(f"\n[Performance Log] Large-scale project SBFL scoring took: {elapsed_ms:.2f} ms")
    
    assert len(res) == num_statements
    assert list(res.columns) == ['statement_id', 'Ochiai', 'Tarantula', 'DStar']
    # Must run fast (typically takes 5-15ms on modern machines, let's set a conservative 150ms limit)
    assert elapsed_ms < 150, f"SBFL scoring took too long: {elapsed_ms:.2f} ms"

def test_min_max_normalize():
    """
    Test min_max_normalize helper function matching Equations 459 and 460
    of the RLSFLoc research article.
    """
    from src.utils.sbfl import min_max_normalize
    
    scores = np.array([2.0, 4.0, 6.0, 8.0])
    
    # Normalized: (score - 2) / (8 - 2 + 1e-12) = (score - 2) / (6 + 1e-12)
    normalized = min_max_normalize(scores, epsilon=1e-12)
    
    assert np.isclose(normalized[0], 0.0)
    assert np.isclose(normalized[1], 1.0 / 3.0)
    assert np.isclose(normalized[2], 2.0 / 3.0)
    assert np.isclose(normalized[3], 1.0)
