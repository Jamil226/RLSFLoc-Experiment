import numpy as np
import pandas as pd
import scipy.sparse as sp

def compute_sbfl_scores(coverage_matrix, test_results, dstar_alpha=2, epsilon=1e-12):
    """
    Compute Ochiai, Tarantula, and DStar suspicion scores for program statements
    using mathematically correct formulas optimized for large-scale projects.

    This function utilizes highly optimized vectorized calculations (matrix-vector dot
    products) instead of slow loops, supporting dense NumPy arrays, Pandas DataFrames,
    and SciPy sparse matrices (CSR/CSC formats) to minimize memory overhead and runtime.

    Mathematical Formulas:
    ----------------------
    Let:
      - F  = total number of failed test cases
      - P  = total number of passed test cases
      - ef = number of failed test cases that execute statement s
      - ep = number of passed test cases that execute statement s
      - nf = number of failed test cases that do NOT execute statement s (F - ef)
      - np = number of passed test cases that do NOT execute statement s (P - ep)

    1. Ochiai:
       Ochiai(s) = ef / sqrt(F * (ef + ep))

    2. Tarantula:
       Tarantula(s) = (ef / F) / ((ef / F) + (ep / P))

    3. DStar:
       DStar(s) = (ef^alpha) / (ep + nf) = (ef^alpha) / (ep + F - ef)

    Parameters:
    -----------
    coverage_matrix : array-like or scipy.sparse matrix of shape (num_tests, num_statements)
        Binary/boolean coverage matrix. 1 or True indicates the test executes the statement.
    test_results : array-like of shape (num_tests,)
        Binary/boolean outcomes of test cases. 1 or True indicates the test failed.
    dstar_alpha : float, optional (default=2)
        Exponent parameter for the DStar formula.
    epsilon : float, optional (default=1e-12)
        A small constant added to denominators to prevent division by zero in DStar
        while maintaining relative ranking. Set to 0 to use mathematically strict division
        (returns np.inf if ef > 0 and ep + nf = 0).

    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: ['statement_id', 'Ochiai', 'Tarantula', 'DStar']
        Rows represent statements and are ordered by their original input order.
    """
    # 1. Validate and normalize test results
    test_results_arr = np.asarray(test_results)
    if test_results_arr.ndim != 1:
        raise ValueError(
            f"test_results must be a 1D array-like, but got shape {test_results_arr.shape}"
        )

    num_tests = len(test_results_arr)

    # 2. Validate and normalize coverage matrix
    # Extract statement ids if Pandas DataFrame is passed
    statement_ids = None
    if isinstance(coverage_matrix, pd.DataFrame):
        if coverage_matrix.shape[0] != num_tests:
            raise ValueError(
                f"Dimension mismatch: coverage_matrix has {coverage_matrix.shape[0]} tests, "
                f"but test_results has {num_tests} tests."
            )
        statement_ids = coverage_matrix.columns.values
        # Convert to numpy array or keep sparse if already sparse elements (unlikely for DF)
        coverage_data = coverage_matrix.values
    elif sp.issparse(coverage_matrix):
        if coverage_matrix.shape[0] != num_tests:
            raise ValueError(
                f"Dimension mismatch: coverage_matrix has {coverage_matrix.shape[0]} tests, "
                f"but test_results has {num_tests} tests."
            )
        coverage_data = sp.csr_matrix(coverage_matrix)
    else:
        coverage_data = np.asarray(coverage_matrix)
        if coverage_data.ndim != 2:
            raise ValueError(
                f"coverage_matrix must be a 2D array-like, but got shape {coverage_data.shape}"
            )
        if coverage_data.shape[0] != num_tests:
            raise ValueError(
                f"Dimension mismatch: coverage_matrix has {coverage_data.shape[0]} tests, "
                f"but test_results has {num_tests} tests."
            )

    num_statements = coverage_data.shape[1]
    if statement_ids is None:
        statement_ids = np.arange(num_statements)

    # 3. Create float-based masks for failing/passing tests to perform dot products
    failed_mask = (test_results_arr == 1).astype(np.float64)
    passed_mask = (test_results_arr == 0).astype(np.float64)

    # Total failed (F) and total passed (P)
    F = np.sum(failed_mask)
    P = np.sum(passed_mask)

    # 4. Compute 'ef' and 'ep' using highly optimized matrix-vector dot products.
    # For large systems, this runs in O(nnz) time and avoids loops or matrix duplication.
    if sp.issparse(coverage_data):
        # coverage_data.T is CSC, CSC.dot(vector) is extremely fast
        ef = np.array(coverage_data.T.dot(failed_mask)).flatten()
        ep = np.array(coverage_data.T.dot(passed_mask)).flatten()
    else:
        ef = coverage_data.T.dot(failed_mask)
        ep = coverage_data.T.dot(passed_mask)

    # Wrap computations in errstate to silence warnings (e.g. division by zero for unexecuted statements)
    with np.errstate(divide='ignore', invalid='ignore'):
        # 5. Compute Ochiai suspiciousness score
        ochiai_denom = np.sqrt(F * (ef + ep))
        ochiai = np.where(ochiai_denom > 0, ef / ochiai_denom, 0.0)

        # 6. Compute Tarantula suspiciousness score
        s_f = np.where(F > 0, ef / F, 0.0)
        s_p = np.where(P > 0, ep / P, 0.0)
        tarantula_denom = s_f + s_p
        tarantula = np.where(tarantula_denom > 0, s_f / tarantula_denom, 0.0)

        # 7. Compute DStar suspiciousness score
        dstar_denom = ep + (F - ef)
        if epsilon > 0:
            dstar = (ef ** dstar_alpha) / (dstar_denom + epsilon)
        else:
            # Strict mathematical division: if denom is 0, return infinity if ef > 0, else 0.0
            dstar = np.where(
                dstar_denom > 0,
                (ef ** dstar_alpha) / dstar_denom,
                np.where(ef > 0, np.inf, 0.0)
            )

    # 8. Package results into pandas DataFrame
    results_df = pd.DataFrame({
        'statement_id': statement_ids,
        'Ochiai': ochiai,
        'Tarantula': tarantula,
        'DStar': dstar
    })

    return results_df

def min_max_normalize(scores, epsilon=1e-12):
    """
    Perform min-max normalization on a numeric array/Series as defined in Equation 459/460
    of the RLSFLoc research article:

    normalized_score = (score - min_score) / (max_score - min_score + epsilon)

    Parameters:
    -----------
    scores : array-like
        The array of scores to normalize.
    epsilon : float, optional (default=1e-12)
        A small constant to prevent division by zero.

    Returns:
    --------
    numpy.ndarray
        The min-max normalized scores.
    """
    scores_arr = np.asarray(scores)
    min_val = np.min(scores_arr)
    max_val = np.max(scores_arr)
    denom = max_val - min_val
    return (scores_arr - min_val) / (denom + epsilon)
