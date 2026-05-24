import pytest
import numpy as np
from src.statistical_validation import cliffs_delta

def test_cliffs_delta_correctness():
    """
    Test Cliff's Delta helper for accurate computation and categorical mappings.
    """
    # Case A: Identical distributions (should have d = 0, effect = Negligible)
    x = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5]
    d, effect = cliffs_delta(x, y)
    assert np.isclose(d, 0.0)
    assert effect == "Negligible"
    
    # Case B: Completely superior (d = +1.0, effect = Large)
    x_sup = [10, 20, 30]
    y_inf = [1, 2, 3]
    d_sup, effect_sup = cliffs_delta(x_sup, y_inf)
    assert np.isclose(d_sup, 1.0)
    assert effect_sup == "Large"
    
    # Case C: Completely inferior (d = -1.0, effect = Large)
    d_inf, effect_inf = cliffs_delta(y_inf, x_sup)
    assert np.isclose(d_inf, -1.0)
    assert effect_inf == "Large"
    
    # Case D: Mixed values (threshold testing)
    # x = [2, 3, 4], y = [1, 3, 5]
    # pairs:
    # 2 > 1 (+), 2 < 3 (-), 2 < 5 (-) -> net -1
    # 3 > 1 (+), 3 == 3 (0), 3 < 5 (-) -> net 0
    # 4 > 1 (+), 4 > 3 (+), 4 < 5 (-) -> net +1
    # sum of signs = -1 + 0 + 1 = 0 -> d = 0 / 9 = 0
    d_mixed, effect_mixed = cliffs_delta([2, 3, 4], [1, 3, 5])
    assert np.isclose(d_mixed, 0.0)
    assert effect_mixed == "Negligible"
