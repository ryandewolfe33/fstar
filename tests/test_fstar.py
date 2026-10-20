import pytest
import scipy.sparse as sp

import fstar


def test_result():
    c1 = sp.csr_array([
        [False, True, False, True],
        [True, True, False, False],
        [False, False, True, False],
    ])
    c2 = sp.csr_array([
        [False, True, True, True],
        [True, True, False, False],
        [True, False, False, False]
    ])
    result = fstar.fstar(c1, c2)
    assert result == pytest.approx(0.7417, abs=1e-4)


@pytest.mark.parametrize("outliers", [True, False])
@pytest.mark.parametrize("drop_outliers", [False, 'c1', 'c2', 'either', 'both'])
@pytest.mark.parametrize("alpha", [0.5, 0, 1])
def test_inputs(outliers, drop_outliers, alpha):
    c1 = sp.csr_array([
        [False, True, False, True, False],
        [True, True, False, False, False],
        [False, False, True, False, False],
    ])
    c2 = sp.csr_array([
        [False, True, True, True, False],
        [True, True, False, False, False],
        [True, False, False, False, False]
    ])
    result = fstar.fstar(c1, c2, outliers, drop_outliers, alpha)
    assert result >= 0 and result <= 1
