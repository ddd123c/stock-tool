import pandas as pd

from src.basis import basis, residual_basis, rolling_zscore, theoretical_futures_price


def test_theoretical_price_at_zero_dte_equals_spot():
    spot = pd.Series([100.0])
    dte = pd.Series([0])
    out = theoretical_futures_price(spot, 0.05, 0.02, dte)
    assert out.iloc[0] == 100.0


def test_basis():
    out = basis(pd.Series([105.0]), pd.Series([100.0]))
    assert abs(out.iloc[0] - 0.05) < 1e-12


def test_residual_basis_uses_fair_value():
    spot = pd.Series([100.0])
    futures = pd.Series([100.0])
    dte = pd.Series([365])
    fair = theoretical_futures_price(spot, 0.05, 0.05, dte)
    out = residual_basis(futures, spot, 0.05, 0.05, dte)
    assert abs(out.iloc[0] - (100.0 / fair.iloc[0] - 1.0)) < 1e-12


def test_zscore_has_nan_before_min_periods():
    out = rolling_zscore(pd.Series([1.0, 2.0, 3.0]), window=3, min_periods=3)
    assert pd.isna(out.iloc[0])
