# Stock Futures Arbitrage Research

Streamlit research dashboard for Taiwan single-stock futures (SSF) basis, fair-value residuals, Z-score signals, time-to-expiry effects, liquidity filters, and convergence backtests.

## Research principles

- Never treat raw futures-minus-spot as mispricing without cost-of-carry adjustment.
- Use actual executable bid/ask where available; last price alone can create false signals.
- Model dividends, financing, contract multiplier, fees, taxes, borrow costs, and slippage before calling a result arbitrage.
- Keep research signals separate from live execution.
- Avoid look-ahead bias: only use information available at signal time.

## Current modules

- `src/basis.py`: fair value, basis, residual basis, annualized basis, rolling Z-score.
- `src/signals.py`: transparent liquidity and research score.
- `src/backtest.py`: basic positive-basis convergence event study.
- `src/data.py`: market-data normalization/alignment.

## Roadmap

1. Integrate authoritative TWSE/TAIFEX historical data.
2. Add contract metadata and dividend/corporate-action adjustments.
3. Add bid/ask and executable spread modeling.
4. Add expiry-specific convergence statistics (DTE 1/2/3/5/10/20).
5. Add transaction-cost-aware backtesting and performance metrics.
6. Add Streamlit scanner and single-symbol diagnostics.
7. Add scheduled data refresh and data-quality monitoring.
