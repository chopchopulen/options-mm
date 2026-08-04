# options-mm — working rules

This repo is a **stochastic simulator**. Read these rules before touching anything.

## 0. `docs/FINAL_NUMBERS.md` is the single source of truth

**No number leaves this repo — into a README, a commit message, or a résumé — unless it
appears in `docs/FINAL_NUMBERS.md` first.** §(e) is authoritative over §(a)–(d) wherever they
disagree; (a)–(d) predate Phase E.

Hard bans, all established by the audit:

- **"Sharpe 1.32" has never existed at any commit.** Not in the worktree; `git log --all
  -S"1.32"` finds only an unrelated `results/sensitivity.csv` cell. If you see it anywhere,
  delete it.
- **No P&L or signal-to-noise figure may be quoted as a performance result**, in either
  configuration, at any commit — the P&L LEVEL is unidentified until a competition model runs.
- **`compute_sharpe` no longer exists.** It is `compute_pnl_signal_to_noise`
  (`src/backtest/report.py:8`) because there is no capital base anywhere in this repo.
- **The sensitivity grid's ranking carries no information** (defect #18). Never cite "best combo."
- **Say 3.1×, not 2.6×**, for the vol-informed vs spot-informed cost ratio. Commit `8d18931`'s
  message says 2.6×; its own ledger says 3.10×. See FINAL_NUMBERS §(e).
- **Never write "recovered X% informed share matching PIN."** `lambda_noise=177.8` was solved
  backwards from a 15% target inside the PIN band. It is circular.
- **Heston here is semi-analytic, not closed-form.**

## 1. Every result cites its seed

No number — P&L, Sharpe, win rate, drawdown, attribution line — may be stated without the
seed that produced it. "Total P&L is $33,838" is meaningless; "Total P&L is $33,838
(seed 42)" is a claim. Multi-seed results cite the seed list.

## 2. Never tune a parameter to improve a metric

Spread coefficients (`QUOTER.base_spread`, `gamma_coeff`, `vega_coeff`), hedge thresholds
(`HEDGER.delta_threshold`), informed arrival rates (`ORDER_FLOW.staleness_threshold`,
`lambda_noise`), risk limits — all of these can trivially inflate P&L and Sharpe on one
Heston path. They are not knobs.

A performance improvement is only legitimate if it comes from:
- a **corrected model** (wrong discretization, wrong formula, wrong sign),
- a **fixed bug**, or
- **better attribution** (the same P&L, explained more honestly).

If you find yourself adjusting a coefficient because a number looks better: **stop and flag
it to the user.** Do not make the edit.

## 3. No metric changes without a before/after vs `bench/BASELINE.md`, across MULTIPLE seeds

`bench/BASELINE.md` is frozen. It records exact seed-42 numbers, the 20-seed distribution,
the reproduction commands, and the git commit. Any change that moves a metric must ship
with a before/after table over **at least the 20 seeds in `results/multi_seed.csv`**, not
seed 42 alone.

Context for why: at baseline the 20-seed Sharpe distribution is mean `-0.14`, std `3.07`,
range `[-5.15, +4.51]`. Seed 42's `1.849` is one draw from that — and is **retired**, being
48% overstated by defect #1 and 89% day-0 artifact from defect #2. A single-seed improvement
of ±1 is **noise**, not signal.

## 4. Auditors are read-only until the user approves

Agents in `.claude/agents/*-auditor.md` and `adversarial-reviewer.md` have `Read`, `Grep`,
`Glob`, and `Bash` for running tests only. They must not `Edit` or `Write` simulator code.
The `builder` and `test-runner` agents are idle during audit phases. The user decides what
gets fixed; the architect never writes code either.

## 5. Reproducibility is a gate, not a nice-to-have

The backtest is currently **bit-reproducible** for a fixed seed across processes (verified,
see `bench/BASELINE.md`). Any change that breaks that is a defect regardless of what it does
to P&L. Known latent hazard: `src/pricing/monte_carlo.py:24-25` defaults `rng=None` →
fresh unseeded `default_rng()`. Off the backtest path today; wiring `mc_price` into the
engine without threading a seed silently breaks the gate.

## 6. Watch for numbers that don't mean what they appear to mean

- `report.py:41` reports residual as a fraction of **net** P&L only. Net is a small
  difference between large gross components. Always report the gross-flow denominator too.
- `compute_pnl_signal_to_noise` (`report.py:8`) uses population std and `np.sqrt(252)` on 30
  observations. It is a signal-to-noise ratio of a dollar stream, invariant to leverage and
  book size — **not** a Sharpe ratio. State whether an interval accompanies it.
- Attribution components are not independent; a large `residual` and a large `volga_pnl`
  can be the same modelling error twice.

## Commands

```bash
MPLBACKEND=Agg python3 -c "import configs.default as cfg; from src.backtest.engine import BacktestEngine; from src.backtest.report import print_summary; print_summary(BacktestEngine(cfg, seed=42).run())"
MPLBACKEND=Agg python3 -m src.backtest.multi_seed      # 20 seeds -> results/multi_seed.csv
MPLBACKEND=Agg python3 -m src.backtest.sensitivity     # 27-combo grid x 5 seeds -- RANKING IS RETIRED (defect #18)
python3 -m pytest -q                                   # 103 tests
python3 run_backtest.py                                # seed 42 + backtest_results.png
```

Note: `python` is not on PATH in this environment. Use `python3`.
