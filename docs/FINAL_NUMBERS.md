# FINAL NUMBERS — options-mm

**Read this file before quoting any number from this project.**

Repo: `/Users/harry/Desktop/options-mm`. Branch `audit/blind-audit-2026-07-26`.
Audit trail: `audit/FINDINGS.md`, `audit/RECONCILIATION.md`, `audit/ITEM8_FILL_MODEL.md`.
Frozen pre-audit baseline: `bench/BASELINE.md`.

Everything below is measured on **seeds 0–19, day 0 excluded**, unless stated otherwise. The
backtest is bit-reproducible for a fixed seed across processes; the gate is re-verified after
every change.

---

## ⚠️ The caveat that attaches to every Sharpe figure in this project

**There is no capital base anywhere in this repo.** `compute_sharpe`
(`src/backtest/report.py:8-13`) computes

```
sqrt(252) × mean(daily DOLLAR P&L) / population-std(daily DOLLAR P&L)
```

That is a **signal-to-noise ratio of a dollar P&L stream**, not a risk-adjusted return. It is
invariant to leverage and to book size: double every position and it does not move. It also
subtracts a daily rate (0.02/252) from a dollar series — dimensionally incoherent, and worth
1.3e-07 — and uses `ddof=0`, which inflates it relative to the sample statistic.

`grep -rniE "capital|equity|nav|aum|account_value|initial_cash|book_size"` returns **zero hits**
in `src/`, `configs/` or `bench/`.

**Every Sharpe number below is written `Sharpe*` and must carry this caveat verbatim.** A
percentage drawdown is likewise undefinable here for the same reason.

---

## (a) CLAIMABLE

Numbers that can be stated, with what was measured and the caveat each must carry.

### Current model state

| metric | value | measured on | caveat it must carry |
|---|---:|---|---|
| median total P&L | **−$1,179,948** | seeds 0–19, ex day 0 | Dominated by an assumed reservation-price scale that is almost certainly too tight (p_fill ≈ 1.3%). Not a measurement of strategy quality. |
| median Sharpe* | **−43.537** | seeds 0–19, ex day 0 | Not a Sharpe ratio — see above. Also inherits the caveat directly above. |
| seeds with positive P&L | **0/20** | seeds 0–19, ex day 0 | Same. |
| median residual / gross flow | **2.59%** | seeds 0–19, ex day 0 | This one is solid. The attribution identity closes and explains 97.4% of gross flow. |

### Attribution, current state (medians, seeds 0–19, ex day 0)

| component | value | caveat |
|---|---:|---|
| `spread_capture` | +$431,914 | Gross quoted spread, measured against quote-time fair. Must be read together with `adverse_selection`; their **sum** is the edge against contemporaneous fair. |
| `adverse_selection` | −$1,021,235 | Applies to informed flow. Inflated by the reservation gate screening out benign flow while toxic flow is unaffected. |
| `hedge_cost` | −$598,193 | **100% an artifact of `transaction_cost=0.001`** (10bps/hedge), which is 20–100× realistic institutional SPY execution. Never corrected — see OPEN. |
| `theta_pnl` | +$540 | |
| `gamma_pnl` | −$248 | |
| `vega_pnl` | +$3,170 | |
| `vanna_pnl` | −$244 | |
| `volga_pnl` | −$5,404 | |
| `residual` | −$10,210 | 2.59% of gross. |

### Reproducibility — claimable without qualification

| claim | evidence |
|---|---|
| A 30-day backtest is **bit-identical** across processes for a fixed seed | Full daily-P&L + attribution + price-path JSON compares byte-equal. Re-verified after every commit in this sequence. |
| Every stochastic source on the backtest path is seeded from `BacktestEngine.seed` | Heston path (`seed`), order flow (`seed+1`). `mc_price` defaults to an unseeded generator but is off the backtest path. |

### Pricing and Greeks — claimable, audited clean

| claim | measured |
|---|---|
| BS vs CRR(200) vs MC(400k) vs Heston-CF agree | CRR ≤ $0.022, MC ≤ $0.035, Heston-CF ≤ **$0.0002** of BS across K ∈ {427.5, 450, 472.5} × T ∈ {30d, 60d} |
| Analytical Greeks match finite differences | delta 1.8e-09, gamma 6.8e-08, vega 2.0e-10; vanna 3.2e-08 and volga 1.2e-07 measured **off-ATM** where they are non-zero |
| Carr-Madan FFT matches direct quadrature | max abs error **0.00000**, max relative 3.3e-05 across 30d/60d/1y; deep-OTM strikes agree to 1e-6 |
| Heston discretization bias is immaterial at these parameters | Feller `2κθ = 0.16 ≥ ξ² = 0.09` **satisfied**; min variance across 400 paths 0.00183 vs a 1e-8 floor; `E[S_T]` bias −0.016% |
| BS-on-Heston model gap is second-order | 0.04×–0.39× of the quoted half-spread on every leg; never exceeds it |
| MC converges at the theoretical rate | log-log slope **−0.487** vs theory −0.500, once benchmarked against BS at the same σ |

### Derived quantities — claimable as derivations, not as calibrations

| quantity | value | derivation |
|---|---:|---|
| informed arrival probability | 0.1609 | `2(1−Φ(threshold / σ√(staleness·dt)))`, staleness = 1 step |
| E[\|move\| \| informed] | 0.002647 (E[dS] $1.191) | truncated normal `s_m·φ(k)/(1−Φ(k))` |
| informed share of volume | **0.15** (target) | `informed_vol / (informed_vol + noise_vol)`; anchored to the PIN literature's 10–20% for liquid US equities |
| adverse-selection charge | $0.037–$0.096/leg | `E[edge \| informed] × informed share` — Glosten-Milgrom break-even |
| half-spread, % of premium | **1.13%–3.17%** | base + gamma carry + vega carry + adverse selection. Low single digits, with no fitted coefficient anywhere. |

**Caveat on the informed share:** PIN is estimated as a probability of informed *order arrival*,
not a share of *volume*. Informed traders here are larger on average (7.5 vs 3.0 contracts), so a
matched-volume share implies a lower arrival share than the PIN figure it anchors to. Choosing
15% (the midpoint of 10–20%) is the single discretionary numeric step in the whole sequence; it
was made before P&L was measured.

### Structural results — the most robust claims here

| claim | evidence |
|---|---|
| Before ITEM 11/13, **fill probability was completely independent of quote width** | Sweeping `base_spread` 16× produced **5,335 fills / 19,388 contracts at every width**, identical to the unit, with `spread_capture` exactly linear (`19,388 × 100 × Δbase_spread` reproduces every increment to the dollar) |
| Requiring informed edge > half-spread breaks that linearity | fills now vary 2,066 → 1,505 across the sweep; `spread_capture` roughly flat |
| A reservation price bounds P&L in width | P&L no longer grows without limit; it asymptotes to ≈0 from below |
| **There is no interior optimum in quote width** | P&L rises monotonically with width; the best point tested is the widest, where 9 fills occur in a month. That is a boundary solution meaning "do not trade" and is **not** an optimum |

---

## (b) RETIRED

Numbers that were previously reported and **must not be quoted again**.

| retired figure | why |
|---|---|
| **Sharpe 1.85** (seed 42) | Three compounding defects. 89% of the P&L was a vol warm-up artifact; the P&L itself was 48% overstated; and it is not a Sharpe ratio. Ex-day-0 it was already −0.24 across 20 seeds. |
| **Total P&L $33,837.79** (seed 42) | Overstated by 48%. An independent cash+mark ledger of the same trades gives **$17,258.37**. On seed 3 the defect flipped the sign: +$147,022 reported against −$37,249 true. |
| **"88% residual"** framing | Never reproduced at any commit. The residual was 21.67% of net and 2.94% of gross at the audited baseline. Separately, the *net* denominator is unstable — seed 1 shows 2264% of net against 4.00% of gross. |
| **Residual 21.67% "is a floor given this vol estimator"** (README) | False. It was an implementation gap. Two defects of opposite sign were partly cancelling; the corrected residual is now 2.59% of gross. |
| **Win rate 46.7%, max DD $35,112** (seed 42) | Computed on the 48%-overstated P&L series. Max DD also exceeded total P&L, which the pairing with "Sharpe 1.85" concealed. |
| **`spread_capture` $22,315** (seed 42, baseline) | Measured against a stale mark the book was never valued at; ~85% of the then-residual was this one misfiling. |
| **`vega_pnl` $21,598 / `volga_pnl` $23,003** (seed 42) | Volga carried the wrong sign for endpoint Greeks. Vol terms totalled $44,038 against an exact reprice of **$2,966.70** — a 15× over-attribution. |
| **"Glosten-Milgrom adverse selection mechanism"** (README) | The label was not earned: no informed-arrival probability, no conditional expectation, no Bayesian update. A GM-derived charge exists only as of ITEM 10, and the quoter is still not a GM maker. |
| **"λ=8/day"** as an accurate description of the code | The code applied it against an annualized dt, yielding ~5.7 noise trades per **month**. Docs and code now agree. |
| Intermediate figures from this fix sequence — **P&L +$2,802,607 / Sharpe\* +31.71** (post-ITEM 6) | Absurd. The 19%-of-premium half-spread quoted on 99.9% of opportunities. |
| **P&L −$747,275 / Sharpe\* −33.32** (post-ITEM 7) | Superseded: the adverse-selection charge was missing from the spread entirely. |
| **P&L +$103,675 / Sharpe\* +6.38** (post-ITEM 10) | Superseded: fill probability was still independent of quote width. |
| **P&L +$33,955 / Sharpe\* +2.34** (post-ITEM 11) | Superseded: noise flow still perfectly inelastic; 79.7% informed share. |
| **P&L +$290,930 / Sharpe\* +9.75** (post-ITEM 12) | Superseded by ITEM 13. Also the most flattering intermediate figure, produced by the two changes that push P&L up. |
| Any single-seed comparison | 20-seed Sharpe* at baseline had mean −0.14 and std 3.07. A ±1 move on one seed is noise. |

---

## (c) OPEN — known limitations, unresolved

| # | Limitation | Consequence |
|---|---|---|
| 1 | **Reservation-price scale is almost certainly too tight.** Anchored to the MM's inventory horizon τ, giving p_fill ≈ **1.33%** at the derived ATM spread. | The current −$1.18M median is an artifact of this assumption. **This is the single biggest open item.** There is a circularity: `λ` sets `τ = spd/λ`, and `τ` sets the reservation scale, so adding flow simultaneously tightens every counterparty's reservation price. Counterparty behaviour should not be anchored to the market maker's holding horizon. |
| 2 | **No competition model.** No competing quotes, no queue, no NBBO reference. | Option C from `audit/ITEM8_FILL_MODEL.md` is unimplemented. The MM is a monopolist facing price-taking flow. |
| 3 | **Hedge cost is 100% a 10bps assumption.** `transaction_cost=0.001` × $29.85M notional reproduced the reported hedge cost to the cent. 20–100× realistic institutional SPY execution. | −$598,193 of the current median P&L is this assumption. Never corrected — it was flagged as a realism finding, not a bug, and the decision was deferred. |
| 4 | **Informed traders are profitable 100% by construction.** They observe the contemporaneous fair exactly and are gated on `edge > half_spread`. | There is no informed trader who is *wrong*. Real informed flow has a noisy signal. Adverse selection is therefore an upper bound. |
| 5 | **Flat implied-vol surface.** One `sigma_implied` for all strikes and expiries, from a 10-sample rolling realized-vol window. | No skew, no term structure. The MM cannot be picked off on relative value, and every vol-Greek term attributes estimator noise. `sigma_uncertainty_window=10` is small enough that step-by-step vol attribution is unusable — the EOD basis is a workaround, not a fix. |
| 6 | **No inventory skew.** `Quoter.quote` returns `fair ± hs` and is never passed the position. | Nothing pulls the book toward flat except the directional size throttle. `test_symmetric_around_fair` pins this as *current* behaviour, not desired. |
| 7 | **Single underlying, single vol regime.** One Heston path family, one parameter set, 30 days. | No regime shifts, no cross-asset effects, no earnings/event risk. |
| 8 | **No position aging or expiry handling.** Legs are 30d/60d and the run is 30 days; positions near expiry are simply skipped when `T ≤ 0`. | No pin risk, no assignment, no roll. |
| 9 | **Greek caps are redundant with the position limit** for this universe (set at 100% of declared capacity). | Not redundant in principle — a longer-dated or larger book would push per-contract vega up until they bind first. A genuinely binding aggregate cap needs a capital base, which does not exist. |
| 10 | **Intra-step risk-limit overshoot (latent).** `port_g` is computed once per step; six legs then quote against it. Portfolio \|vega\| reached 3.27× its cap pre-fix. | Bounded by the per-leg cap in practice, but the mechanism is still there. |
| 11 | **`src/pnl/attribution.py` is dead code.** The engine inlines its own attribution. | Four tests still exercise the module that never runs. |
| 12 | **30 observations, one path per seed.** | A defensible Sharpe interval needs ~153 independent 30-day paths (≈18 years). ~88% of across-seed variance is 30-observation estimation noise, not strategy variability. **Longer paths, not more seeds.** |

---

## (d) THE DEFECT LOG

Every defect found, how it surfaced, and what it did to the reported numbers. This is the most
valuable artifact of the exercise.

| # | Defect | Location | How it was found | Effect on reported numbers |
|---|---|---|---|---|
| 1 | **`realized_pnl` substituted for cash flow in the MTM identity.** `fill_option` booked an inception-to-date gain, not cash. Opening premium never entered P&L; closing trades booked prior days' gains twice. | `inventory.py:23,34`, `engine.py:186` | Built an independent cash+mark ledger and compared. | **Total P&L overstated 48%.** $33,838 reported vs $17,258 true (seed 42). **Sign-flipped seed 3**: +$147,022 vs −$37,249. Every downstream metric inherited it. |
| 2 | **Vol estimator seeded with ten zero returns**, so `std()==0` and σ hit the 0.01 floor against a true 0.20 for 10 steps. | `engine.py:65`, consumed at `:87-89` | Noticed day 0 was $30,222 of a $33,838 month; measured σ per step. | **89% of seed-42 P&L was this artifact.** Ex-day-0 across 20 seeds: median Sharpe* +1.22 → **−0.24**, median P&L +$23,414 → **−$2,662**. |
| 3 | **Informed traders took the wrong side on every put.** `generate_trades` never received `option_type`; side came from the sign of the underlying move alone. | `order_flow.py:16-17,28-36` | Read the signature; measured directional edge per leg. | **0 of every put contract traded profitably, in all 6 seeds tested.** "Adverse selection" was a **+$28,513 subsidy to the MM**, 0 of 152 contracts losing. |
| 4 | **Order flow read `prices[idx+1]`** — the counterparty saw the next bar. | `engine.py:84-86,127` | Traced index arithmetic against the "no look-ahead" comment. | Informed win rate **100% (334/334)**. Adverse selection was mechanically guaranteed, not probabilistic. |
| 5 | **`lambda_noise` applied against an annualized dt**, 252× too low. README said 8/day. | `configs/default.py:18`, `order_flow.py:20` | Computed `8.0 × dt × steps` = 5.71 expected trades **per month**. | **Zero noise trades in every seed.** Flow was 100% informed. `spread_capture` was not liquidity-provision revenue at all. |
| 6 | **Vega cap bound at ~19 contracts** for the whole book — less than one leg's limit of 20. | `limits.py:15-17`, `configs/default.py:39` | Instrumented `adjusted_quote_size`. | **Quoting stopped at step 4 of 2,340. 1.75% participation**, 96.7% of refusals from the vega cap, 0 from the per-leg cap. Every Greek line was the carry of a static 44-contract book. |
| 7 | **Size throttle used `abs(position)`**, refusing both sides at the cap. | `limits.py:19-20` | Traced why relaxing the vega cap made participation *worse*. | Book froze at ±20/leg and could only exit via expiry. Terminal book was **never mixed** — all six legs always the same sign. Fixing it took participation ~5% → **96.6%**. |
| 8 | **Volga (and vanna) added with the wrong sign** for Greeks evaluated at the endpoint σ. | `engine.py:208-209` | Exact reprice of the EOD book vs three candidate expansions. | Vol terms $44,038 against a true revaluation of **$2,966.70** — 15× over-attribution. Error $41,072 → $3,809 after correction. |
| 9 | **Half-spread was 10–51% of premium**; `gamma_coeff × \|gamma\| × contract_size` built a dollar charge from a dimensionless second derivative with no horizon or position size. | `quoter.py:12-15` | Measured half-spread as a fraction of premium per leg. | 97%+ of the half-spread came from that one term. `spread_capture` was an artifact of quote width, not liquidity provision. |
| 10 | **`spread_capture` measured against a stale mark** the book was never valued at. | `engine.py:111,136,146` | Recomputed against contemporaneous fair. | **Residual was 104.72% of gross** — the attribution explained none of the P&L. Adding the adverse-selection component took it to **1.42%**. |
| 11 | **Fill probability independent of quote width.** | `order_flow.py:19-26,33-44` | Swept `base_spread` 16× and compared fill counts. | **5,335 fills / 19,388 contracts at every width, identical to the unit.** P&L exactly linear and unbounded in a quantity the MM chooses. |
| 12 | **Informed traders crossed spreads wider than their own edge.** | `order_flow.py:33-44` | Compared per-fill edge to half-spread. | 7.2% of informed fills (302/4,216) were certain losses. |
| 13 | **Quote staleness of 2 steps = 10 minutes.** | `configs/default.py` | Converted steps to wall-clock. | Informed arrivals on **32% of all steps**; informed share 88.7%. Real makers requote in milliseconds. |
| 14 | **`test_components_plus_residual_equal_total` was a tautology** — `residual := total − Σcomponents`. | `test_attribution.py:25`, same defect at `test_engine.py:11-20` | Changed the P&L identity and watched all 70 tests still pass. | **This is why every defect above survived a green suite.** The test passed unchanged while P&L was 48% wrong. |
| 15 | **Put-call parity test on `heston_price` was a tautology** — the put branch *is* `call − (S − Ke^−rT)`. | `test_characteristic_function.py:18-22` | Read the implementation. | The Heston put was never independently validated. |
| 16 | **`plot_convergence` benchmarked a GBM MC against the Heston CF price.** | `comparison.py:171-212` | Computed MAE against both benchmarks across N. | Error floored at the **0.0265 model gap** instead of converging. Plot was titled "MC Convergence" and printed slope **−0.395** against "theory −0.500". Correct benchmark gives −0.487, and the discrepancy *widened* with N. |
| 17 | **`heston_price` returned small negatives deep OTM**; `heston_price_grid` already clipped. | `characteristic_function.py:68` | Compared the two routes at wide strikes. | Cosmetic, but the two pricing routes disagreed on sign. |
| 18 | **Sensitivity grid manufactures selection bias.** Best-of-27 on 5 shared seeds. | `sensitivity.py:139-205` | Simulated the null with observed noise. | `E[max of 27] − true` = **+1.319** at ρ=0. Observed best−mean was +0.670, and across-combo std (0.4913) was *smaller* than the single-combo SE (0.6606) — the ranked table carries no demonstrated information. |
| 19 | **Greek caps mutually inconsistent by ~100×.** `max_vega` bound at 6.6% of declared capacity, `max_gamma` at 646%. | `configs/default.py` | Derived per-contract Greek load from the option universe. | One cap refused quotes the position limit permitted; the other never bound. |
| 20 | **"Sharpe" is not a Sharpe ratio.** No capital base exists. | `report.py:8-13` | Grepped for any capital/equity/NAV concept. | Every Sharpe figure ever reported by this project. Still unfixed — the metric is retained with the caveat rather than renamed. |

### Refuted hypotheses

Tested and false. Recorded so they are not re-raised.

| hypothesis | verdict |
|---|---|
| Hedge cash flows are double-counted | **REFUTED.** Hand-worked: cash out plus mark, counted once. |
| Hedge transaction cost is double-charged | **REFUTED.** Embedded in the fill price once; the attribution line explains it, it does not re-charge it. |
| SOD/EOD sigma revaluation gap | **REFUTED.** `max\|sod_book(d) − eod_book(d−1)\| = 0.00` exactly, across all 30 days. |
| `fill_size` vs `spread_capture` size mismatch | **REFUTED.** Both use `fill_size`. |
| BS-on-Heston mismatch is a material driver | **REFUTED.** 0.04×–0.39× of the quoted half-spread; never exceeds it. |
| Greeks are inaccurate | **REFUTED.** ≤1.7e-07 vs finite differences off-ATM. |
| Pricing methods disagree | **REFUTED.** CRR ≤ $0.022, MC ≤ $0.035, CF ≤ $0.0002. |
| Heston variance scheme is materially biased | **REFUTED at these parameters.** Feller satisfied; min variance 0.00183 vs a 1e-8 floor; `E[S_T]` bias −0.016%. |

---

## The one-paragraph summary

The pre-audit headline — **Sharpe 1.85 on $33,838** — was wrong in three independent ways at
once: the P&L was 48% overstated by an accounting defect, 89% of what remained was a
volatility-initialization artifact, and the statistic is not a Sharpe ratio because no capital
base exists. Twenty defects were found; a green 70-test suite detected none of them, because
the load-bearing test was a tautology. The pricing and Greeks core audited clean throughout —
**the failures were all in accounting, risk limits, and the market model.** The simulator now
has a closing attribution identity (residual 2.59% of gross), a spread derived from carry cost
and Glosten-Milgrom break-even with no fitted coefficients, and fill probability that responds
to quote width. It does not yet have a defensible P&L number, and the current −$1.18M is
dominated by one asserted behavioural parameter that is probably too tight. **That is the next
thing to fix, and until it is, no P&L figure from this project should be quoted as a result.**
