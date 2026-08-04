# FINAL NUMBERS — options-mm

**Read this file before quoting any number from this project.**

> ## ⚠️ PHASE E SUPERSEDES PARTS OF THIS FILE
>
> Sections (a)–(d) below were written at commit `985b294`, **before** the Phase E sequence
> (`9772ded` → `60a1db4` → `f137383` → `8d18931` → `95c74b8`). Where Phase E changed a number,
> **§(e) at the bottom of this file is authoritative and this one is not.** Specifically:
>
> | Stale here | Current |
> |---|---|
> | Attribution has 8 components + residual | **9 components** — `adverse_selection_vol` added |
> | residual 2.50% of gross | **2.07% of gross** |
> | median P&L +$290,930 (default config) | **−$649,300** (seeds 0–19, ex day 0, post-ITEM 17) |
> | OPEN #6 "flat implied-vol surface" | **False** — a Heston IV surface with skew and term structure exists (`src/pricing/vol_surface.py`) |
> | OPEN #3 "no competition model … unimplemented" | **Built and tested, gated OFF**; raises without real SPY data (`src/market/competition.py`) |
>
> Everything else in (a)–(d) — the defect log, the retirements, the pricing/Greeks audit, the
> capital-base caveat — stands unchanged.

Repo: `/Users/harry/Desktop/options-mm`. Branch `main` (merged from
`audit/blind-audit-2026-07-26` at `78168bd`).
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

### ⛔ The P&L level is NOT claimable, in either configuration

**This is the terminal finding of the exercise, and it is a result, not a failure.**

> **With no competition model, this simulator's P&L LEVEL is not identified.**

There are two configurations, and neither yields a claimable performance number:

| configuration | median P&L | median Sharpe* | why it is NOT claimable |
|---|---:|---:|---|
| **`use_reservation_price=False`** (repo default) | +$290,930 | +9.752 | **Unbounded in width, level not identified.** Noise flow is perfectly inelastic, so P&L rises monotonically with quote width. The level is a function of a width *we chose*, not of anything the model determines. Quote wider and it goes up, without limit. |
| **`use_reservation_price=True`** (ITEM 13) | −$1,132,830 | −42.342 | **Artifact of the mis-anchored reservation scale.** p_fill ≈ 1.33% at the derived spread — 99% of benign flow screened out — because the scale is anchored to the market maker's own inventory horizon τ, which is circular (λ sets τ sets the scale). |

The default is set to `False` because the repo's default configuration must not be one
identified as an artifact — **not** because that configuration is more trustworthy. It is
unbounded rather than wrong, which is a different defect, not a smaller one.

Do not quote either P&L figure, either Sharpe*, or the positive-seed counts (19/20 and 0/20
respectively) as a performance result. Both are recorded in RETIRED for that reason.

### What IS claimable

Everything below is robust to the calibration question above.

| claim | value | measured on |
|---|---|---|
| **The attribution identity closes.** | residual median **2.50%** of gross flow; per-seed range 0.05%–6.90%; medians across the fix sequence ranged 1.42%–3.01% | seeds 0–19, ex day 0, verified closing on **all 20 seeds** in both configurations |
| **The backtest is bit-reproducible.** | byte-identical daily P&L, attribution and price path across separate processes | seed 42, re-verified after every commit |
| **The pricing and Greeks core is clean.** | see table below | multiple grids |
| **The derived adverse-selection share matches the measured one, independently.** | derived **0.8868** from the Heston dynamics and arrival model; measured **~89%** from fill counts | two independent routes, no shared inputs |
| **Fill probability was completely independent of quote width.** | **5,335 fills / 19,388 contracts at every width** across a 16× sweep, identical to the unit | seed 42 |
| **The half-spread derivation carries no fitted coefficient.** | 1.13%–3.17% of premium across the six legs | analytic, at σ=0.20 |

### Attribution, default configuration (medians, seeds 0–19, ex day 0)

Shape and sign are informative; the **levels inherit the identification problem above.**

| component | value | caveat |
|---|---:|---|
| `spread_capture` | +$1,219,681 | Gross quoted spread, measured against quote-time fair. Must be read together with `adverse_selection`; their **sum** is the edge against contemporaneous fair. Level is a function of the chosen quote width. |
| `adverse_selection` | −$380,191 | Applies to informed flow only. Under `use_reservation_price=True` this roughly triples, to −$1,021,235 — see the gating asymmetry in OPEN. |
| `hedge_cost` | −$525,702 | **100% an artifact of `transaction_cost=0.001`** (10bps/hedge), which is 20–100× realistic institutional SPY execution. Never corrected — see OPEN. |
| `theta_pnl` | +$80 | |
| `gamma_pnl` | +$9 | |
| `vega_pnl` | +$12,407 | |
| `vanna_pnl` | −$637 | |
| `volga_pnl` | −$1,580 | |
| `residual` | +$12,527 | 2.50% of gross. |

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
| **P&L +$290,930 / Sharpe\* +9.75** (default config, `use_reservation_price=False`) | Not a performance result. Unbounded in quote width — noise flow is inelastic, so the level is a function of a width we chose. |
| **P&L −$1,132,830 / Sharpe\* −42.34** (`use_reservation_price=True`) | Not a performance result. Artifact of the mis-anchored reservation scale (p_fill ≈ 1.33%). |
| **The sensitivity grid's ranked table and "Best combo" line** (`results/sensitivity.csv`, README) | The ranking carries no demonstrated information. Across-combo std (0.4913) is smaller than the single-combo standard error (0.6606); the full best-to-worst spread is consistent with no effect. Retired in the README and annotated in place. |
| **README "Resume Bullets"** — the order-flow, attribution, and sensitivity bullets | Four of five state things the audit disproved: a two-population flow model that produced zero noise trades, a Glosten-Milgrom label that was not earned, informed traders "exploiting staleness" who actually read the next bar, a "hard closure test" that was the keystone tautology, an "88% → 22%" residual figure that never reproduced, and a ranking with no information. Kept for provenance with corrections attached. |
| **README sample output** — Total P&L $33,837.79 / Sharpe 1.849 / residual 21.67% | Published the exact figures retired above. Replaced, with the retirement stated inline. |
| **The hedge-cost line as a realistic execution estimate** (−$525,702 median) | It is an artifact of a 10bps assumption that is 20–100× too large (defect #21). Correcting it would move P&L **substantially in the flattering direction** on a metric already established as unidentified, so it is documented rather than changed. Do not quote the hedge-cost line as a cost estimate, and do not net it out to claim a better result. |
| Any single-seed comparison | 20-seed Sharpe* at baseline had mean −0.14 and std 3.07. A ±1 move on one seed is noise. |

---

## (c) OPEN — known limitations, unresolved

| # | Limitation | Consequence |
|---|---|---|
| 1 | **Reservation-price scale is almost certainly too tight.** Anchored to the MM's inventory horizon τ, giving p_fill ≈ **1.33%** at the derived ATM spread. | Gated OFF by default for this reason. When enabled it produces a −$1,132,830 median that is an artifact of the assumption, not a measurement. **This is the single biggest open item**, because it is the only mechanism here that bounds P&L in quote width at all — without it the level is unidentified. There is a circularity: `λ` sets `τ = spd/λ`, and `τ` sets the reservation scale, so adding flow simultaneously tightens every counterparty's reservation price. Counterparty behaviour should not be anchored to the market maker's holding horizon. |
| 2 | **Gating asymmetry between the two counterparty populations.** Noise traders face a *probabilistic* reservation gate (`p_fill = exp(-cost/scale)`); informed traders face a *deterministic* one (`edge > half_spread`). | Non-commensurate screens. This is the mechanism behind adverse selection roughly TRIPLING under ITEM 13 (−$380,191 → −$1,021,235): the probabilistic gate removes benign flow stochastically while the deterministic gate leaves toxic flow that clears its threshold entirely intact. Any future fix must gate both populations on comparable terms — either both probabilistic or both deterministic — or the flow mix is distorted by the screening itself, independently of any calibration. |
| 3 | **No competition model.** No competing quotes, no queue, no NBBO reference. | Option C from `audit/ITEM8_FILL_MODEL.md` is unimplemented. The MM is a monopolist facing price-taking flow. |
| 4 | **Hedge cost is 100% a 10bps assumption.** `transaction_cost=0.001` × $29.85M notional reproduced the reported hedge cost to the cent. 20–100× realistic institutional SPY execution (defect #21). | −$525,702 of the default-configuration median P&L. **Left uncorrected on purpose.** Fixing it would add roughly half a million dollars to a P&L whose LEVEL is already established as unidentified — a large move in the flattering direction on a number that cannot yet be interpreted. It should be corrected only alongside a competition model, so the result is judged against an identified baseline rather than a chosen quote width. |
| 5 | **Informed traders are profitable 100% by construction.** They observe the contemporaneous fair exactly and are gated on `edge > half_spread`. | There is no informed trader who is *wrong*. Real informed flow has a noisy signal. Adverse selection is therefore an upper bound. |
| 6 | **Flat implied-vol surface.** One `sigma_implied` for all strikes and expiries, from a 10-sample rolling realized-vol window. | No skew, no term structure. The MM cannot be picked off on relative value, and every vol-Greek term attributes estimator noise. `sigma_uncertainty_window=10` is small enough that step-by-step vol attribution is unusable — the EOD basis is a workaround, not a fix. |
| 7 | **No inventory skew.** `Quoter.quote` returns `fair ± hs` and is never passed the position. | Nothing pulls the book toward flat except the directional size throttle. `test_symmetric_around_fair` pins this as *current* behaviour, not desired. |
| 8 | **Single underlying, single vol regime.** One Heston path family, one parameter set, 30 days. | No regime shifts, no cross-asset effects, no earnings/event risk. |
| 9 | **No position aging or expiry handling.** Legs are 30d/60d and the run is 30 days; positions near expiry are simply skipped when `T ≤ 0`. | No pin risk, no assignment, no roll. |
| 10 | **Greek caps are redundant with the position limit** for this universe (set at 100% of declared capacity). | Not redundant in principle — a longer-dated or larger book would push per-contract vega up until they bind first. A genuinely binding aggregate cap needs a capital base, which does not exist. |
| 11 | **SUBSTANTIVE** | **Intra-step risk-limit overshoot (latent).** `port_g` is computed once per step; six legs then quote against it. Portfolio \|vega\| reached 3.27× its cap pre-fix. | Bounded by the per-leg cap in practice, but the mechanism is still there. |
| 12 | **`src/pnl/attribution.py` is dead code.** The engine inlines its own attribution. | Four tests still exercise the module that never runs. |
| 13 | **30 observations, one path per seed.** | A defensible Sharpe interval needs ~153 independent 30-day paths (≈18 years). ~88% of across-seed variance is 30-observation estimation noise, not strategy variability. **Longer paths, not more seeds.** |

---

## (d) THE DEFECT LOG

Every defect found, how it surfaced, and what it did to the reported numbers. This is the most
valuable artifact of the exercise.

**Defect #14 is the KEYSTONE.** `test_components_plus_residual_equal_total` asserted
`Σcomponents + residual == total` where the engine *defines* `residual := total − Σcomponents` —
an identity asserted against its own definition. It passes for any inputs, including arbitrarily
wrong ones. **A green 70-test suite caught none of the other twenty defects because the
load-bearing test could not fail.** It passed unchanged while total P&L was 48% overstated, and
passed again, still unchanged, when that was corrected and seed 42's total moved by $16,579.
Fix the keystone first: every other defect below was findable only once something could fail.

Tiers: **KEYSTONE** — why the rest survived. **SEVERE** — changed the sign, magnitude, or
meaning of a headline number. **SUBSTANTIVE** — materially distorted a reported quantity or the
model's economics. **MINOR** — real but small, or latent.

| # | Tier | Defect | Location | How it was found | Effect on reported numbers |
|---|---|---|---|---|---|
| 1 | **SEVERE** | **`realized_pnl` substituted for cash flow in the MTM identity.** `fill_option` booked an inception-to-date gain, not cash. Opening premium never entered P&L; closing trades booked prior days' gains twice. | `inventory.py:23,34`, `engine.py:186` | Built an independent cash+mark ledger and compared. | **Total P&L overstated 48%.** $33,838 reported vs $17,258 true (seed 42). **Sign-flipped seed 3**: +$147,022 vs −$37,249. Every downstream metric inherited it. |
| 2 | **SEVERE** | **Vol estimator seeded with ten zero returns**, so `std()==0` and σ hit the 0.01 floor against a true 0.20 for 10 steps. | `engine.py:65`, consumed at `:87-89` | Noticed day 0 was $30,222 of a $33,838 month; measured σ per step. | **89% of seed-42 P&L was this artifact.** Ex-day-0 across 20 seeds: median Sharpe* +1.22 → **−0.24**, median P&L +$23,414 → **−$2,662**. |
| 3 | **SUBSTANTIVE** | **Informed traders took the wrong side on every put.** `generate_trades` never received `option_type`; side came from the sign of the underlying move alone. | `order_flow.py:16-17,28-36` | Read the signature; measured directional edge per leg. | **0 of every put contract traded profitably, in all 6 seeds tested.** "Adverse selection" was a **+$28,513 subsidy to the MM**, 0 of 152 contracts losing. |
| 4 | **SEVERE** | **Order flow read `prices[idx+1]`** — the counterparty saw the next bar. | `engine.py:84-86,127` | Traced index arithmetic against the "no look-ahead" comment. | Informed win rate **100% (334/334)**. Adverse selection was mechanically guaranteed, not probabilistic. |
| 5 | **SUBSTANTIVE** | **`lambda_noise` applied against an annualized dt**, 252× too low. README said 8/day. | `configs/default.py:18`, `order_flow.py:20` | Computed `8.0 × dt × steps` = 5.71 expected trades **per month**. | **Zero noise trades in every seed.** Flow was 100% informed. `spread_capture` was not liquidity-provision revenue at all. |
| 6 | **SEVERE** | **Vega cap bound at ~19 contracts** for the whole book — less than one leg's limit of 20. This was the **APPARENT** cause of the quoting freeze; the actual binding constraint was #7. | `limits.py:15-17`, `configs/default.py:39` | Instrumented `adjusted_quote_size`; **corrected by relaxing the suspected cause and observing the OPPOSITE of the predicted effect**. | **Quoting stopped at step 4 of 2,340. 1.75% participation**, 96.7% of refusals attributed to the vega cap, 0 to the per-leg cap. Every Greek line was the carry of a static 44-contract book. **The causal attribution was wrong**: raising `max_vega` to 100% of declared capacity made participation *worse* (6.47% → 3.41%), because the constraint simply moved to the `abs(position)` throttle (#7), which then froze the book at ±20/leg. Fixing #7 alone took participation to 96.6% with the vega cap untouched. The cap was miscalibrated and worth fixing, but it was not what stopped the market maker quoting. |
| 7 | **SEVERE** | **Size throttle used `abs(position)`**, refusing both sides at the cap. **The actual cause of the freeze that #6 was blamed for.** | `limits.py:19-20` | Traced why relaxing the vega cap made participation *worse* — a predicted-effect failure that falsified the #6 hypothesis. | Book froze at ±20/leg and could only exit via expiry. Terminal book was **never mixed** — all six legs always the same sign. Fixing it took participation ~5% → **96.6%**. |
| 8 | **SUBSTANTIVE** | **Volga (and vanna) added with the wrong sign** for Greeks evaluated at the endpoint σ. | `engine.py:208-209` | Exact reprice of the EOD book vs three candidate expansions. | Vol terms $44,038 against a true revaluation of **$2,966.70** — 15× over-attribution. Error $41,072 → $3,809 after correction. |
| 9 | **SUBSTANTIVE** | **Half-spread was 10–51% of premium**; `gamma_coeff × \|gamma\| × contract_size` built a dollar charge from a dimensionless second derivative with no horizon or position size. | `quoter.py:12-15` | Measured half-spread as a fraction of premium per leg. | 97%+ of the half-spread came from that one term. `spread_capture` was an artifact of quote width, not liquidity provision. |
| 10 | **SUBSTANTIVE** | **`spread_capture` measured against a stale mark** the book was never valued at. | `engine.py:111,136,146` | Recomputed against contemporaneous fair. | **Residual was 104.72% of gross** — the attribution explained none of the P&L. Adding the adverse-selection component took it to **1.42%**. |
| 11 | **SUBSTANTIVE** | **Fill probability independent of quote width.** | `order_flow.py:19-26,33-44` | Swept `base_spread` 16× and compared fill counts. | **5,335 fills / 19,388 contracts at every width, identical to the unit.** P&L exactly linear and unbounded in a quantity the MM chooses. |
| 12 | **MINOR** | **Informed traders crossed spreads wider than their own edge.** | `order_flow.py:33-44` | Compared per-fill edge to half-spread. | 7.2% of informed fills (302/4,216) were certain losses. |
| 13 | **SUBSTANTIVE** | **Quote staleness of 2 steps = 10 minutes.** | `configs/default.py` | Converted steps to wall-clock. | Informed arrivals on **32% of all steps**; informed share 88.7%. Real makers requote in milliseconds. |
| 14 | **KEYSTONE** | **`test_components_plus_residual_equal_total` was a tautology** — `residual := total − Σcomponents`. | `test_attribution.py:25`, same defect at `test_engine.py:11-20` | Changed the P&L identity and watched all 70 tests still pass. | **This is why every defect above survived a green suite.** The test passed unchanged while P&L was 48% wrong. |
| 15 | **MINOR** | **Put-call parity test on `heston_price` was a tautology** — the put branch *is* `call − (S − Ke^−rT)`. | `test_characteristic_function.py:18-22` | Read the implementation. | The Heston put was never independently validated. |
| 16 | **MINOR** | **`plot_convergence` benchmarked a GBM MC against the Heston CF price.** | `comparison.py:171-212` | Computed MAE against both benchmarks across N. | Error floored at the **0.0265 model gap** instead of converging. Plot was titled "MC Convergence" and printed slope **−0.395** against "theory −0.500". Correct benchmark gives −0.487, and the discrepancy *widened* with N. |
| 17 | **MINOR** | **`heston_price` returned small negatives deep OTM**; `heston_price_grid` already clipped. | `characteristic_function.py:68` | Compared the two routes at wide strikes. | Cosmetic, but the two pricing routes disagreed on sign. |
| 18 | **SEVERE** | **Sensitivity grid manufactures selection bias.** Best-of-27 on 5 shared seeds. | `sensitivity.py:139-205` | Simulated the null with observed noise. | `E[max of 27] − true` = **+1.319** at ρ=0. Observed best−mean was +0.670, and across-combo std (0.4913) was *smaller* than the single-combo SE (0.6606) — the ranked table carries no demonstrated information. |
| 19 | **MINOR** | **Greek caps mutually inconsistent by ~100×.** `max_vega` bound at 6.6% of declared capacity, `max_gamma` at 646%. | `configs/default.py` | Derived per-contract Greek load from the option universe. | One cap refused quotes the position limit permitted; the other never bound. |
| 20 | **SUBSTANTIVE** | **"Sharpe" is not a Sharpe ratio.** No capital base exists. | `report.py:8-13` | Grepped for any capital/equity/NAV concept. | Every Sharpe figure ever reported by this project. Still unfixed — the metric is retained with the caveat rather than renamed. |
| 21 | **SUBSTANTIVE** | **Hedge transaction cost of 10bps is roughly 20–100× realistic.** `transaction_cost=0.001` charged on the full notional of every hedge, with a full-flatten policy and no dead band. | `configs/default.py` (`HEDGER.transaction_cost`), applied at `hedger.py:15-23` | Multiplied the assumption by the measured hedged notional and reproduced the reported hedge-cost line **to the cent** ($29,851,796 × 0.001 = $29,851.80). | Accounts for **−$525,702 of the default-configuration median P&L** (seeds 0–19, ex day 0) — the largest single negative component. **External anchor:** institutional equity execution in a name like SPY is low single-digit basis points all-in (a 1-cent half-spread on a $450 underlying is ~0.11bp, plus fees), so 10bps is 20–100× too large. Hedging also fires on ~48% of 5-minute bars against a static book, so the turnover compounding the assumption is itself gratuitous. **DELIBERATELY NOT FIXED** — see RETIRED and OPEN #4. |

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
base exists. Twenty-one defects were found; a green 70-test suite detected none of them, because
the load-bearing test was a tautology. The pricing and Greeks core audited clean throughout —
**the failures were all in accounting, risk limits, and the market model.** The simulator now
has a closing attribution identity (residual 2.50% of gross), a spread derived from carry cost
and Glosten-Milgrom break-even with no fitted coefficients, and an optional fill model that
responds to quote width. It does not have an identified P&L LEVEL, and that is the terminal finding rather than a loose
end: with noise flow inelastic the level is unbounded in a quote width we choose, and the one
mechanism that bounds it rests on a behavioural scale anchored to the wrong quantity. **Fixing
that requires a competition model. Until there is one, no P&L figure from this project should be
quoted as a result — in either configuration.**

---

# (e) PHASE E — ITEM 15–18 (authoritative over §(a)–(d) where they disagree)

Commits: `9772ded` (SPY chain loader) → `60a1db4` (Heston IV surface) → `f137383` (quote every
leg off the surface) → `8d18931` (vol-informed traders, split adverse selection) → `95c74b8`
(competition module, data pending). Written 2026-08-04 to close the gap between §(a)–(d) and HEAD.

## Attribution at HEAD — 9 components

`src/backtest/engine.py:385-392` plus the hedge-cost accumulator at `:314`. Medians, seeds 0–19,
ex day 0, ITEM 16 → ITEM 17:

| component | ITEM 16 | ITEM 17 (HEAD) |
|---|---:|---:|
| `spread_capture` | +$1,184,181 | +$1,262,628 |
| `adverse_selection` (spot gap, `fair_now − fair`) | −$375,786 | −$343,157 |
| `adverse_selection_vol` (vol gap, `fair_true − fair_now`) | — | **−$1,064,081** |
| `vega_pnl` | +$13,747 | +$80,475 |
| `hedge_cost` | −$527,595 | −$530,570 |
| `residual` | −$1,968 | −$70,704 |
| **total P&L** | **+$315,669** | **−$649,300** |
| **residual / gross** | 2.52% | **2.07%** |

`theta_pnl`, `gamma_pnl`, `vanna_pnl`, `volga_pnl` complete the nine; all are small (|·| < $2k).

**The residual as an instrument.** Adding a P&L source with no matching attribution component
sent the residual to **44.12% of gross** — defect #10 reproduced in miniature, caught on contact.
It returned to 2.07% once adverse selection was split by population. Always quote the residual
**as a fraction of gross**; the net denominator is unstable (seed 1: 2264% of net vs 4.00% of gross).

## ⚠️ CORRECTION — the vol-informed cost multiple is 3.1×, not 2.6×

Commit `8d18931`'s message states *"vol-informed flow costs the maker 2.6x what spot-informed
flow costs it."* **That figure is wrong and cannot be reproduced from any basis in this repo.**
The same commit's own ledger gives:

```
1,064,081 / 343,157 = 3.10x        <- correct, HEAD medians
1,064,081 / 375,786 = 2.83x        <- against the ITEM 16 spot figure
(1,064,081/11.7) / (343,157/5.1) = 1.35x   <- normalised per unit of flow share
```

**Cite 3.1×.** The commit message is immutable and stays as written; this row is the correction.

## Flow mix — and why "recovered … matching PIN" is CIRCULAR

Measured at **seed 42** after ITEM 17: **83.2% noise / 5.1% spot-informed / 11.7% vol-informed
= 16.8% informed.** Commit `8d18931` describes this as *"inside the 10–20% PIN range this was
anchored to, and arrived at independently."*

**The "independently" claim is not supportable, and this is a documented limitation.**
`configs/default.py:23-27` contains the inversion table used to *choose* `lambda_noise`:

```
# SHARE of volume in the empirically observed range, noise volume must satisfy
# noise = informed * (1 - share) / share:
#     share 10% -> lambda_noise 282.4
#     share 15% -> lambda_noise 177.8      <- lambda_noise=177.8 is the shipped value
#     share 20% -> lambda_noise 125.5
```

Commit `1fb43aa` ("anchor noise arrivals to PIN") set it. The later 16.8% is that chosen input
re-emerging with a second informed population layered on top — **not an independent recovery of
an empirical quantity.** §(a) already concedes the underlying point ("the single discretionary
numeric step in the whole sequence"); this states the consequence explicitly.

**Do not write "recovered X% informed share matching PIN."** The defensible statement is: the
noise arrival rate was calibrated to place the informed share inside the empirical PIN band, and
adding a vol-informed population left it there at 16.8%.

## Heston implied-vol surface (ITEM 15/16) — supersedes OPEN #6

OPEN #6 ("flat implied-vol surface") is **no longer true**. `src/pricing/vol_surface.py`
separates level from shape:

```
quoted_iv(K,T) = atm_level_estimate(T) + [surface_iv(K,T) − surface_atm_iv(T)]
```

- **Shape** derived from the Heston characteristic function at the model's own parameters
  (ρ=−0.7 → negative skew, ξ=0.3 → smile curvature, κ/θ → term structure). **No fitted constants.**
- **Level** remains the maker's own 10-sample rolling realized-vol estimate, so the staleness and
  adverse-selection story is preserved.
- Precomputed onto a (log-moneyness × maturity) grid and interpolated with `RectBivariateSpline`
  (~14,000 CF inversions per backtest avoided).

**It is semi-analytic, not closed-form.** Two routes, `src/pricing/characteristic_function.py`:
little-trap formulation (Albrecher et al. 2007) for the branch cut; direct Gauss-Kronrod
quadrature truncated at **u=500** (`limit=500`, `epsabs=1e-9`); and Carr–Madan FFT on a strike
grid. Agreement: **max abs 0.00000, max relative 3.3e-05** across 30d/60d/1y; deep-OTM to 1e-6.

**OPEN — not measured:** the grid interpolation error. The docstring asserts "exact to the grid
resolution"; no number exists. **There is no calibration** — the surface is self-consistent with
the simulator's own Heston parameters, not fitted to market data.

## Vol-informed counterparty population (ITEM 17)

A second informed population trading the surface against the maker's lagging vol estimate. Both
inputs derived, not fitted:

- **True ATM IV** — under Heston, `E[v̄] = θ + (v_t−θ)(1−e^{−κT})/(κT)`; ATM implied vol is its
  square root. Reduces to `√θ` when `v_t = θ` and to `√v_t` as `T → 0`; both limits verified numerically.
- **Arrival threshold** — the standard error of the maker's own estimator,
  `σ/√(2·window) = 0.045` at σ=0.20, window=10. A signal below that is indistinguishable from
  sampling noise, so a rational informed trader would not act on it.

Uses contemporaneous `v_now`, gated on `edge > half_spread`, same information discipline as the
spot population. **Known gap, recorded not patched:** the quoter's Glosten–Milgrom charge
compensates for *spot*-informed arrival only, so vol-informed flow is currently traded against
for free. Fixing it moves P&L substantially upward on a metric still unidentified in quote width.

## Competition module (ITEM 18) — supersedes OPEN #3

OPEN #3 said "unimplemented." It is now **built, tested, and deliberately gated OFF.**

`src/market/competition.py` samples the competing half-spread from the **empirical distribution
of real SPY quotes**, bucketed by moneyness and maturity — no dispersion parameter, because the
dispersion *is* the observed distribution. It gates all flow, informed and noise alike.

**It has never run.** `data/` is empty. `CompetitiveQuotes.from_cache()` raises
`MarketDataUnavailable` rather than substituting an invented distribution — verified. The eight
tests use a synthetic surface as a **fixture, explicitly not a calibration.**

> **Nothing in this project has been validated against real market data.**

**Remaining work:** run `python3 -m src.backtest.data` between 09:30–16:00 ET on a weekday,
commit `data/spy_surface.csv`, set `use_competition=True`, re-run the `base_spread` sweep and test
for an interior optimum. If one exists, comparing the optimal half-spread against the ITEM 10
derived Glosten–Milgrom charge is the real result — two independent routes to the same number.

## What Phase E did NOT change

The terminal finding stands: **the P&L level is still unidentified.** Competition is the
mechanism that would identify it and it has not been run. No P&L or signal-to-noise figure from
this project — at any commit, in any configuration, including −$649,300 — may be quoted as a
performance result.

`compute_sharpe` was renamed **`compute_pnl_signal_to_noise`** (`src/backtest/report.py:8`,
commit `1b3d334`). There is no capital base anywhere in this repo. **"Sharpe 1.32" has never
existed at any commit** — it is not in the worktree and `git log --all -S"1.32"` finds only an
unrelated `results/sensitivity.csv` cell. If you see it on a résumé or in a draft, delete it.
