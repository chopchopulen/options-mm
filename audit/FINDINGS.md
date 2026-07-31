# BLIND AUDIT — options-mm

Commit `dbdc9cf`, branch `audit/blind-audit-2026-07-26`. Baseline: `bench/BASELINE.md`.
All dollar figures are seed 42 unless a seed list is given.

**Verdict column** is an adversarial pass. `CONFIRMED` = I reproduced the finding myself,
independently of the auditor that raised it. `PLAUSIBLE` = the auditor produced evidence, I did
not re-derive it. `REFUTED` = tested and false.

## Coverage gap — read this first

Four subsystems were audited. The **quant-model subsystem got no subagent** — it died twice to
API faults and then the session hit its rate limit. I audited it myself inline, which covered
pricing agreement, Greeks, the Heston/BS mismatch, and discretization, but **not**
`comparison.py`, `heston_price_grid`'s FFT path, or the CF put-call-parity branch. Those remain
unaudited. Everything else below had a dedicated auditor plus my verification.

---

## Severity ranking

| # | Finding | Tag | Verdict |
|---|---|---|---|
| 1 | `total_pnl` is arithmetically wrong — 48% off | correctness | CONFIRMED |
| 2 | The vega cap stops quoting after step 4; this is not market making | correctness | CONFIRMED |
| 3 | 89% of reported P&L is a vol warm-up artifact | correctness + statistics | CONFIRMED |
| 4 | Informed traders trade the wrong direction on every put | correctness | CONFIRMED |
| 5 | Noise flow does not exist — λ applied against an annualized `dt` | correctness | CONFIRMED |
| 6 | Order flow reads `prices[idx+1]` — the counterparty sees the next bar | correctness | CONFIRMED |
| 7 | Volga added with the wrong sign for endpoint Greeks | correctness | CONFIRMED (sign) / PLAUSIBLE (magnitude) |
| 8 | "Sharpe" is not a Sharpe — no capital base exists in the repo | statistics | CONFIRMED |
| 9 | Sharpe CI includes zero; P(SR<0) = 27% | statistics | PLAUSIBLE |
| 10 | Residual reported against the flattering denominator; ✓ passes on seed 42 only | statistics | CONFIRMED |
| 11 | Half-spread is ~19% of premium, 97.6% from one dimensionally-odd term | economics | PLAUSIBLE |
| 12 | Hedge cost is 100% an artifact of a 10bps assumption | economics | CONFIRMED |
| 13 | Glosten-Milgrom label is not earned | economics | CONFIRMED |
| 14 | One staleness event fires a synchronized 6-leg block; effective n = 14 | statistics | PLAUSIBLE |
| 15 | Attribution module is dead code; its tests are tautological | correctness | CONFIRMED |
| 16 | `spread_capture` marked at a price the book is never valued at | correctness | PLAUSIBLE |
| 17 | No inventory skew; throttle is symmetric so the book cannot unwind | economics | CONFIRMED |
| 18 | Sensitivity grid manufactures ~+0.7 to +1.3 Sharpe from selection | statistics | PLAUSIBLE |
| 19 | Gamma P&L uses post-move Greeks | correctness | PLAUSIBLE (immaterial, $3.75) |
| 20 | Intra-step limit overshoot (latent) | correctness | PLAUSIBLE |

Refuted below: hedge cash double-counting, hedge-cost double-charge, SOD/EOD sigma chaining
gap, BS-on-Heston mismatch as a material driver, Greeks accuracy, pricing-method disagreement.

---

## 1. `total_pnl` is arithmetically wrong — reported P&L is 48% higher than cash+mark truth
**`src/backtest/engine.py:183,186` · `src/mm/inventory.py:23,34` · correctness · CONFIRMED**

The daily identity is `mtm = (eod_book − sod_book) + realized_pnl_delta + underlying_pnl`. An
MTM identity requires **cash flow** in the middle term. `fill_underlying` (`inventory.py:41`)
books cash — correct. `fill_option` (`inventory.py:23,34`) books an **inception-to-date realized
gain**, not cash. Two failure modes:

- **Opening trades contribute nothing.** Sell 5 calls @ $3.00, mark $2.90: cash received
  $1,500, `realized_pnl` change **$0.00**. Engine books `−1450 + 0 = −$1,450`; truth is `+$50`.
  The premium is missing entirely — yet `spread_capture` still books the $50, explaining P&L
  that is not in `mtm_pnl` at all.
- **Closing trades double-count.** The inception→SOD gain was already earned through prior days'
  `eod_book − sod_book` and is booked a second time.

**My independent verification** (cash ledger via monkeypatched `Inventory`, terminal mark at the
engine's own estimated sigma):

```
reported total_pnl    33837.79
indep. cash+mark      17566.31
GAP                   16271.48 = 48.1% of reported
```

The auditor measured $16,579.42 / 49.0% using the engine's exact terminal sigma; my $16,271.48 /
48.1% uses a re-derived sigma. Same defect, same magnitude.

Across seeds the sign can flip: **seed 3 reports +$147,022 while true cash P&L is −$37,249**;
seed 13 reports +$180,958 against +$14,889. Every downstream metric — Sharpe, win rate, max
drawdown, `results/`, `backtest_results.png` — inherits this.

**Fix:** track an option cash ledger (`cash += -signed*price*contract_size`) and use it in the
identity. Keep `realized_pnl` for reporting only.

**Refutation experiment:** force the book flat at EOD each day. Engine `mtm` and cash truth must
then converge to pennies daily. If they still diverge, the ledger is wrong, not the engine.

## 2. The vega cap stops quoting at step 4 of 2340 — this is a static book, not a market maker
**`src/risk/limits.py:15-17` · `configs/default.py:39` · correctness · CONFIRMED**

`max_vega=50000` is compared against portfolio vega aggregated as `vega × quantity ×
contract_size`. One ATM 30d contract carries vega ≈ 26 per 1.00 of vol → ×100 = **2,626 vega
units per contract**. The cap binds at **~19 contracts across the entire book** — less than a
single leg's cap of 20.

**My independent verification** (instrumented `RiskLimits.adjusted_quote_size`, seed 42):

```
quote-size calls 14040   nonzero 246 (1.75%)
zero-by-vega 13344       zero-by-leg 0
terminal legs: 427.5p30 +8, 450c30 +8, 450p30 +8, 472.5c30 +8, 427.5p60 +6, 450c60 +6
```

The gamma cap is 12× too loose and never binds; the vega cap is ~10× too tight and causes
**96.7% of all refusals**. Fills happen on 14 steps out of 2,340.

**Consequence:** every Greek line in the attribution table — theta −$12,003, vega +$21,598,
volga +$23,003 — is the carry of a **static +44-contract long-options book** acquired in 14
blocks on day 0–1 and then delta-hedged 1,122 times. It is not market-making economics.

**Fix:** express the Greek caps in the same units as the intended book size and add a test that
a full book at `max_contracts_per_leg` sits inside them. **This is a unit/consistency fix, not a
tuning opportunity — do not adjust it to move P&L.**

**Refutation experiment:** raise `max_vega` to 400k and report the nonzero-quote fraction. If
quoting still dies early, something else is responsible.

## 3. 89% of the reported P&L is a volatility warm-up artifact
**`src/backtest/engine.py:65`, consumed at `:87-89` · correctness + statistics · CONFIRMED**

`log_ret_history = [0.0] * sigma_window` seeds the estimator with ten **zero** returns, so at
step 0 `np.std(...) == 0` → `sigma_implied` is clamped to **0.01** by line 89. The MM quotes,
computes Greeks, hedges, and **marks its book** at 1% vol while true Heston vol is 20%. Recovery
takes exactly 10 steps, all inside day 0. Model error at step 0 on the ATM 30d call: fair value
$1.2955 at σ=0.01 versus $12.9136 at σ=0.20 — **$1,161.81 per contract**.

**My independent verification, seeds 0–19:**

| metric | with day 0 | ex-day 0 |
|---|---|---|
| median Sharpe | **+1.2232** | **−0.2355** |
| mean Sharpe | −0.1437 | −0.3691 |
| median total P&L | **+$23,413.6** | **−$2,661.6** |
| seeds with P&L > 0 | 11/20 | 10/20 |
| mean \|day 0\| vs mean \|other day\| | $25,047.7 vs $9,889.6 | **ratio 2.53** |

Seed 42: Sharpe **1.849 → 0.242**, total P&L **$33,837.79 → $3,616.13**.

**Fix:** burn in `sigma_window` steps before day 0 and exclude them from P&L, or seed the window
from a pre-sample path. Drop the `max(..., 0.01)` clamp in favour of asserting the window is
full. **Not a parameter tune — the clamp is masking an uninitialized estimator.**

**Refutation experiment:** prepend 10 real Heston steps and re-run seeds 0–19. If day-0 \|P&L\|
becomes indistinguishable from other days (ratio ≈ 1.0, not 2.53) and median total P&L stays
near $23,414, the attribution is wrong.

## 4. Informed traders trade the wrong direction on every put
**`src/market/order_flow.py:16-17, 28-36` · correctness · CONFIRMED**

`generate_trades` never receives `option_type`. It computes spot mispricing and applies
`mispricing > 0 → informed buys` to whatever leg it is called for. For a **put**, `S_true >
S_stale` means the put is worth *less* — the informed trader should sell. Three of six legs are
puts (`configs/default.py:9,11,13`).

I confirmed from the signature and body directly: the side is chosen from the sign of the
underlying move alone, with no reference to the option type.

Auditor measurement, `dir_edge = ±(BS(S_true) − BS(S_stale))` signed by counterparty direction:

| seed | call contracts / correct | put contracts / correct | informed edge at mid |
|---|---|---|---|
| 42 | 76 / **76** | 76 / **0** | **−$6,197.54** |
| 0 | 66 / 64 | 66 / **0** | −$5,597.92 |
| 3 | 588 / 588 | 588 / **0** | +$69,891.66 |

**0 of every put contract, in all six seeds, traded profitably.** The adverse-selection mechanism
is a net *subsidy* to the MM: measured MM edge versus true value is **+$28,512.75**, with
**0 of 152 filled contracts losing money**.

`tests/test_order_flow.py:64-73` enshrines the bug — it asserts `side == "buy"` when `S_true >
S_stale` with no notion of option type, so it passes on the broken behaviour.

**Fix:** pass `option_type` and flip the put branch; better, compare `bs_price(S_true, ...)`
against the quoted bid/ask and take whichever side is actually profitable.

**Refutation experiment:** flip the put branch and re-measure. Seed 42's informed edge should go
from −$6,198 to roughly +$8,000 and contracts-where-MM-lost from 0/152 to nonzero.

## 5. Noise flow does not exist — λ is applied against an annualized `dt`
**`configs/default.py:18` · `src/market/order_flow.py:20` · `src/backtest/engine.py:45` · correctness · CONFIRMED**

`dt = 1/252/78 = 5.086e-5` is a fraction of a **year**. `poisson(8.0 × 5.086e-5)` = 4.07e-4
expected arrivals per leg-step.

**My independent verification:** `8.0 × dt × 30 days × 78 steps × 6 legs = 5.71` expected noise
trades **for the entire month**. The auditors measured **zero noise trades in every seed tested
(42, 0, 1, 2, 3, 4)** — informed share **100.0%** (84/84 at seed 42). Real liquid-options
informed share is ~10–30%.

`README.md:140` documents the intent as "λ=8/day", which requires `dt` in days — so the README
and the code disagree about the unit.

**Consequence:** `spread_capture` is not liquidity-provision revenue. There is no benign
two-sided flow anywhere in the simulation. Every fill is against a counterparty that has seen
the next bar (Finding 6).

**Fix:** decide the unit and make it explicit — pass `dt_days = 1/spd`, or set `lambda_noise =
8*252`. Add a test asserting expected arrivals per day. **State the intended unit before
changing the number; this is a units fix, not a knob.**

**Refutation experiment:** if λ is genuinely meant as 8/year/leg, this is a config choice, not a
bug — but 8 trades per option per year is not a market, and the README says otherwise.

## 6. Order flow at step `idx` is generated from `prices[idx+1]` — the counterparty sees the next bar
**`src/backtest/engine.py:84-86,127` · `src/market/order_flow.py:28-36` · correctness · CONFIRMED**

**The σ estimator is clean — the comment at `engine.py:87` is correct.** At `idx`, line 88 reads
`log_ret_history` *before* line 151-152 appends `log(prices[idx+1]/prices[idx])`, so
`sigma_implied` at step `idx` is a function of `prices[0..idx]` only. Verified by tracing.

**The trade generator is not.** Line 127 passes `S_true = prices[idx+1]` — the price at the *end*
of the step being quoted into. `order_flow.py:29` computes the mispricing from it and injects a
trade on the profitable side. The counterparty conditions on a price that does not exist yet for
anyone at quote time. Adverse selection is therefore **mechanically guaranteed**, never
probabilistic: the informed trader is never wrong about the next bar's direction.

**My index trace:** `idx=0 → S_stale=prices[0]`; `idx=1 → prices[0]`; `idx≥2 → prices[idx-1]`.
Against `S_true = prices[idx+1]`, the span is 2 steps — matching `quote_staleness_steps=2`. But
the decomposition matters: the MM's observable price at the decision point is `prices[idx]`, so
the configured 2 steps is **1 step of legitimate staleness + 1 step of look-ahead granted to the
counterparty**.

**Fix:** pass `S_true = prices[idx]`. That yields a genuine 1-step asymmetry with no future
information anywhere.

**Refutation experiment:** make that change and re-run seeds 0–19. If P&L is within noise, the
look-ahead is not load-bearing and this is a documentation bug only.

## 7. Volga is added with the wrong sign for endpoint Greeks
**`src/backtest/engine.py:198,209` · correctness · CONFIRMED (sign) / PLAUSIBLE (magnitude)**

`port_eod` Greeks are evaluated at σ_EOD — the **endpoint** of the move. Expanding backwards from
the endpoint gives `V(σ_eod) − V(σ_sod) ≈ vega(σ_eod)·Δσ − ½·volga(σ_eod)·Δσ²`. Line 209 uses
**`+0.5*volga*Δσ²`**.

**My independent verification** (exact BS reprice vs both expansions, ATM 30d call):

| σ move | exact | endpoint `+` err | endpoint `−` err |
|---|---|---|---|
| 0.20→0.24 | 2.4717 | −0.0003 | **+0.0001** |
| 0.20→0.16 | −2.4717 | +0.0005 | **−0.0001** |
| 0.15→0.25 | 6.1791 | −0.0018 | **+0.0009** |
| 0.30→0.22 | −4.9426 | −0.0012 | **−0.0004** |

`−½volga` wins in all four. The sign is wrong as coded.

**Magnitude — PLAUSIBLE, not confirmed.** I verified the engine's vol terms sum to
**$44,038.34** (vega $21,597.79 + vanna −$562.08 + volga $23,002.63). The auditor measured the
*exact* vol revaluation of the EOD book at **$2,966.70**, i.e. a **15× over-attribution**, and
daily Σ\|err\| of $18,631 (vega alone) / $41,634 (`+½volga`) / $6,817 (`−½volga`). I did not
re-derive the $2,966.70. The sign claim stands on my own numbers; the 15× does not.

**Fix:** either flip the sign at `:209`, or evaluate `port_eod` at σ_SOD and keep `+½volga`. Not
both.

**Refutation experiment:** hold S and T fixed, move σ 0.20→0.24 on a single held option, compare
against both expansions with Greeks at each endpoint. If `+` wins with endpoint Greeks, I am
wrong.

## 8. "Sharpe Ratio" is not a Sharpe ratio — no capital base exists anywhere in the repo
**`src/backtest/report.py:8-13` · statistics · CONFIRMED**

1. `arr` is **dollar** daily P&L, not a return.
2. `excess = arr - 0.02/252` subtracts **$0.0000794** from a ~$1,128 mean. Measured effect on the
   output: **1.3e-07**. Dimensionally meaningless (a rate minus a dollar amount) and numerically
   zero. It exists only to make the line look like a Sharpe.
3. `np.std` defaults to **ddof=0**. Seed 42: **1.8492 (ddof=0)** vs **1.8181 (ddof=1)** — the
   headline uses the version that inflates the number by **+0.031** for free.
4. `sqrt(252)` on a dollar mean over a dollar std is dimensionally fine, so the *ratio* is
   well-defined — but it is only a Sharpe if deployed capital is constant, and nothing pins it.

**Capital-base grep** (`capital|equity|nav|notional|aum|account_value|initial_cash|book_size`)
returns **zero hits in `src/`, `configs/`, or `bench/`**. `Inventory` has no cash account, no
margin, no equity curve; `engine.run()` returns only `daily_pnl`, `daily_attribution`,
`total_pnl`, `prices`.

**What 1.849 actually is:** `sqrt(252) × mean(daily dollar P&L) / population-std(daily dollar
P&L)` over 30 days on one path. A **signal-to-noise ratio of a dollar P&L stream** — invariant
to leverage and book size. A reader will assume a return-based, capital-normalized, ddof=1
statistic. It is none of those.

**Fix:** rename to `pnl_signal_to_noise`, or add an explicit `CAPITAL` config entry, divide by
it, and use `ddof=1`.

## 9. The 30-day Sharpe cannot be distinguished from zero; most multi-seed spread is estimation noise
**`bench/BASELINE.md` · `src/backtest/report.py:8-13` · statistics · PLAUSIBLE**

Auditor measurements, seed 42 (I did not re-derive these):

- **Lo (2002), applied correctly:** the formula applies at the *observation* frequency.
  `SE(SR_daily) = sqrt((1 + 0.116486²/2)/30) = 0.183192`; annualized **SE = 2.9081**, 95% CI
  **[−3.851, +7.549]**.
  The variant `SR_ann × sqrt((1+SR_d²/2)/n)` gives 0.3388 and a tight CI [1.185, 2.513] — that is
  **wrong**, multiplying an annualized point estimate by a daily-frequency SE and understating by
  8.6×. Anyone quoting a tight CI here has made this error.
- **Bootstrap, B=50,000:** sd **2.9602**, 95% [−4.5573, +7.1514], **P(SR < 0) = 26.9%** —
  confirming the correct Lo SE to within 2%.
- **Across-path, seeds 0–19:** sd **3.1520**. Since across-path variance = true path variation +
  within-path estimation variance, true dispersion is only `sqrt(3.152² − 2.960²) = 1.08`.
  **~88% of the across-seed variance in `multi_seed.csv` is 30-observation estimation noise**, not
  strategy variability. More seeds do not fix this; **longer paths** do.
- **To pin Sharpe to ±0.5 at 95%:** N ≥ **153 independent 30-day paths ≈ 18.2 years**. The current
  20 × 30 days buys ±1.38 at best.

**Fix:** print `n`, ddof, and a bootstrap CI alongside every Sharpe; make `BASELINE.md` lead with
the interval, not the point estimate. Raise `n_days` before raising seed count.

## 10. Residual is reported against the flattering denominator, and the ✓ passes on seed 42 only
**`src/backtest/report.py:41-42` · `src/backtest/engine.py:217` · statistics · CONFIRMED**

`residual = mtm_pnl − explained` is a **plug** — defined to close, so the identity always
"holds" and any bug lands there invisibly. Findings 1 and 7 both land here, with **opposite
signs**, which is why seed 42 looks clean: Finding 1 inflates `mtm_pnl` by ~+$16.3k while
Finding 7 inflates `explained`. They partly cancel on the one seed that gets reported.

`report.py:41` divides by **net** and prints `✓` under 30%. Net is a small difference of large
offsetting flows — it can be made arbitrarily small or large without the attribution improving.

| seed | residual | /net | /gross | ✓? |
|---|---|---|---|---|
| **42** | 7,331.79 | **21.67%** | 2.09% | **✓** |
| 1 | 10,384.65 | **2264.63%** | 4.00% | ✗ |
| 2 | −34,817.20 | 57.83% | 11.89% | ✗ |
| 3 | 67,988.71 | 46.24% | 7.20% | ✗ |
| 7 | 31,031.46 | 80.32% | 15.99% | ✗ |
| 13 | 119,258.54 | 65.90% | 17.92% | ✗ |
| 99 | 109,593.32 | 66.54% | 17.12% | ✗ |
| 123 | −170,212.12 | 76.47% | 30.44% | ✗ |

**Seed 42 is the only one of eight that passes the gate.**

**Which denominator is honest:** gross, but not unconditionally. Gross is scale-invariant to
offsetting flows and does not blow up when net ≈ 0. But it is self-serving the other way — the
engine's own inflated `volga_pnl` and `hedge_cost` sit in the gross sum, so **the more the
decomposition over-attributes, the smaller residual/gross looks.** Report both, with median and
worst case across ≥20 seeds. (My baseline gross figure is $249,315.35; the auditor's per-seed
table uses a slightly different gross convention — the ratios are directionally identical.)

## 11. Half-spread is ~19% of premium, and 97.6% of it is one dimensionally-odd term
**`src/mm/quoter.py:12-15` · economics · PLAUSIBLE**

`gamma_coeff × |gamma| × contract_size` = `2.0 × 0.0105 × 100 = 2.09` dollars — a *dollar*
half-spread built from a dimensionless second derivative scaled by contract size, with no
position size, holding horizon, or vol in it. It is not a risk loading in any recognizable sense.

Auditor measurements, seed 42 (I did not re-derive):

| leg | half-spread | base | gamma term | vega term | hs/premium |
|---|---|---|---|---|---|
| 427.5 | $2.4145 | 0.05 | **2.3565 (97.6%)** | 0.0081 | **18.2%** |
| 450.0 | $2.1432 | 0.05 | **2.0903 (97.5%)** | 0.0030 | **19.0%** |
| 472.5 | $0.1144 | 0.05 | 0.0641 | 0.0003 | **261.3%** |

Full quoted width on the ATM 30d leg is **$4.29 on an $11.30 option (38% of premium)**. Real SPY
ATM 30d is $0.05–$0.15 wide → **30–70× too wide**. `spread_capture` = $22,315 over 152 contracts
= **$146.81/contract**, which is why the "informed" traders lose money to the MM.

A related consequence: `sensitivity.py`'s "10/20/50 bps" `base_spread` grid moves a term worth
2.3% of the half-spread, so **that grid axis is nearly a no-op on quote width**.

**Fix:** make the loading dimensional — e.g. `gamma_coeff × |gamma| × S² × σ² × horizon`, the
actual gamma cost of holding over the quote's life. **Do not recalibrate to a P&L target.**

## 12. Hedge cost is 100% an artifact of the 10bps assumption
**`src/mm/hedger.py:15-23` · `configs/default.py:34` · economics · CONFIRMED**

Two compounding assumptions: `transaction_cost=0.001` = **10bps of notional per hedge**, and
`hedge_shares = -portfolio_delta` — a **full flatten** every time the 25-share band is crossed,
with no dead-band.

```
hedges 1122   mean |shares| 62.8   max 736.2
total hedged notional $29,851,796
cost @10bp $29,852 | @1bp $2,985 | @0.5bp $1,493
book gross notional $1,980,000 -> hedge turnover 15.1x
```

$29,851,796 × 0.001 = **$29,851.80, exactly the reported `hedge_cost`**. Realistic institutional
SPY execution is ~0.1–0.5bps all-in, so the assumption is **20–100× too large**. Hedging fires on
**48% of 5-minute bars** on a *static* book.

**Reported as realism, not proposed for tuning** — per the project rule, the decision is yours.

**Refutation experiment:** if the 10bps proxies market impact, note the max clip is 736 shares ≈
$331k ≈ 0.0004% of SPY ADV, where impact is essentially zero. The proxy fails on its own terms.

## 13. The Glosten-Milgrom label is not earned
**`src/mm/quoter.py:12-20` · `README.md:11,142,295` · economics · CONFIRMED**

In Glosten-Milgrom the MM sets `ask = E[V | order = buy]` and `bid = E[V | order = sell]`, so the
spread is *derived* from the informed-arrival probability π, adverse selection is priced into the
quote, beliefs update Bayesianly after each trade, and the MM breaks even against informed flow
in expectation.

This code has **no π term, no conditional expectation, no Bayesian update, and no dependence on
order direction or order history**. The mid is always `fair`, regardless of what just traded.
`grep -rn "posterior\|bayes\|p_informed\|prob_informed" src/` returns nothing. It is a symmetric
risk-loaded quoter — closer to a crude Ho-Stoll / Avellaneda-Stoikov reservation-spread rule, and
even then missing the reservation-price skew that is the core of those models.

Empirically it fails the defining GM property in the *opposite* direction: the MM does not break
even against informed flow, it **wins $28,513 on 152/152 contracts** (seed 42).

**Fix:** relabel honestly, or implement GM properly (maintain π, price order-conditionally,
update after each fill). The first is cheap and honest; the second is the real project.

## 14. One staleness event fires a synchronized 6-leg block — effective n is 14, not 84
**`src/backtest/engine.py:106-127` · statistics · PLAUSIBLE**

`generate_trades` is called once per option inside the leg loop with the **same** `S_true`/
`S_stale`. The staleness condition is a property of the *underlying*, so when it fires it fires
on all six legs, each drawing an independent `U[3,12]` size.

```
legs filled in the same step -> count of steps: {6: 14}
max contracts in a single step: 26 ; steps with any fill: 14 of 2340
```

**Every one of the 14 trading events filled all six legs.** There is no step where 1–5 legs
traded. So the 84 "informed trades" are **14 independent draws**, and the effective sample size
for any inference about adverse selection is 6× smaller than the trade count suggests. This is a
large part of why the 20-seed Sharpe std is 3.07.

**Fix:** draw the informed arrival once per step at the portfolio level and route it to the leg
where the mispricing is actually exploitable. Informed traders pick the best instrument; they do
not simultaneously lift six unrelated strikes.

## 15. The attribution module is dead code, and its tests are tautological
**`src/pnl/attribution.py` · `tests/test_attribution.py` · correctness · CONFIRMED**

`grep -rn "PnLAttributor" src tests run_backtest.py` hits only the definition and the tests.
`engine.py` never imports it — attribution is hand-inlined at `engine.py:211-229`. **All five
tests exercise the module production never runs.**

Worse, they cannot fail:
- `test_components_plus_residual_equal_total` asserts `Σcomponents + residual == total`. Since
  `residual := mtm − explained` by construction, **this is a tautology** — it passes for any
  inputs, including arbitrarily wrong ones. This is exactly the false confidence that let
  Findings 1 and 7 survive.
- `test_no_activity_zero_pnl` and `test_nonzero_vanna_volga_pass_through` assert on hand-fed
  `mtm_pnl` values.
- Only `test_short_call_theta_pnl_positive` tests real behaviour — and it asserts theta is
  positive for a **short** call, while the engine's actual book is net **long** every day
  (Finding 17). The test encodes the intended economics; the simulator delivers the opposite, and
  nothing notices.

The module's docstrings (`attribution.py:15-16`) say vanna/volga arrive as **per-step sums** —
precisely the basis the engine rejected at `engine.py:203-206`. Two implementations, one dead,
silently divergent in basis, and the sign convention specified nowhere (which is how Finding 7
survived).

**Fix:** delete the module and test the engine's path directly, or refactor the engine to call
it. Replace the tautological test with an independently-computed cash+mark portfolio value that
`mtm_pnl` must match — the assertion that would have caught Finding 1 on day one.

**Refutation experiment:** flip the sign of `hedge_cost_total` at `engine.py:216` and run
`pytest tests/test_attribution.py`. Prediction: all five still pass.

## 16. `spread_capture` is marked against a price the book is never valued at
**`src/backtest/engine.py:111,136,146` · correctness · PLAUSIBLE**

`fair = bs_price(S_stale, ...)`, but `_book_value` only ever uses `S_sod`/`S_eod`. `S_stale` never
appears in the book. Booked `spread_capture` = $22,315.22; recomputed against `bs_price(S_true,
...)` = $28,512.75 — a difference of **$6,197.53, 27.8% of the booked figure**, which currently
sits in `residual` and accounts for **~85% of the $7,331.79 residual at seed 42**.

That quantity is the **staleness P&L** — economically first-class, and it should be its own line
item rather than buried in the plug.

**Also measured:** 80 of 84 trades were size-truncated by `RiskLimits`, dropping **481 of 633
offered contracts (76%)**. Only 152 contracts filled in a month.

**Fix:** book `spread_capture` against `bs_price(S_true, ...)` and add
`staleness_pnl = (fair_stale − fair_true) × signed_size × cs` as a separate component.

## 17. No inventory skew, and the throttle is symmetric — the book cannot unwind
**`src/mm/quoter.py:17-20` · `src/risk/limits.py:19-20` · economics · CONFIRMED**

`quote()` returns `fair ± hs` with no dependence on inventory — the quoter is never even passed
the position. Nothing pulls the book toward flat. Compounding it, `leg_headroom = max(0,
max_contracts_per_leg - abs(current_leg_position))` uses `abs(...)` and returns a single `size`
used for **both** sides, so at `|position| = 20` the MM stops quoting the **position-reducing**
side exactly as hard as the increasing side. The book freezes and can only exit via expiry.

Terminal inventory: seed 42 **+8,+8,+8,+8,+6,+6 = +44, all six legs long** (my own measurement).
Seeds 0 and 2 all short (−36); seed 3 **pinned at +20 on every leg (+120)**; seed 4 pinned at
−120. **The book is never mixed** — all six legs always end the same sign, a direct consequence
of Finding 14. Seed 3 ends +$147,022 and seed 4 ends −$148,121: the entire outcome is the sign of
the day-0 block trade.

**Fix:** skew the mid against inventory, and split `adjusted_quote_size` into `bid_size`/
`ask_size` applying leg headroom **signed** so the reducing side gets full size.

## 18. The sensitivity grid manufactures ~+0.7 to +1.3 Sharpe from selection alone
**`src/backtest/sensitivity.py:139-205` · statistics · PLAUSIBLE**

Within-combo noise in `results/sensitivity.csv` gives SE of a combo's mean Sharpe = **0.6606**.
Under the null that all 27 combos are identical, `E[max of 27] − true` = **+1.319** (ρ=0),
+1.104 (ρ=0.3), +0.836 (ρ=0.6), +0.591 (ρ=0.8).

Observed: best `mean_sharpe` **0.7728**, grid mean **0.1026** → **best − mean = +0.670**. The
across-combo std is **0.4913**, *smaller* than the 0.6606 single-combo SE. **The entire observed
range from best to worst is consistent with — in fact narrower than — what identical combos on
five shared seeds would produce by chance.** The ranked table carries no demonstrated information
about parameter quality.

This matters because `configs/default.py:1` says "set once, never tuned per run," yet
`sensitivity.py` prints a "Best combo" line that reads as a recommendation. Acting on it would
import +0.67 to +1.3 of Sharpe that does not exist, validated against a benchmark whose own SE is
±2.9 (Finding 9). **The two errors compound: a search with ~1.3 of selection bias, checked
against a benchmark with 2.9 of noise, can "prove" almost anything.**

**Refutation experiment:** re-run the grid on disjoint seeds (100–104). Under the null the
current top combo's expected rank is uniform — 14th of 27.

## 19–20. Lower-severity confirmed items

**19. Gamma P&L uses post-move Greeks** (`engine.py:151,166,174-177`, correctness, immaterial).
`log_ret` spans `[idx, idx+1]` but `port_now` is evaluated at `prices[idx+1]` — after the move. A
gamma term needs pre-move Greeks. The formula is otherwise dimensionally sound. Measured effect:
**$3.75, 0.19% of the gamma term, 0.011% of total**. Reported so nobody mistakes it for a
residual explanation — it explains 0.05% of the residual.

**20. Intra-step limit overshoot** (`engine.py:104,117-130`, correctness, latent). `port_g` is
computed once per step, so leg 6 quotes against Greeks from before legs 1–5 traded; and
`fill_size = min(trade["size"], size)` reuses `size` for every trade without decrementing.
Portfolio `|vega|` reaches **163,415 against a 50,000 cap (3.27×)**. The per-leg part is
**latent** — unreachable today because noise flow is absent (Finding 5), demonstrated only by
restoring it (`maxlegpos=21` vs cap 20). It goes live the moment Finding 5 is fixed.

---

## Refuted — tested and false

These were raised as hypotheses and **do not survive**. Recording them so they don't get
re-raised.

1. **Hedge cash flows are double-counted.** REFUTED. Hand-worked: buy 100 shares at S=450,
   tc=0.001 → fill 450.45, `realized_pnl` −45,045, position +100. Engine computes `underlying_pnl
   = 100·S_eod`, total `100·S_eod − 45,045` — cash out plus mark, **counted once**.
2. **Hedge transaction cost is double-charged.** REFUTED. The $45 is inside the cash flow and thus
   inside `mtm_pnl` exactly once. `hedge_cost_total` at `engine.py:216` is an **explanation** term
   subtracted from `explained`, not from `mtm_pnl`. The sign is correct. Test: run with
   `transaction_cost=0` — total P&L should rise by exactly $29,851.80, not $59,703.60.
3. **SOD/EOD sigma revaluation gap.** REFUTED. `max over 30 days of |sod_book(d) −
   eod_book(d−1)| = 0.00`, exactly. Nothing mutates `sigma_implied` between `engine.py:182` and
   `:74` of the next iteration. `sigma_sod` at `:71` equals the σ `sod_book` was marked at, so
   `delta_sigma` at `:190` **does** measure the σ move embedded in the book change. The comment at
   `:188-189` is accurate.
4. **`fill_size` vs `spread_capture` size mismatch.** REFUTED. Both use `fill_size`
   (`engine.py:134` vs `:137`, `:143` vs `:146`).
5. **The BS-on-Heston model mismatch is a material driver.** REFUTED — my own measurement. For
   the actual 6-leg universe at config params:

   | leg | Heston CF | BS@0.20 | gap | half-spread | gap/hs |
   |---|---|---|---|---|---|
   | 427.5 30d put | 4.2572 | 3.7769 | +0.4803 | 1.9152 | **0.25×** |
   | 450.0 30d call | 12.8076 | 12.9136 | −0.1060 | 2.6380 | **0.04×** |
   | 450.0 30d put | 11.7374 | 11.8434 | −0.1060 | 2.6380 | **0.04×** |
   | 472.5 30d call | 4.0647 | 4.7544 | −0.6897 | 2.1665 | **0.32×** |
   | 427.5 60d put | 8.1216 | 7.5024 | +0.6191 | 1.5751 | **0.39×** |
   | 450.0 60d call | 18.2978 | 18.5609 | −0.2631 | 1.8931 | **0.14×** |

   **The model gap is 0.04×–0.39× of the quoted half-spread on every leg** — it never exceeds it.
   The MM is not systematically quoting on the wrong side of true value because of Heston-vs-BS.
   This is a real but second-order effect, far behind Findings 1–6.
6. **Greeks are inaccurate.** REFUTED. Analytical vs finite-difference at S=K=450, T=30/252,
   σ=0.20: delta 1.8e-09, gamma 6.8e-08, vega 2.0e-10. Off-ATM (K=427.5/472.5): vanna 3.2e-08 /
   6.8e-08, volga 1.2e-07 / 1.7e-07. (At ATM, vanna and volga are ≈0, so relative error there is
   meaningless — not a defect.) Only `theta` shows 5.4e-03, which is `theta_fd`'s `h = 1/365`
   truncation against a 252-day year — a **minor units inconsistency in the test helper**, not in
   the analytical Greek.
7. **The pricing methods disagree.** REFUTED. Across K ∈ {427.5, 450, 472.5} × T ∈ {30d, 60d},
   errors vs BS: CRR(200) ≤ $0.022, MC(400k) ≤ $0.035, Heston CF (ξ→0) ≤ **$0.0002**. The CF
   implementation is excellent.
8. **The Heston variance scheme is materially biased.** REFUTED at these parameters. Feller
   `2κθ = 0.1600 ≥ ξ² = 0.0900` is **satisfied**; across 400 paths the minimum variance ever
   reached is **0.00183**, nowhere near the `1e-8` floor, so the truncation choice never binds.
   `E[S_T]` bias is **−0.016%**. Worth noting for the record that the scheme is *not* textbook
   full truncation — it uses `max(v,0)` in drift and diffusion but **carries** `max(v+dv, 1e-8)`,
   which is absorption rather than full truncation — but it is immaterial here. It would matter
   under Feller-violating parameters.

---

## The one-line summary

At seed 42 the simulator quotes on **1.75% of opportunities**, trades on **14 steps out of
2,340**, accumulates a **static +44-contract long-options book** on day 0–1 while its volatility
estimator is wrong by 20×, and then delta-hedges it 1,122 times at a cost assumption 20–100× too
high — against counterparties who see the next bar, trade the wrong way on every put, and are
outnumbered by noise traders 820:1 in the wrong direction. The reported total P&L is then **48%
higher than a cash-and-mark ledger of the same trades**, and the residual looks tidy only because
two large errors of opposite sign happen to cancel on this seed.

**No number in `bench/BASELINE.md` currently describes market making.**
