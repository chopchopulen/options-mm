# PHASE 3 — Reconciliation against the outside review

Written after `audit/FINDINGS.md` was committed (`923cf3a`). Each claim independently verified.
Nothing accepted on authority.

---

## Claim 1 — "The P&L may be a long-vol position rather than market making"

Stated evidence: vega is $21,598 of $33,838 (64%), cumulative theta is NEGATIVE (implying a net
LONG options book, backwards for a spread-capturing MM), gamma is positive, win rate 46.7%.

### Verdict: **CONFIRMED** — and the conclusion is stronger than the argument given for it.

The arithmetic checks out: vega $21,597.79 / total $33,837.79 = 63.8%; cumulative theta is
−$12,002.57; gamma +$2,006.81; win rate 46.7%. The inference from negative theta to a long book
is correct — `analytical.theta` is per-year and negative for long options, and
`engine.py:176` accumulates `port_now["theta"] * step_dt`, so a negative sum means net long.

**Directly measured (my own instrumentation, seed 42), the book is long on every leg:**

```
terminal legs: 427.5p30 +8, 450c30 +8, 450p30 +8, 472.5c30 +8, 427.5p60 +6, 450c60 +6
net +44 contracts, all six legs long, monotonically from day 0
```

Per the attribution auditor: **0 of 30 days have portfolio vega < 0; 30 of 30 days have portfolio
theta < 0.** No leg is ever short. Across seeds the book is never *mixed* — seeds 0 and 2 end all
short (−36), seed 3 pins at **+20 on every leg (+120)**, seed 4 at −120.

### Where the outside review understates the problem

It frames this as "the P&L *may be* a long-vol position rather than market making." Three
measurements say the situation is more specific than that:

1. **The MM is not quoting at all for 98.25% of the run.** The vega cap binds at ~19 contracts
   for the entire book, so `adjusted_quote_size` returns 0 on **13,794 of 14,040** calls
   (96.7% of refusals are the vega cap; zero are the per-leg cap). Fills occur on **14 steps out
   of 2,340**. This is not a long-vol *strategy* — it is a static book bought in 14 blocks on day
   0–1 and then held. (`FINDINGS.md` #2)
2. **The vega number it cites is itself inflated ~15×.** Engine vol terms total $44,038.34
   (vega + vanna + volga) against an exact revaluation of the EOD book of $2,966.70. Volga alone
   is $23,002.63 — 68% of the month's P&L — and is added with the **wrong sign** for endpoint
   Greeks. So "vega is 64% of P&L" is quoting a number that is partly an attribution artifact.
   (`FINDINGS.md` #7)
3. **The 64% is computed against a total that is itself 48% too high.** (`FINDINGS.md` #1)

So: the claim's *conclusion* is right and its *evidence* is real, but the underlying quantities
it reasons from are each individually defective. It arrives at the right answer through numbers
that don't mean what they appear to mean.

### Root cause of the long bias

The outside review does not identify one. Measured, it is the interaction of three defects:
`rho=-0.7` makes down-moves larger and more frequent → the informed-trader rule (which ignores
option type, `FINDINGS.md` #4) systematically sells puts into the MM's bid → `Quoter.quote` is
symmetric around `fair` with no inventory skew and `RiskLimits` throttles the position-reducing
side as hard as the increasing side (`FINDINGS.md` #17), so nothing pushes back.

### Proposed ablation — NOT RUN, as instructed

Design constraints: every arm runs the **same 20 seeds (0–19)**; report median and IQR, never a
single seed; report both total P&L and the ex-day-0 total P&L (`FINDINGS.md` #3), since day 0
otherwise dominates every arm identically and masks the differences.

| arm | what changes | isolates | prediction if Claim 1 is right |
|---|---|---|---|
| **A. Full MM** (control) | nothing — baseline | — | median P&L +$23,414, ex-d0 −$2,662 |
| **B. Hold-the-book, delta-hedged only** | acquire the identical day-0/day-1 position, then set `desired_quote_size=0` for the rest of the run; keep hedging | separates *carry of the book* from *ongoing market making* | **If B ≈ A, the P&L is entirely the static book and there is no MM edge.** Given only 14 fill-steps exist, I expect B to reproduce >90% of A |
| **C. No informed traders** | `staleness_threshold = ∞` | separates spread capture from adverse selection | Given noise flow is ~5.7 trades/month (`FINDINGS.md` #5), C should trade **almost nothing** and produce ≈$0. That is itself the finding: it proves all flow is informed |
| **D. No hedge cost** | `transaction_cost = 0` | isolates the 10bps assumption | P&L should rise by **exactly $29,851.80** at seed 42. Any other number means the cost is not counted once (a check on the refuted double-charge hypothesis) |

**Two arms the outside review did not propose, which I think matter more:**

| arm | what changes | isolates |
|---|---|---|
| **E. Cash-accounting P&L** | compute `total_pnl` from a cash+mark ledger instead of `realized_pnl` | `FINDINGS.md` #1 — whether the whole comparison is being run on a wrong number. **This must run first**; arms A–D are uninterpretable until the P&L measure is correct |
| **F. Warm-started sigma** | pre-roll 10 Heston steps into `log_ret_history` | `FINDINGS.md` #3 — removes the day-0 artifact from every other arm |

**Ordering matters:** E and F are corrections, not ablations. Running A–D on top of a P&L that is
48% wrong and 89% warm-up artifact would produce four confidently-wrong numbers. E → F → A → B →
C → D.

**Falsification condition, stated up front:** Claim 1 is refuted if arm B produces materially
*less* than arm A across seeds — that would mean ongoing quoting contributes real edge and the
book is not merely being carried.

---

## Claim 2 — "Sharpe 1.85 is 30 daily observations on ONE Heston path and has no confidence interval"

### Verdict: **CONFIRMED**, and understated in one respect, overstated in none.

### Exactly how Sharpe is computed (`src/backtest/report.py:8-13`)

```python
arr    = np.array(daily_pnl)                  # DOLLAR P&L, not returns
excess = arr - risk_free_daily                # risk_free_daily = 0.02/252 = 7.94e-5
return float(np.sqrt(252) * np.mean(excess) / np.std(excess))   # ddof=0
```

Four separate problems, all verified:

1. **It is not a Sharpe ratio.** `arr` is dollars, not returns. `grep -rniE
   "capital|equity|nav|notional|aum|account_value|initial_cash|book_size"` returns **zero hits in
   `src/`, `configs/`, or `bench/`**. `Inventory` has no cash account, no margin, no equity curve.
   There is no denominator anywhere in the repo.
2. **The risk-free subtraction is meaningless and numerically zero.** Subtracting $0.0000794 from
   a $1,127.93 mean changes the output by **1.3e-07**. It is dimensionally incoherent (a rate
   minus a dollar amount) and exists only to make the line look like a Sharpe.
3. **Yes, it is annualized** — by `sqrt(252)`. That scaling is dimensionally fine (dollar units
   cancel in the ratio), so the *ratio* is well-defined; it is only a *Sharpe* if deployed capital
   is constant, which nothing establishes.
4. **`ddof=0`.** Seed 42: **1.8492 (population)** vs **1.8181 (sample)**. The headline uses the
   version that inflates the figure by **+0.031** for free.

**What 1.849 actually is:** `sqrt(252) × mean(daily dollar P&L) / population-std(daily dollar
P&L)`, 30 days, one path, seed 42. A **signal-to-noise ratio of a dollar P&L stream** — invariant
to leverage and book size. Double every position and it is unchanged.

### What a defensible interval would require

| method | result (seed 42) |
|---|---|
| Lo (2002), applied at daily frequency then annualized | **SE = 2.9081**, 95% CI **[−3.851, +7.549]** |
| Bootstrap, 50,000 resamples of the 30 daily obs | sd **2.9602**, 95% **[−4.557, +7.151]**, **P(SR < 0) = 26.9%** |
| Across-path, seeds 0–19 | sd **3.1520**; 95% CI for the mean **[−1.525, +1.238]** |

**A trap worth naming.** The Lo formula must be applied at the *observation* frequency and then
annualized: `SE(SR_daily) = sqrt((1 + SR_d²/2)/n) = 0.1832`, ×`sqrt(252)` = **2.9081**. Writing it
as `SR_ann × sqrt((1 + SR_d²/2)/n)` gives 0.3388 and a tight-looking CI of [1.185, 2.513] — that
is **wrong by a factor of 8.6**, and it is the natural mistake to make here. Anyone who reports a
narrow interval for this Sharpe has probably made it.

**The point the outside review misses entirely:** adding seeds does not help. Across-path variance
= true path variation + within-path estimation variance, so true dispersion is only
`sqrt(3.152² − 2.960²) = 1.08`. **About 88% of the across-seed spread in `multi_seed.csv` is
30-observation estimation noise, not strategy variability.** Running 200 seeds of 30 days would
not tighten the estimate of the underlying Sharpe much; running **longer paths** would.

**Requirement to pin Sharpe to ±0.5 at 95%:** N ≥ **153 independent 30-day paths ≈ 4,590 trading
days ≈ 18.2 years**, or equivalently a single path of ~4,580 days. The current 20 × 30 = 600 days
buys **±1.38 at best**. And all of this is moot until `total_pnl` itself is correct
(`FINDINGS.md` #1).

---

## Claim 3 — "The 88% residual is measured against NET P&L, which is a small difference between large gross components (~$88k gross)"

### Verdict: **PARTIALLY CORRECT** — the methodological point is right; the numbers do not
reproduce at this commit.

**The 88% does not exist at `dbdc9cf`.** Measured, seed 42:

```
residual $7,331.79  =  21.67% of net ($33,837.79)
                    =   2.94% of gross flow ($249,315.35)
```

`report.py:42` prints **`✓`** because 21.67% < 30%. Nor does the "~$88k gross" figure reproduce —
my gross flow (Σ over days of Σ|component|) is **$249,315.35**. The 88% and the $88k appear to
come from a different commit, a different seed, or a different definition. **They should not be
quoted against this baseline.** (Coincidentally, `hedge_cost` alone is −$29,852 = 88.2% of net —
possibly the source of the confusion.)

### The methodological point, however, is exactly right, and matters more than the number

Net P&L is a small difference between large offsetting flows, so it is an unstable denominator —
it can be driven to zero without the attribution improving at all. Measured across seeds:

| seed | residual | /net | /gross | `✓` printed? |
|---|---|---|---|---|
| **42** | 7,331.79 | **21.67%** | 2.09% | **✓** |
| 1 | 10,384.65 | **2264.63%** | 4.00% | ✗ |
| 2 | −34,817.20 | 57.83% | 11.89% | ✗ |
| 3 | 67,988.71 | 46.24% | 7.20% | ✗ |
| 7 | 31,031.46 | 80.32% | 15.99% | ✗ |
| 13 | 119,258.54 | 65.90% | 17.92% | ✗ |
| 99 | 109,593.32 | 66.54% | 17.12% | ✗ |
| 123 | −170,212.12 | 76.47% | 30.44% | ✗ |

**Seed 42 is the only one of eight that passes the 30% gate.** Seed 1 shows the pathology
precisely: total P&L is −$458.56, so residual/net reads **2264.63%** while residual/gross is a
benign 4.00%. The reported baseline is the single most flattering seed in the set.

### Which denominator is honest — neither, alone

The outside review implies gross is the honest one. It is *better*, but it is self-serving in the
opposite direction, and I want that on the record:

- **Net** is unstable: it explodes when P&L ≈ 0 and can be shrunk arbitrarily.
- **Gross** is scale-invariant to offsetting flows, but **the engine's own over-attributed
  components sit in the gross sum**. `volga_pnl` ($23,003, wrong sign) and `hedge_cost` ($29,852,
  an artifact of the 10bps assumption) inflate the denominator, so **the more the decomposition
  over-attributes, the smaller `residual/gross` looks.** A decomposition can improve its own grade
  by inventing components.
- Simple demonstration: multiply everything by a constant and both ratios are invariant. Now *add*
  an offsetting spurious ±$1M pair — `residual/gross` collapses toward 0 while `residual/net` is
  unchanged. Neither is sufficient.

**Recommendation:** report **both**, plus the median and worst case across ≥20 seeds, and add a
hard assertion that `mtm_pnl` equals an independently computed cash+mark portfolio change — which
the code would currently **fail** (`FINDINGS.md` #1). A residual gate on a single seed is not a
test.

---

## Claim 4 — "The listed residual causes (discrete hedging, IV estimator lag, missing Vanna/Volga, BS-on-Heston mismatch) are untested hypotheses"

### Verdict: **CONFIRMED** that they are untested — and **one of the four is now tested and
REFUTED**, while the list omits the actual dominant cause.

Two of the four are also stale as written: **Vanna and Volga are not missing** — they are computed
at `engine.py:208-209` and reported as line items ($−562.08 and $23,002.63). The problem is the
opposite of absence: volga is added with the **wrong sign** and the vol terms over-attribute.

### Controlled experiments, one per hypothesis

Each is designed so it **could fail**. Every arm: seeds 0–19, report median and IQR, report
residual against **both** net and gross, and run only after `FINDINGS.md` #1 (cash accounting) is
corrected — otherwise the residual being decomposed is not the right residual.

| # | Hypothesis | Controlled experiment | Refutation condition |
|---|---|---|---|
| **H1** | Discrete hedging | Sweep `steps_per_day` ∈ {78, 156, 312, 624} holding all else fixed. Discretization error scales as O(√dt), so residual/gross should fall ~√2 per doubling | If residual/gross is flat in `steps_per_day`, discrete hedging is not the cause. **This is a convergence test, not a tuning sweep — the deliverable is the slope, not the best value** |
| **H2** | IV estimator lag | Sweep `sigma_uncertainty_window` ∈ {10, 50, 100, 500}, and separately replace the rolling estimator with the **known Heston instantaneous** `sqrt(v_t)` (an oracle arm — not a strategy, a diagnostic) | If the oracle arm leaves residual/gross unchanged, estimator lag is not the cause. Prediction: step-by-step and EOD vol attribution **converge** as the window widens; that convergence is the actual test that vol attribution means anything |
| **H3** | Vanna/Volga | Three arms: (a) drop both terms, (b) as coded `+½volga`, (c) sign-corrected `−½volga`. Compare each day's vol terms against an **exact reprice** of the EOD book from σ_SOD to σ_EOD | Already partly answered: `−½volga` beat `+½volga` on all four test moves I ran. Refuted if, with endpoint Greeks, `+` produces smaller error against the exact reprice |
| **H4** | BS-on-Heston mismatch | **ALREADY RUN — REFUTED.** Priced all 6 legs under the Heston CF at config params vs BS@0.20 | Gap is **0.04×–0.39× of the quoted half-spread on every leg**, never exceeding it. Max gap $0.69 on the 472.5 30d call against a $2.17 half-spread. This is real but second-order |

**Detail for H4 (the refuted one):**

| leg | Heston CF | BS@0.20 | gap | half-spread | gap/hs |
|---|---|---|---|---|---|
| 427.5 30d put | 4.2572 | 3.7769 | +0.4803 | 1.9152 | 0.25× |
| 450.0 30d call | 12.8076 | 12.9136 | −0.1060 | 2.6380 | 0.04× |
| 450.0 30d put | 11.7374 | 11.8434 | −0.1060 | 2.6380 | 0.04× |
| 472.5 30d call | 4.0647 | 4.7544 | −0.6897 | 2.1665 | 0.32× |
| 427.5 60d put | 8.1216 | 7.5024 | +0.6191 | 1.5751 | 0.39× |
| 450.0 60d call | 18.2978 | 18.5609 | −0.2631 | 1.8931 | 0.14× |

### The cause the list omits, which is larger than all four

**Stale-mark spread capture.** `fair = bs_price(S_stale, ...)` (`engine.py:111`) but the book is
only ever valued at `S_sod`/`S_eod` — `S_stale` never appears in `_book_value`. Booked
`spread_capture` = $22,315.22; recomputed against `bs_price(S_true, ...)` = $28,512.75. The
**$6,197.53 difference is ~85% of the entire $7,331.79 residual at seed 42.**

That quantity is the *staleness P&L* and is economically first-class — it belongs as its own line
item, not in the plug. **Experiment:** re-run with `quote_staleness_steps=0`; booked and
true-price spread capture must become identical and the residual must fall by exactly $6,197.53.
If it doesn't move by that amount, the decomposition is wrong.

### A structural point about all of this

`residual := mtm_pnl − explained` (`engine.py:217`) is a **plug**. It is defined to close, so any
bug anywhere lands in it silently and the identity always "holds". At seed 42 the residual looks
tidy because **Findings #1 and #7 have opposite signs and partially cancel**: #1 inflates
`mtm_pnl` by ~+$16.3k while #7 inflates `explained`. Chasing the residual's *magnitude* on one
seed is therefore not a diagnostic — a smaller residual can mean two larger errors cancelling
better. The right test is the assertion that `mtm_pnl` matches an independent cash+mark ledger,
which the code currently fails.

---

## What Phase 2 found that this outside list missed

Ranked by materiality. This is the part I'd weight most.

1. **`total_pnl` is arithmetically wrong — 48% too high** (`FINDINGS.md` #1). `fill_option` books
   an inception-to-date realized *gain* where the MTM identity needs a *cash flow*. Opening
   premium never enters P&L; closing trades book prior days' gains twice. Independently
   reproduced: reported $33,837.79 vs cash+mark $17,566.31. **On seed 3 it flips the sign —
   +$147,022 reported against −$37,249 true.** Every claim in the outside review is computed on
   this number, including its own "vega is 64% of P&L."

2. **The MM stops quoting at step 4 of 2,340** (`FINDINGS.md` #2). The vega cap binds at ~19
   contracts for the whole book. Independently reproduced: **246 nonzero quotes of 14,040
   (1.75%)**, 13,344 refusals from the vega cap, **0** from the per-leg cap. Fills on 14 steps.
   This is the mechanism behind Claim 1, and the outside review does not identify it.

3. **89% of the P&L is a warm-up artifact** (`FINDINGS.md` #3). `log_ret_history = [0.0]*10` →
   first sigma clamped to 0.01 against a true 0.20. Independently reproduced across seeds 0–19:
   **median Sharpe +1.2232 → −0.2355, median P&L +$23,414 → −$2,662** when day 0 is dropped.
   This alone reverses the sign of the headline result and is invisible in every number the
   outside review cites.

4. **Informed traders trade the wrong direction on every put** (`FINDINGS.md` #4).
   `generate_trades` never receives `option_type`. **0 of every put contract in all six seeds
   tested traded profitably.** The "adverse selection" is a net *subsidy*: MM edge vs true value
   **+$28,512.75, with 0 of 152 contracts losing**. And `tests/test_order_flow.py:64-73` asserts
   the broken behaviour, so it can never be caught.

5. **Noise flow does not exist** (`FINDINGS.md` #5). `lambda_noise=8.0` is multiplied by an
   annualized `dt = 5.086e-5`, giving **5.71 expected noise trades for the entire month**;
   measured zero in every seed. Informed share **100%**. `README.md:140` says "λ=8/day", so code
   and docs disagree on the unit. There is no benign flow to earn a spread from — which
   invalidates the framing of `spread_capture` as market-making revenue.

6. **Order flow reads `prices[idx+1]`** (`FINDINGS.md` #6). The counterparty conditions on a price
   that exists for nobody at quote time. The configured `quote_staleness_steps=2` decomposes as
   1 step of legitimate staleness + **1 step of look-ahead**. Separately: the σ estimator's
   no-look-ahead claim **is** correct — I verified the index arithmetic. The comment is right
   about the thing it comments on and silent about the thing that's wrong.

7. **The attribution tests are tautological and test dead code** (`FINDINGS.md` #15).
   `src/pnl/attribution.py` is never imported by the engine; all five tests exercise it.
   `test_components_plus_residual_equal_total` asserts `Σcomponents + residual == total` where
   `residual := mtm − explained` **by construction** — it passes for any inputs, including
   arbitrarily wrong ones. This is the mechanism by which Findings #1 and #7 survived a
   70-test green suite.

8. **The sensitivity grid manufactures +0.7 to +1.3 Sharpe from selection alone**
   (`FINDINGS.md` #18). Best-of-27 under the null with the observed noise. The observed
   across-combo std (0.4913) is *smaller* than the single-combo SE (0.6606) — the entire ranked
   table is consistent with no effect. Directly relevant to this project's never-tune rule: a
   search carrying ~1.3 of selection bias, validated against a benchmark with ±2.9 of noise, can
   "prove" almost anything.

**And one thing the outside review was right to be suspicious of that turned out clean:** the
pricing stack. CRR(200) within $0.022 of BS, MC(400k) within $0.035, Heston CF within **$0.0002**;
analytical vs finite-difference Greeks agree to 1e-7 or better (vanna/volga checked off-ATM,
where they are nonzero). Feller is satisfied (`2κθ = 0.16 ≥ ξ² = 0.09`) and the minimum variance
across 400 paths is 0.00183 — nowhere near the `1e-8` floor, so the discretization scheme never
binds. **The quantitative core is not the problem. The accounting, the risk limits, and the
market model are.**
