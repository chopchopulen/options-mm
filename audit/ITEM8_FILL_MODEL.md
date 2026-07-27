# ITEM 8 — Fill probability is independent of quote width

Investigation and options. **Nothing here is implemented.** Commit `a298b22` + ITEM 7.

## Verdict: CONFIRMED, and stronger than stated

Your reading is exactly right, and it is not approximate — fill counts are *literally
invariant* to quote width.

### Code evidence

`src/market/order_flow.py:19-26` (noise):

```python
n_noise = self.rng.poisson(self.lambda_noise * dt_days)
for _ in range(n_noise):
    side  = "buy" if self.rng.random() < 0.5 else "sell"
    size  = int(self.rng.integers(1, self.max_noise_size + 1))
    price = ask if side == "buy" else bid
```

`bid` and `ask` enter only to be *recorded* as the trade price. Arrival count, side and
size are drawn without reference to them. There is no reservation price, no competing
quote, no elasticity.

`src/market/order_flow.py:33-44` (informed): the trigger is
`abs(mispricing) > self.staleness_threshold` — a property of the **underlying only**. The
informed trader's edge is never compared to the spread it must cross, so it trades even
when the quote is wider than the edge.

### Measured (seed 42, sweeping `base_spread` over a 16× range)

| base_spread | fills | contracts | spread_capture | total_pnl |
|---|---|---|---|---|
| 0.025 | **5335** | **19388** | 325,495 | −917,241 |
| 0.050 | **5335** | **19388** | 373,965 | −868,771 |
| 0.100 | **5335** | **19388** | 470,905 | −771,831 |
| 0.200 | **5335** | **19388** | 664,785 | −577,951 |
| 0.400 | **5335** | **19388** | 1,052,545 | −190,191 |

Fill count and contract count are **identical to the unit** across the whole sweep. And
`spread_capture` is exactly linear: `19,388 contracts × 100 × Δbase_spread` reproduces
every increment to the dollar (0.025 → 0.05 predicts +$48,470; observed +$48,470).

So the P&L is, exactly:

```
total_pnl = (everything else) + contracts × 100 × half_spread
```

with `contracts` a constant. **P&L is unbounded and linear in a quantity we choose.** That
is why every correct fix made the number worse: each one removed something that was
suppressing fills or masking this term, and none of them touched the term itself.

It also means the ITEM 7 calibration — and any future spread calibration — is load-bearing
in a way it should not be. In a real market, quoting too wide costs you the fill. Here it
is free money, and quoting narrower is pure loss.

### One extra finding

7.2% of informed fills (302 of 4,216, seed 42) have an edge **smaller than the half-spread
they cross** — the informed trader knowingly trades at a loss. Mean informed edge is
$0.5265 against a mean half-spread of $0.1919 (ratio 2.74).

---

## Options

Ordered by how much has to be invented. "Derivable" means it comes from quantities already
in the repo; "invented" means a new free constant with no anchor, which under the project
rules is a tuning knob.

### Option A — Informed traders require edge > half-spread

**Requires: nothing new.** Both quantities already exist at the call site.

```python
if abs(option_edge_value) > half_spread:   # currently: if abs(mispricing) > threshold
```

Informed fills become width-dependent immediately: quote wider and the informed stop
crossing. This is arguably a **bug fix rather than a model addition** — a trader defined as
informed should not knowingly cross a spread larger than its edge, and 7.2% of current
informed fills do exactly that.

- **Covers:** informed flow only. Noise flow stays perfectly inelastic.
- **Cost:** none. No parameters, no new model.
- **Limitation:** informed flow is *adverse*, so making it elastic reduces the MM's losses
  as spread widens. It fixes the sign of the elasticity but the noise term — the actual
  source of the unbounded P&L — is untouched. **Necessary, not sufficient.**

### Option B — Counterparty reservation price

Each arriving noise trader draws a reservation price; it trades only if the quote is
inside it.

```
r ~ fair + eps,  eps ~ D(scale)     trade iff (buy) ask <= r,  (sell) bid >= r
```

- **Requires:** a distribution `D` and a scale. The shape is a modelling choice
  (exponential and logistic are the usual ones and give different tail behaviour).
- **Derivable?** *Partially.* The scale can be anchored to the option's own price
  uncertainty over the holding horizon — the same `(ξ/2)·√τ · vega` quantity ITEM 7 already
  computes — on the argument that a counterparty's willingness to pay is dispersed by
  roughly the amount the price itself moves over that horizon. That is a real argument, not
  a fitted number, but it is an *assumption* about counterparty behaviour, not a
  consequence of anything in the model.
- **Cost:** one distributional assumption; no free numeric constant if anchored as above.
- **Gives:** smooth elasticity, an interior optimum in spread width, and a well-posed
  question "what width maximises expected P&L" that currently has the answer "infinity".

### Option C — Competing-quote reference (NBBO)

Model a competitor quoting at half-width `h*`; the MM fills only when it is at or inside
that quote (or wins a share of flow proportional to how far inside it is).

- **Requires:** `h*`, and — if you want partial fills rather than winner-take-all — a
  competitor count or a queue/share rule.
- **Derivable?** `h*` can be set to the carry-cost half-spread from ITEM 7, on the argument
  that a rational competitor charges the same risk premium. But then the MM is quoting
  exactly at `h*` by construction and the fill rule is degenerate at equality; breaking the
  tie requires a dispersion parameter or a competitor count, **and that is invented.**
- **Cost:** one or two invented constants, or a competition model that is itself a research
  project.
- **Gives:** the most realistic microstructure, and it is the natural home for the
  Glosten-Milgrom rebuild (finding 13) — GM needs a reference value to condition on.

### Option D — Width-dependent fill probability

```
p_fill = exp(-half_spread / h0)
```

- **Requires:** `h0`.
- **Derivable?** Only by setting `h0` to the carry-cost half-spread, which makes
  `p_fill = exp(-1)` at the calibrated width — a constant, self-referential and not really
  a model of anything. Otherwise `h0` is **invented**.
- **Cost:** cheapest to implement, weakest justification. It is Option B with the
  behavioural story removed.

---

## What I would want you to decide

**A is close to free and I would take it regardless of what else you choose** — it removes
a defect (informed traders crossing spreads wider than their edge) using only quantities
already present.

But A alone leaves the unbounded term intact, because that term is driven by *noise* flow.
For that you need B, C, or D, and all three require at least one behavioural assumption.
**B is the cheapest defensible one**; C is the most realistic and the right foundation if
you also intend to rebuild Glosten-Milgrom; D I would not recommend — it has the same cost
as B with a weaker story.

I have not picked one, and I have not implemented any of them.

---

## Caveat that applies to every number above and in this project

**The reported "Sharpe" is not a Sharpe ratio.** There is no capital base anywhere in this
repo — original audit finding 8, confirmed by grep across `src/`, `configs/` and `bench/`.
`compute_sharpe` (`src/backtest/report.py:8-13`) computes

```
sqrt(252) × mean(daily DOLLAR P&L) / population-std(daily DOLLAR P&L)
```

That is a **signal-to-noise ratio of a dollar P&L stream**, invariant to leverage and book
size, not a risk-adjusted return. It also subtracts a daily rate from a dollar series (an
effect of 1.3e-07, dimensionally incoherent) and uses `ddof=0`, which inflates it. Every
Sharpe figure quoted in this project carries that caveat, including all of the ones in the
remediation commits.
