# Adversarial Evaluation Contract

This document records the builder-critic hardening pass performed against the
artifact false-positive claim. It is an engineering audit, not a clinical or
population validation result.

## Frozen Question

Can the previously documented tuned waveform-length threshold gates reduce
false activations without losing delivered intended clicks under an evaluator
that does not let a held-out session select its own gates?

The two predeclared candidates are returned by `frozen_decision_grid()`:

| Candidate | enter | exit | enter dwell | release dwell | min event | refractory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Runtime default | 0.6 | 0.4 | 2 | 2 | 50 ms | 80 ms |
| Previously documented tuned gates | 0.8 | 0.4 | 2 | 2 | 200 ms | 80 ms |

No new model family or broad gate grid is admitted to this acceptance run.

## Acceptance Contract

The candidate must satisfy all of the following relative to the frozen
runtime-default baseline:

- reduce the pooled unmatched-down rate by at least 30%;
- retain every intended press delivered by the baseline;
- add no more than one false down in any outer session; and
- worsen median onset latency by no more than 25 ms.

Gate selection is nested. For each outer held-out session, both candidates are
compared using inner leave-one-session-out folds made only from the other
sessions. An inner candidate is eligible only with 100% delivered detection,
median onset latency at most 200 ms, absolute p95 onset error at most 400 ms,
and observed rest and artifact exposure. The outer session is then replayed
once with the frozen inner choice. If no inner candidate is eligible, the
selection is explicitly a fallback and the acceptance gate fails.

The runtime-default profile is also scored on each outer session as the frozen
comparison baseline. `assess_candidate_acceptance()` applies all four criteria
to those outer-fold scores, including identity-level retention of each
baseline-delivered intended click. The CLI succeeds only when every inner gate
and every baseline-relative criterion passes; otherwise it exits with status 3.
Only the exact two-row frozen grid is eligible for that passing status. A full
or custom grid remains exploratory even when run through nested folds.

## Corrected Metric Semantics

- A reference press is matched at click onset with a frozen `[-200, +200] ms`
  window. Matching maximizes delivered matches, then total intended
  activations, then minimizes absolute onset error.
- Every unmatched emitted `down` is a false down, including a later-cancelled
  activation. Completed false clicks are a reported subset, not an additional
  count.
- Rest, artifact, and other false downs partition the unmatched downs. `other`
  is included in totals and ranking.
- Cancelled and unterminated intended activations do not count as delivered.
- Adjacent phase windows are half-open, so boundary events belong to the phase
  that starts at that timestamp. Overlapping or unordered windows are errors.
- Every phase, click, and switch-event timestamp must be a finite integer inside
  the first-to-last usable raw-sample span. An annotation outside that observed
  coverage is rejected instead of inflating the exposure denominator.
- A raw-sample gap over 100 ms invalidates continuous exposure for the session;
  the evaluator rejects it rather than counting unsampled time as quiet time.
- A rate with zero exposure is unavailable, never zero.
- Counts are pooled over exposure; per-session results remain visible.

## Adversarial Findings

The earlier evaluator could not support its stated false-`down` claim:

- the legacy matching window ran from 400 ms before click onset through 600 ms
  after the *end* of the interval, allowing a one-second prompt to match a down
  as late as 1.6 seconds after onset;
- cancelled activations emitted a real `down` but disappeared from false-down
  counts;
- `false_downs_other` disappeared from aggregate results and gate ranking; and
- runtime gates were selected after inspecting aggregate results from the same
  outer folds used for the reported number;
- annotation windows could extend beyond usable raw coverage and enlarge the
  exposure denominator; and
- the four frozen baseline-relative criteria were documented but not all
  enforced by the CLI's success status.

Threshold training also rescored every frame at every candidate threshold,
making it quadratic. Five concurrent evaluations pegged five CPU cores. The
replacement uses sorted prefix counts and bisect lookup in `O(n log n)`, with
differential tests for duplicates, ties, endpoint rounding, overflow, and
invalid inputs.

## Current Result

Commanded over all five main prompted sessions (`prompt-gui-001` through
`prompt-gui-005`, including the formerly excluded 004), all recorded for one
wearer in one sitting:

Directories already named `*-bad-*` were treated as acquisition-invalid by the
preexisting name rule and excluded before detector output was inspected. No
main-session result was removed after scoring.

```bash
reson-eval --sessions sessions --include-glob 'prompt-gui-*' \
  --sweep threshold:wl --decision-grid frozen --sweep-top 2

reson-eval --sessions sessions --include-glob 'prompt-gui-*' \
  --nested-sweep threshold:wl --per-session
```

Fixed-candidate descriptive results under the corrected scorer:

| Candidate | Delivered | False downs total | Rest/min | Artifact/min | Other | Median onset | Cancelled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Runtime default | 53/100 | 156 | 9.10 | 18.00 | 62 | 136 ms | 1 |
| Previously tuned | 46/100 | 79 | 0.91 | 6.00 | 64 | 142 ms | 14 |

Known negative exposure was 541.4 seconds: 461.404 seconds of rest and 79.996
seconds of artifact phases. The tuned gates reduce the pooled false-down count
by about 49%, but lose seven already-delivered intended presses. They therefore
fail the frozen detection-retention criterion and are rejected.

In the nested run, neither candidate reached the 100% inner delivery gate in
any of the five outer folds. Every outer selection was labeled a fallback; no
artifact false-positive improvement was accepted. `reson-eval` exits with
status 3 for this condition so an automation cannot mistake the fallback
report for a passing acceptance run. The machine-readable acceptance report
also records each criterion, reduction fraction, lost click identities,
per-session false-down deltas, and latency regression.

## Maximum Honest Claim

The strongest current statement is:

> The adversarial pass found that the prior 100%-detection claim was an
> evaluator artifact. Under corrected onset matching and complete down-event
> accounting, neither predeclared gate configuration meets the engineering
> acceptance contract on five same-wearer, same-day recordings.

These recordings and evaluator decisions have already been inspected, so even
nested recomputation is retrospective and exploratory. No cross-day,
cross-placement, cross-user, prospective, fatigue, posture, or production
reliability claim is justified. The next valid promotion test requires newly
recorded, sealed cross-day sessions.
