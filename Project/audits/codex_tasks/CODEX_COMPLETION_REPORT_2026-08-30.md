# Codex completion report — Tasks 01 through 05

**Implemented and reviewed by Codex on 30 August 2026.**

This is the detailed maintainer handoff for the five briefs originally added in
commit `a0ef65d`.  The work is tooling, dashboard privacy, and source analysis;
it does not optimize a kernel and it does not claim new GPU evidence.  Codex did
not modify the official scripts, README, shape/manifest inputs, frozen harness,
runner-owned results, loop/gate machinery, generated submission, or kernels.

The repository owner's configured identity remains the Git author and
committer, matching the existing Claude convention.  Tasks 01 and 02 identify
Codex with an `Implemented-by: Codex` trailer.  Tasks 03 through 05 additionally
carry `Co-Authored-By: OpenAI Codex <noreply@openai.com>`; the earlier commits
were deliberately not rewritten.

## Commit and file ledger

| Task | Commit | Files in the Codex commit |
|---|---|---|
| 01 — streamed Shape 14 evaluator | `fcef49edc58a04de309eb56384481bba5ebea7fd` | `Project/tools/shape14_eval.py` |
| 02 — official predicate in side tools | `4556a044c780816c7e58a32c78fcbaa643e6952f` | `Project/tools/shape6_local_eval.py`; three smoke tools |
| 03 — freeze-bound ship manifest | `1e87013f3ade409e407660301e002d3f69192e64` | `Project/tools/ship_manifest.py` |
| 04 — dashboard presentation mode | `58ff65f9892f4ca7f1be84e496fa568916031921` | `Project/tools/dashboard.py` |
| 05 — Shape 14 memory inversion analysis | `d36be9d760c9b8d06d2cd71c91cf170b11cdc8d5` | `Project/audits/external/memory_inversion_b1_gt_b2.md` |

Fable made separate commits while this work was in progress.  The Codex commits
remain isolated to the file sets above.  The later
`Project/audits/codex_tasks/06_ship_verdict_filter.md` brief was not one of the
five tasks authorized here and Codex did not implement it.

## Task 01 — streamed official-shape evaluation

Brief: `Project/audits/codex_tasks/01_shape14_eval_streaming_fix.md`

The old Shape 14 evaluator accepted arbitrary small batch sizes and directly
loaded standalone `k014`, so a B1/B2 proof did not bind the shipped generated
submission or represent the official B32 workload.  Task 01 changes `eval` to a
fixed official configuration of B32, S100000, d_model 1024, 16 heads, two
layers, causal, while keeping GPU memory bounded by processing 32 independently
staged B1 slices in serial.

Important implementation details:

- The candidate is imported from
  `Project/submission/torch_transformer_benchmark_submission.py`, instantiated
  as `UserOptimizedTransformer`, loaded from one deterministic baseline state,
  and bound to the generated submission SHA.
- Correctness defaults to five base seeds.  Slice `i` uses deterministic seed
  `base_seed * 1000 + i`, and the packet records the formula and every base seed.
- Pass/fail exactly mirrors the official rule: both tensors must be finite and
  each element must satisfy absolute error `<= 0.002` **or** relative error
  `<= 0.02 * abs(reference)`.  It reports violations, non-finites, worst base
  seed/slice, maximum error, and a float64 element-weighted mean.
- Timing defaults to three repeats.  Every repeat times 32 individual B1 CUDA
  calls, sums those slice times, and contributes one sum to the reported median.
  It also records the staging-inclusive wall time and per-repeat allocated and
  reserved peaks.  The packet says plainly that this is 32 serial B1 calls, not
  one literal B32 call.
- Successful `eval` output uses schema `streamed-v1` and filename
  `shape14_streamed_<UTC>.json`.  It carries evaluator, submission, candidate,
  official, environment, configuration, seeding, predicate, timing, and memory
  provenance.
- `decomp-check` compares a reduced B8/S2048 full-batch call against separate
  B1 calls using distinct candidate instances.  It is a decomposition sanity
  check; because S2048 is below the long-sequence threshold, it is not by itself
  proof of historical k014 long-route memory behavior.
- The existing streamed-oracle implementation and its tight feasible-size
  dense-validation gate were intentionally preserved.

No packet was generated and no CUDA code was run.  The maintainer still needs
to run, in order, `validate`, `decomp-check`, and then the default full `eval` on
the intended GPU.  `decomp-check` currently always exits zero, so inspect its
printed `max_abs_difference` manually: it should be exact or approximately
`1e-6` and must not approach `2e-3`.  Do not describe the resulting streamed
timing as a literal B32 call.

If this tool is modified later, preserve the generated-submission binding,
official predicate, deterministic slice seeds, per-repeat sum/median semantics,
and packet hashes.  `ship_manifest.py` recognizes a full preferred packet by
both `schema_version == "streamed-v1"` and `shape.batch_size == 32`.

## Task 02 — official correctness semantics in side tools

Brief: `Project/audits/codex_tasks/02_shape6_eval_predicate_fix.md`

The following four tools now use the official finite absolute-OR-relative
predicate rather than `torch.isclose`'s additive absolute-plus-relative
tolerance behavior:

- `Project/tools/shape6_local_eval.py`
- `Project/tools/smokes/shape6_core_smoke.py`
- `Project/tools/smokes/padded_mask_smoke.py`
- `Project/tools/smokes/shape14_core_smoke.py`

Shape 6 additionally accepts a comma-separated `--seeds` list, retains bounded
per-seed tensor lifetimes, reports per-seed and aggregate violations/non-finites,
records candidate/evaluator provenance, and includes both allocated and reserved
peaks for the ten-repeat flat-memory check.  If non-finite values occur,
`max_abs_err` is `null` with an explicit status while the maximum over finite
elements is retained separately; a non-finite is always a violation.

Two boundaries are intentional and important:

1. The evaluator still directly executes `Project/kernels/k015_shape6.py`; it
   does not bind the generated submission.  Changing the candidate source was
   outside this brief's requested scope, and this is not final shipping evidence
   for dispatcher integration.
2. Its CLI default is the historical single seed `1234`.  The open binding work
   still calls for generated-submission evidence with at least five seeds.

No GPU run or new Shape 6 packet was produced.  A later generated-submission
evaluator should reuse `official_error_stats` and the bounded seed loop rather
than weakening either rule.

## Task 03 — manifest provenance and Shape 14 packet selection

Brief: `Project/audits/codex_tasks/03_ship_manifest_bugfixes.md`

`Project/tools/ship_manifest.py` now enforces the documented freeze point:

- It refuses generation when any tracked or untracked porcelain-visible change
  exists.
- It records the current full HEAD revision, reads the generated submission blob
  from that exact commit, hashes both committed bytes and on-disk bytes, and
  refuses if they differ.
- It understands legacy `median_ms` packets and new
  `median_of_sums_ms`/`streamed-v1` packets.
- For Shape 14 it prefers correct B32 `streamed-v1` evidence.  Within the chosen
  class it selects by packet recency, never by comparing latency across
  different batch/protocol classes.  If no preferred full packet exists, it
  selects the newest other correct Shape 14 packet.
- Every other `shape14_*.json` packet is retained under `alternates`, including
  invalid and legacy packets, so provenance is not silently discarded.
- Shape 6 and Shape 14 side evidence receive distinct, truthful audit labels.

`Project/results_side/SHIP_MANIFEST.json` was deliberately not regenerated.
Generate it only after final code and evidence packets are committed.  The
manifest will record the pre-generation HEAD as the code/submission freeze
revision; committing the generated JSON afterward is expected, but subsequent
code or submission changes require another generation cycle.

Post-implementation review found an out-of-brief edge that remains open:
`packet_is_correct(6)` currently checks zero correctness violations but does not
also require `memory.flat == true`, even though the Shape 6 evaluator exits
successfully only when both conditions hold.  The dashboard's Shape 6 side-row
filter has the same pre-existing omission.  A numerically passing packet with a
failed flat-memory gate could therefore be treated as side evidence.  This was
not silently changed after the isolated Task 03 commit; the maintainer should
fix or explicitly adjudicate it before freeze.

The later Task 06 brief addresses a separate manifest risk: journal candidates
with `RULE_VIOLATION`/`RETEST` verdicts must be excluded unless validly cleared,
and shapes without eligible evidence must fail closed.  Task 06 is still outside
this report's implementation scope.

## Task 04 — screen-recordable dashboard mode

Brief: `Project/audits/codex_tasks/04_dashboard_presentation_mode.md`

The dashboard sidebar now contains
`Presentation mode (hide private logs)`, default OFF, plus a caption stating
that the toggle is session-only and writes no configuration.

When ON:

- Raw log expander bodies and code blocks are replaced by
  `(hidden in presentation mode)`; their event titles, normal timestamps,
  friendly names, and verdict labels remain visible.
- CAGE plan predictions and changed-file detail excerpts are suppressed.
- Dynamic scoreboard, idea, event-message, timestamp, audit-label, and remark
  text passes through a path redactor.  It covers POSIX absolute paths, Windows
  drive/UNC paths, and `file:///` paths without treating normal HTTP(S) URLs as
  filesystem paths.
- Reviewer prompt/response bodies are not rendered.

When OFF, the pre-task rendering behavior remains intact apart from the required
toggle and caption.  The forced dark theme, ten-second fragment refresh, metric
cards, responsive no-clipping scoreboard, and Right-now strip were not changed.
The responsive leaderboard fix was already present and was deliberately
preserved.

Streamlit was not launched, per the brief.  Before recording, visually inspect
both modes with representative current logs.  Presentation mode is a rendering
safeguard, not an adversarial sandbox: any future dynamic render surface must
also use `presentation_text`, and any future raw body must honor the mode before
calling `st.code`, `st.markdown`, or equivalent.

## Task 05 — B1-greater-than-B2 Shape 14 memory analysis

Brief: `Project/audits/codex_tasks/05_memory_inversion_analysis.md`

The one permitted output is
`Project/audits/external/memory_inversion_b1_gt_b2.md`.  It was independently
checked against both packets, historical commits, the old/new evaluators, k014,
and the generated dispatcher.

Its main finding is stronger than an allocator guess: the two historical
packets executed different candidate bytes.  B1 SHA `ee85ee29...` maps to k014
v1 at `8db6826`; B2 SHA `68d62045...` maps to the memory-disciplined v2 at
`851845c`.  V2 explicitly frees full-sequence staging and Q/K/V tensors and
chunks the projection/residual/FFN tail.  The historical evaluator also used one
peak window spanning candidate correctness, streamed-oracle comparison, and
timing, so the old values are mixed-phase process maxima rather than controlled
candidate-only B-scaling evidence.

For the current generated-submission path, the analysis derives a source-visible
2.760116577 GiB footprint before library workspace and predicts the new full
32-slice protocol should center near 2.85 GiB allocated, normally 2.8–3.0 GiB
per repeat, with 3.2 GiB as a conservative source-based ceiling.  This is a
falsifiable forecast, not measured evidence.  The exact packet hashes and
phase-separated peaks must diagnose any miss; memory magnitude alone cannot
prove that an old candidate or retained oracle was used.

## Verification performed

The following non-GPU checks passed after all five commits:

```text
PYTHONPYCACHEPREFIX=/tmp/track3_codex_final_pycache python3 -m py_compile \
  Project/tools/shape14_eval.py \
  Project/tools/shape6_local_eval.py \
  Project/tools/smokes/shape6_core_smoke.py \
  Project/tools/smokes/padded_mask_smoke.py \
  Project/tools/smokes/shape14_core_smoke.py \
  Project/tools/ship_manifest.py \
  Project/tools/dashboard.py
```

- `git show --check` passed for all five Codex commits.
- Per-commit name inspection confirmed Task 01 changed one file, Task 02 changed
  exactly four files, Task 03 one file, Task 04 one file, and Task 05 exactly one
  newly created file.
- `rg -n "torch\\.isclose" Project/tools` reports only
  `Project/tools/shape14_eval.py:193`, the intentionally unchanged tight
  oracle-versus-pinned-dense validation check.
- Task 04 received a separate redaction/render-surface review after its final
  path-handling changes; no remaining brief defect was found.
- Task 05 received a separate provenance, citation, arithmetic, and prediction
  review; one initially over-broad decomposition-test statement was corrected,
  and the final document passed.

No GPU kernels, benchmark, frozen runner, Streamlit server, or manifest
generation were executed by Codex.  Therefore syntax and source-level behavior
are verified here, while runtime CUDA evidence remains explicitly pending.

## Maintainer sequence from here

1. Complete and commit any outstanding Fable/source work, including deciding
   the Shape 6 `memory.flat` filter and the separate Task 06 verdict gate.
2. Rebuild/finalize the generated submission if source components changed, then
   verify its recorded SHA is the one intended for evidence.
3. On the intended GPU, run Shape 14 oracle `validate`, reduced `decomp-check`,
   then the default full streamed `eval`.  Inspect correctness, all three summed
   timing repeats, and the per-repeat allocated/reserved peaks against the Task
   05 prediction.  Manually judge `decomp-check`'s printed difference; its
   current process exit code does not enforce the stated threshold.
4. Produce final generated-submission-bound Shape 6 evidence across at least
   five seeds.  The standalone k015 evaluator can support kernel diagnosis but
   cannot prove dispatcher integration by itself.
5. Inspect and commit final immutable side packets.  Do not overwrite old
   packets; the manifest deliberately preserves Shape 14 alternates.
6. With all code and packets committed and no pending worktree entries, run the
   ship-manifest generator, inspect its report/contents, and commit the generated
   manifest.  Any later code/submission change invalidates that freeze cycle.
7. Launch the dashboard only for the maintainer's visual check.  Exercise both
   presentation-mode states and use ON for screen recording.

This file is the modification map for later agents.  When behavior changes,
update the relevant task section with the new commit, packet schema, and
validation evidence rather than treating these source-only checks as GPU proof.
