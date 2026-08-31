# STATE — first ten seconds of a session

This file is deliberately short and holds no plan. Two files do that:

- **`Project/GRIND_ENTRYPOINT.md`** — the operating manual: the commands, the one
  next permitted action, the lanes, the stop conditions. Get it by running
  `python3 Project/tools/session_bootstrap.py`.
- **`Project/HANDOVER.md`** — state, open defects and the FIX → LOCK → GRIND plan.

Then read all of `Project/memory/LESSONS.md`, every session, and
`Project/research/INDEX.md` before relying on any research note.

**If a command and a document disagree, the command is right** — including this one.

Updated: 2026-09-01 ~03:00 SGT. Branch `grind-lastday`.

---

## ✅ NEW — THE BOARD IS DONE. All 14 shapes on one artifact. Read this before §0a.

1 Sep ~03:00 SGT. **`c2028c4823ff756b062940e4eff35d5a6a341e9538b755256509cf3432e7794b`**
is the shipping artifact and **every one of the 14 rows is measured on it**. This
is the first board in the project where that is true, and it is what
`Project/BOARD.md` now holds, with a packet sha256 and entry id per row.

| | |
|---|---|
| geomean, 12 shapes with a baseline | **11.87×** |
| mean MFU, all 14 weighted equally | **42.7%** |
| best speedup | 31.513× (shape 13) |
| worst speedup | 2.366× (shape 8) |
| best MFU | **88.7%** (shape 14, 48.271 s) |
| correctness | 94 trials, 17,370,759,168 elements, **0 violations** |
| strikes / promoted | 0 / 0 (screening lane, by design) |

Per-shape: 1 **8.998** · 2 **26.691** · 3 **19.776** · 4 **10.643** · 5 **9.823** ·
6 *no baseline, 60.3873 ms, 59.8% MFU* · 7 **29.149** · 8 **2.366** · 9 **4.933** ·
10 **6.846** · 11 **18.377** · 12 **11.642** · 13 **31.513** ·
14 *no baseline, 48.271 s, 88.7% MFU*.

**Everything below this section that quotes 9.45×, 10.14× or 10.6858× is
superseded.** Those boards are kept because their *corrections* are the record of
how this project learned to measure. Do not quote their numbers.

`Project/MEASUREMENT_METHODOLOGY.md` and all four drafts now carry this board and
agree with each other. `Project/loop/cards.jsonl` gained card **C33** for the
newly registered `F-shape12-fusion` family.

## 🔴 THE AUDITOR DIAGNOSIS IN EVERY DOCUMENT WAS WRONG. Real cause found 1 Sep.

**"The audit recorder is broken" is false.** The recorder is fine. **The auditor
never starts.**

`Project/audits/verdict_schema.json:70` uses an `allOf`/`if`/`then`/`else`
conditional. OpenAI's structured-output mode does not permit `allOf`, so Codex
hands the schema over as `response_format`, the API returns
`400 invalid_json_schema`, and the process exits 1 with empty stdout before the
model reads the packet.

Timeline, from the commits and the response artifacts:

| when | what |
|---|---|
| 30 Aug 15:46 | `ed053f2` adds the `allOf`. **The break.** |
| 30 Aug 20:35 | `231e786` switches default backend to Claude, out of Codex quota. Claude has no `--output-schema`, so the bad schema becomes inert prompt text. **The mask.** |
| 30 Aug 22:37 | Audit runs on Claude, returns a full verdict document. |
| 1 Sep 02:44 | Audit runs on Codex. Three 400s in one minute, recorded in the hash chain, escalated to `owner_attention`. |

**Owner fix, one edit, inside the LOCK:** replace that conditional with something
in OpenAI's supported subset, or drop it and let `validate_verdict_document`
enforce it locally, which is already what happens on the Claude path.

Everything around it worked: enqueued, launched, retried, recorded, escalated.
Full write-up in LESSONS 61.

## 🟡 SHIP_MANIFEST.json cannot be generated, and that is downstream of the above

`python3 Project/tools/ship_manifest.py --diagnose` (1 Sep 03:19) returns
**"SHIP MANIFEST REFUSED: No official shape has post-lock bound evidence."**

Two causes, both the design working:

1. Every shape lists `missing_audit_verdict` as a blocking reason. No ship
   manifest can exist while the auditor cannot start, which is the intended
   coupling. Fix the schema and this unblocks.
2. The manifest reads the pre-LOCK journal and `Project/results_side/`. Tonight's
   board is screening-lane, so it lives in `Project/authority/` and the scratch
   namespace by design, and the manifest does not look there. It consequently
   reports only legacy rows measured against kernel files rather than against the
   submission.

Not a blocker for the report or the board — `Project/BOARD.md` §2 indexes every
row's packet, which carries the same environment and hashes. Recorded so nobody
spends the packaging window trying to force the manifest.

---

## 🟢 0a. ATTENTION KERNEL OPENED FOR THE FIRST TIME. Shipping artifact is now `7609fa17…`.

31 Aug ~16:20 SGT. `_sub_attn_heads` holds 12–49% of device time across the fused
shapes and had never had a dedicated architectural search — `_sub_norm_qkv` and
`_sub_attn_block_tail` had both had beams. See LESSONS 57 for how that happened.
Three changes, all standard FlashAttention-2, none previously present:

1. **Base-2 softmax.** `log2(e)` folded into the score scale on the host, so the
   softmax calls `exp2` directly instead of `exp` (which is `exp2` plus a
   multiply on NVIDIA hardware). One fewer multiply per score element per
   iteration, at zero cost — the scale multiply was already there. **Ungated,
   applies to every shape.**
2. **Causal loop split**, gated at `seq_len >= 256`. Key blocks strictly below
   the tile's first row are visible to every row in it, so the mask and the
   bounds check are provable there and come off the loop entirely.
3. **NOT done, deliberately:** folding the scale into `q` before the dot would
   remove the score-tile multiply outright, but `q` is fp16 and that rounds the
   scale into 11 bits. Measured error is already 1.16e-3 of a 2e-3 budget.

**Measured, all `correct: true`, seven seeds, zero failed elements:**

| shape | seq | before | **on the current build** | note |
| --- | --- | --- | --- | --- |
| 13 | 1024 | 31.9119x | **33.6523x** | **+5.5%**; kernel 632.7 → 582.6 us |
| 1 | 128 | 8.5217x | 8.7858x | measured on the ungated build |
| 12 | 32 | 11.3504x | **11.2516x AND 9.7638x** | see below — both, same bytes |

## 🔴 0a-bis. SHAPE 12 REPLICATES 13% APART ON BYTE-IDENTICAL CODE. READ THIS FIRST.

Two runs of the **same artifact**, minutes apart, nothing changed between them:
**11.2516x then 9.7638x, a 13.2% spread.** The campaign's calibrated noise floor
for shapes of this class reads about **0.15%**, so it is wrong by roughly two
orders of magnitude — it is computed by timing the baseline against itself
*inside one process*, which measures second-to-second steadiness, not
run-to-run reproducibility. LESSONS 59.

**Withdrawn as a result — every shape-12 claim made on 31 Aug from single
samples:** that the ungated loop split cost 5.5%, that gating it recovered the
row, and that the row is flat against its history. None of those deltas is
larger than the replicate spread, so none of them was ever resolvable. The gate
on `SPLIT` stays, but on the dead-code argument alone: on seq_len 32 the loop
provably never executes, so not emitting it cannot hurt.

**Why the shape 13 result survives this and shape 12's did not:** the shape 13
comparison is a paired diagnostic whose two UNTOUCHED kernels agreed to within
0.2% between the two runs, which is the evidence that conditions were equal and
the changed kernel really moved. The shape 12 comparison had all four kernels
moving 14–17% together — the signature of a condition change, not a code change.
**Check that the unchanged parts agree before believing any pair.**

**Nine rows outstanding on `7609fa17…`:** 2, 3, 4, 5, 7, 8, 9, 10, 11, plus a
re-run of 1. Each needs a diagnostic (for counter-evidence bound to these bytes)
then a `delta` screening run — `delta` needs no research cycle, which is the
fast path.

**Correction to a claim made earlier in this session:** I reported `reconcile` as
jammed by an interrupted shape-13 diagnostic and asked for an owner quarantine.
That was a misread — the line was `PENDING`, which is informational, and the
request settled on its own. One `quarantine_request` slip use was spent on an
authority receipt that turned out unnecessary; the quarantine itself refused
with "already settled", so nothing was suppressed. Blast radius of that class of
PENDING, measured: nothing is blocked by it.

## 🟡 0b. THE 10.14x AND 10.69x BOARDS BELOW ARE NOT SINGLE-BUILD NUMBERS.

Correction recorded 31 Aug ~15:30 SGT, from the `Project/memory/HOTSPOT_COVERAGE.md`
audit. Two separate problems with the numbers in section 0c:

- **The 10.14x table is on artifact `54057a33…`**, which stopped being the
  shipping artifact hours ago. At least seven builds have followed it.
- **The later 10.6858x board was never on one build.** Its rows come from four
  different artifacts: shapes 4, 5, 9, 10, 12, 13 on `2778b747…`; shapes 1 and 11
  on `418952bf…`; shapes 2 and 7 on `599f5dad…`; shape 3 on `301d7063…`.

Nothing here was averaged across builds dishonestly — each row is a real measurement —
but a geomean whose rows come from four artifacts is **not** "the speedup of the file
that ships", and it must not be quoted as one. **Until the nine outstanding rows are
measured on `7609fa17…`, there is no defensible board geomean. Quote per-shape
measured rows with their artifact hash, or quote nothing.**

## 🟢 0c. THE HEADLINE MOVED: 9.45x → **10.14x**, measured on the file that shipped THEN.

31 Aug ~10:30 SGT. A new kernel design is **integrated into
`Project/submission/dispatcher_region.py`**, the submission is rebuilt (`verified: true`,
byte-identical outside the replacement region), and **all twelve shapes have been
re-measured on the rebuilt artifact itself** — sha `54057a33…`, one screening-lane permit
per row, verified-quiet box, `correct: true` on every row.

| shape | old file `4da76db6…` | **new file `54057a33…`** | delta | note |
| --- | --- | --- | --- | --- |
| 13 | 28.2849x | **30.8989x** | +9.2% | |
| 7 | 20.9595x | **23.8603x** | +13.8% | head_dim 8 |
| 11 | 12.5909x | **17.6588x** | **+40.3%** | head_dim 8, 16 heads |
| 2 | 13.1434x | **15.5315x** | +18.2% | |
| 3 | 12.9618x | **12.1500x** | **−6.3%** | only loss on the board |
| 12 | 10.4348x | **10.5427x** | +1.0% | |
| 4 | 8.9211x | **10.0690x** | +12.9% | |
| 5 | 9.1150x | **9.3050x** | +2.1% | |
| 1 | 8.1673x | **8.3830x** | +2.6% | |
| 10 | 6.5352x | **6.2975x** | −3.6% | |
| 9 | 4.3503x | **4.5459x** | +4.5% | |
| 8 | 2.0160x | **2.0216x** | +0.3% | fp16 branch, **untouched by this change** |
| **geomean** | **9.4470x** | **10.1370x** | **+7.31%** | |

**CITATION CORRECTED 31 Aug ~20:15 — this line used to send you to the wrong file.**
It pointed at `Project/loop/geomean_camp_final.py` for the arithmetic above. That
script holds the LATER, four-artifact board (`2778b747`, `599f5dad`, `301d7063`,
`418952bf`), whose shape-11 row is 17.42 and whose geomean is 10.6858x. Running it
does not reproduce the 10.1370x table above. §0b already said that board was never
single-build; this citation was left behind. The table's twelve rows trace
individually through `Project/loop/gate_log.jsonl` and the packets under
`Project/authority/`, each carrying its own `target_sha256`.

**Quote 10.14x.** Ten of twelve shapes improved, one is flat, one lost.

### What the change is

The fused-block megakernel was two Triton kernels; it is now **three**. The per-head
attention loop is lifted out of the block-tail kernel into its own grid dimension
(`_sub_attn_heads`, grid `(q_tiles, B, H)`), writing each head's output into its own column
slice of a shared context buffer. The tail then does **one full-width output-projection
dot** over the assembled context instead of H dots padded up to Triton's 16-wide `tl.dot`
minimum. Math is unchanged: same online-softmax flash recipe, same fp32 residual stream and
LayerNorm statistics, same exact-erf GELU.

### Where the gain actually comes from — and the mechanism I got wrong

I wrote **two** mechanisms into the design and gave them equal weight. Only one survived.

| group | geomean gain |
| --- | --- |
| the two head_dim-8 shapes (7, 11) | **+26.4%** |
| the other nine fused shapes | **+4.3%** |
| the untouched fp16 shape (8) | +0.3% |

The head-count sweep settles it. Shapes 9, 10, 1 and 11 are the **same 7.52 GFLOP problem**
at 1, 2, 4 and 16 heads, and they returned **+4.5%, −3.6%, +2.6%, +40.3%**. Three of the
four sit within four points of zero; the fourth is the only one with head_dim 8. So the
gain is a **head-width** effect, not a head-count effect: the H-way grid split — the
mechanism the kernel is *named* after — is doing approximately nothing, because every shape
except shape 2 already had a grid past 38 SMs. **The full-width out-projection is doing the
work.** That correction has to travel into the drafts; it is not a detail.

### What the per-shape deltas can and cannot carry

**They cannot carry a claim.** The batch sweep (shapes 2, 3, 4, 1, 5 — identical geometry
at batch 1, 4, 16, 64, 128) returned **+18.2%, −6.3%, +12.9%, +2.6%, +2.1%**: not monotone,
not flat, spanning 24 points. No mechanism produces that shape; **cross-invocation scatter**
does, and this says it is materially larger on small shapes than the ~9% LESSONS 11 quotes.
A preregistered falsifier on shape 4 fired and this is the honest consequence, taken rather
than explained away.

What *does* carry weight: the **geomean over twelve shapes**, which averages that scatter
down; the **+26.4% vs +4.3%** split, which is a between-groups comparison measured in one
session; and **shape 11's +35.5%**, the one figure with a same-session paired control
(measured against a k009 control run minutes apart, not against a stored number).

### Preregistration record for this campaign

Seven falsifiers preregistered, **two fired** (shape 3's, shape 4's) and both are reported
above rather than argued away. Numeric bands went **3 for 14** here against **0 for 26**
before — and all three hits were shapes where I predicted *no change*. The single
gain-predicting band that hit (shape 11) was the only one ever derived from a measurement
that had its own same-session control. That is a statement about controls, not forecasting.

### Still true, and still owner-only

Screening lane, so **nothing here is promotable** and no verdict binds to any row — audit
recording (§1) is still broken. Excludes shapes 6 and 14 (not locally runnable).

## ⓪ WHERE THIS STOPPED, AND WHY — read this before the action list

**A second measurement campaign ran on 31 Aug morning and it changed the headline.** The
"deliberately did not do" list below was written at ~02:10 when the owner was asleep and the
gate was jammed. Both conditions cleared; the first item on it was attempted after all, and
it worked — see §0. The list is kept because its *reasoning* is still the record of what was
weighed and when.

**Superseded by §0:** "the shape-11 head-dim-8 de-padding kernel" was ranked bad odds on the
strength of a `tl.sum`-reduce technique the research note disliked. That judgement was
correct about *that* technique and wrong about the target: a different route to the same
padding problem — splitting the head loop so the out-projection runs at full width — was
built, integrated and measured, and it is where **+26.4%** of the gain sits. The predicted
ceiling of "+4.2% geomean if the gap closes completely" was beaten: the realised figure is
**+7.31%**. I under-estimated the target because I only costed one way of attacking it.

**Still not done, and still for the reasons given:**

- **Sequence-persistent CTAs for the launch-bound family (2, 3, 7, 12).** Untouched. New
  Triton work against a 2e-3 tolerance with hours left; the same promotion block applies.
- **Regenerating `SENSITIVITY.md` against post-LOCK medians.** Needs `sensitivity_board.py`
  to read authority packets instead of `JOURNAL.jsonl`, and `Project/tools/` is write-denied
  to me. Owner-only; low value since the draft now cites that board for its scoring logic,
  not its numbers.

**Current state:** lock valid (29 protected files), `verify-lock` green, no watcher active,
`reconcile` clean, **0 strikes**, nothing promoted, headline **10.14x** measured on the
shipped artifact `54057a33…`.

## ⓞ OWNER ACTIONS — the things only you can do. Everything else is done.

Each is blocked because the agent is denied write access to the files involved, which is
the design working, not a defect. None of them blocks reading the results.

1. **⭐ HIGHEST VALUE: fix the two extreme-shape evaluators (one line each).** They compare
   a mask's device against `torch.device("cuda")` while the official generator returns it on
   `cuda:0`, and those are unequal in PyTorch. Files: `Project/tools/shape14_eval.py:274`
   and `Project/tools/shape6_local_eval.py:146`. Compare `mask.device.type != device.type`,
   or normalise first with `device = torch.zeros(0, device=device).device`. Full diagnosis
   in DECISIONS, 31 Aug ~02:20.

   **This one line gates THREE deliverables, which is why it moved to the top** (found
   31 Aug ~07:15 by reading the tool instead of the note):
   - the shape 6 and shape 14 evidence packets cannot be regenerated (they stay
     PROVISIONAL: one seed, pre-integration sha);
   - **shape 14's full-batch timing — a judge-facing `[PENDING]` in both drafts — cannot be
     produced at all.** `shape14_eval.py eval` requires `--validation-packet` ("a *passed*
     shape14-oracle-validation-v2 packet"), and the `validate` subcommand that mints it is
     labelled `(gate)` in its own help and dies on this bug. The drafts previously called
     that run "queued"; it is blocked, and both now say so.
   - so the only path to filling that `[PENDING]` before freeze runs through this fix.
2. **Fix audit recording.** `audit_champion.py`, inside the LOCK. Three attempts, three
   distinct failures, retry cap exhausted; `run-be8e56a55edd1926a84bf5d1efc0b154` is stuck
   in `owner_attention`. Until this is fixed **nothing can be promoted to champion and no
   verdict binds to any measured row** — the board below is real but unadjudicated.
3. **Dry-run the two smoke scripts before recording the video.** They are a different pair
   of files from the broken evaluators and were not testable from the agent's command
   allowlist, so their status is genuinely unknown rather than assumed good. See the
   caution block in the video script's Scene 5.
4. **Mint a fresh capability before you film.** The overnight ones expire around
   **21:00 on 31 Aug**, inside the recording window, and Scene 3 now shows a real permitted
   run. Without a live capability you cannot issue a permit and that scene cannot be shot.
5. **Nothing to do — this one resolved in your favour.** A 31 Aug probe granted a
   promotion-capable permit despite 28 `RULE_VIOLATION` lines, and I briefly wrote into the
   report that the brake had failed. **That was my error and it is corrected.**
   `verdicts.jsonl` is a display ledger; the authority is the hash-chained
   `Project/audits/audit_events.jsonl`, which shows your **sixteen
   `FINDING_ACCEPTED_ROW_RETIRED` resolutions at 20:54 on 30 Aug**, each with its own signed
   capability nonce. The brake fired on 16 findings and cost 16 signatures to lift, exactly
   as designed. The only honest limit left: it has **never fired on a post-LOCK row**,
   because the recorder (item 2) broke before any campaign row could be adjudicated.

6b. **Verify or drop one clause in §7.** The report says the design review's "final round
   returned APPROVE". Our own memory records a Codex round dying on a provider content
   filter having produced **no verdict line**, with the standing rule that a missing verdict
   is never an APPROVE. I could not confirm round 13 carries a real one. Flagged inline in
   the draft. The thirteen-round count and the ~50 fixes stand either way.

6. **Resolved by measurement, not by archaeology.** §8's "a head-splitting variant came out
   a statistical tie" was untraceable to any file and flagged for deletion. It is now moot:
   head-splitting was built (`k017`), integrated into the shipped dispatcher, and measured
   at **+7.31% on the geomean**. Whatever the old bullet referred to, the claim as written
   is false and the draft must state the measured result instead. **The bullet is replaced,
   not deleted.**
7. **Optional, owed work:** `SENSITIVITY.md` is pre-gate by construction — it reads
   `JOURNAL.jsonl`, which the post-LOCK board never touches — so its MFU column disagrees
   with tech report §2.3 and its `S3 10.95x` is a *withdrawn* figure. The draft now says to
   read it for scoring-convention logic only. Regenerating it against post-LOCK medians
   would need the tool to read the authority packets instead.

*(Housekeeping: `trusted_controller.py status` shows `open_permits: 1` and will forever —
that counter is issued-minus-consumed and a deliberately-unused probe permit never
consumes. `run_gate.py reconcile` returns clean. Nothing is stuck.)*

**Already fixed for you, no action needed** — six hostile-read passes over the three drafts
found twelve defects; all are corrected and committed. Highlights:

- Both the tech report and README claimed shape 8's *baseline* was "already at 64% of the
  fp16 roofline". That is the **candidate's** number; the baseline is at 0.34 MFU and our
  kernel takes it to 0.68. The corrected framing is stronger: 2.02x is what **doubling
  achieved utilisation** looks like.
- **Three documents told the reader to run commands that have not worked since the LOCK**:
  `runner.py run …` (the video's live Scene 3 demo, with "beats while it runs" under it),
  `runner.py check` (Scene 1), and `runner.py leaderboard` (both reproduce blocks). All
  verified dead by running them. Scene 3 now shows the permit refusal as the point, which
  is better television and true.
- Both documents claimed **"every number regenerates from `JOURNAL.jsonl` with one
  command"**. False: the post-LOCK board is in `Project/authority/events.jsonl` plus the
  content-addressed packets under `Project/authority/blobs/`; `JOURNAL.jsonl` holds only
  pre-LOCK history, because screening runs write to the scratch namespace by design.

## 0000. ⬆️ SUPERSEDED BY §0 — this was the 9.45x board, now the *old* column

31 Aug ~04:00 SGT. All twelve shapes measured **on
`Project/submission/torch_transformer_benchmark_submission.py` itself** (sha `4da76db6…`),
not on the kernel modules. That is the artifact a judge would run. **This board is still
valid and still the correct measurement of that file** — it is now the baseline the
split-head rebuild in §0 is compared against, not the headline. Retained in full because
§0's deltas are meaningless without it.

| shape | **shipped file** | kernel module | delta |
| --- | --- | --- | --- |
| 13 | **28.2849x** | 28.4098x | −0.44% |
| 7 | **20.9595x** | 21.9645x | −4.58% |
| 2 | **13.1434x** | 14.3939x | −8.69% |
| 3 | **12.9618x** | 12.6314x | **+2.62%** |
| 11 | **12.5909x** | 12.6797x | −0.70% |
| 12 | **10.4348x** | 10.8141x | −3.51% |
| 5 | **9.1150x** | 9.1536x | −0.42% |
| 4 | **8.9211x** | 8.8774x | **+0.49%** |
| 1 | **8.1673x** | 8.3303x | −1.96% |
| 10 | **6.5352x** | 6.5651x | −0.46% |
| 9 | **4.3503x** | 4.8355x | −10.03% |
| 8 | **2.0160x** | 2.0162x | −0.008% |
| **geomean** | **9.45x** | 9.68x | **−2.4%** |

**Do not quote 9.45x as the headline — see §0.** The module board (9.68x) is a cross-check,
not a result. All twelve `correct: true`, one-use permit per row, quiet box, screening lane.

**Why the module board was not enough.** Those twelve rows measured `k009`/`k010`, chosen
because their headers match the dispatcher's. That is a documentation match — the same
species of reasoning that produced the 22 wrong-kernel attempts this campaign exists to
correct. Card C18 wrote the caveat down at 01:12 and nobody tested it for two hours.

**A hypothesis I preregistered and then killed.** The first four shipped-file rows appeared
to order by candidate time (shape 8 −0.008%, 13 −0.44%, 1 −1.96%, 2 −8.69%), which fits a
fixed ~10 µs per-forward dispatch cost. I preregistered that model on card C32 with eight
point predictions **before** running the rest. Shape 3 came in at **+2.62%** — *higher* than
its proxy, which a fixed cost cannot produce — and shapes 1, 9 and 10 have candidate times
within 4% of each other yet landed at −1.96%, −10.03% and −0.46%. **Model refuted, twice
over.** The spread is ordinary cross-invocation scatter (LESSONS 11, up to ~9% on small
shapes): baseline and candidate are paired *within* each run, so each row is internally
valid, but comparing across separate invocations carries that noise. Shape 9 is an outlier,
not a trend.

## 000. The same board measured on the kernel modules (now a cross-check)

31 Aug ~02:10 SGT. **The correction campaign is finished.** Every locally-runnable shape
has been re-measured on the kernel the dispatcher actually selects, under a one-use permit,
on a verified-quiet box, in the screening lane, `correct: true` on all twelve.

| shape | pre-gate (withdrawn) | **shipped route, measured** | delta | k004 (wrong kernel) |
| --- | --- | --- | --- | --- |
| 13 | 28.82x | **28.4098x** | −1.4% | 5.8096x |
| 7 | 25.57x | **21.9645x** | −14.1% | 3.4781x |
| 2 | 15.26x | **14.3939x** | −5.7% | 8.1115x |
| 11 | 12.98x | **12.6797x** | −2.3% | 4.2433x |
| 3 | 11.96x | **12.6314x** | +5.6% | 7.1845x |
| 12 | 11.44x | **10.8141x** | −5.5% | 3.2334x |
| 5 | 11.40x | **9.1536x** | −19.7% | 2.1475x |
| 4 | 7.30x | **8.8774x** | **+21.6%** | 2.7175x |
| 1 | 10.73x | **8.3303x** | **−22.4%** | 2.1428x |
| 10 | 7.45x | **6.5651x** | −11.9% | 1.5833x |
| 9 | 5.38x | **4.8355x** | −10.1% | 1.1723x |
| 8 † | 2.04x | **2.0162x** | −1.2% | 1.1060x |
| **geomean** | **10.32x** | **9.68x** | **−6.2%** | 2.94x |

† Shape 8 is the only shape on the **other** shipped branch: `d_model` 1024 goes to
`k010_fused_ln.py`, an fp16 tensor-core stack with fp32 accumulation, not the megakernel.
Its file sha (`bda8f703…`) differs from the eleven megakernel runs (`2b96a7c3…`).

**The proxy gap is closed (03:25).** Those twelve rows measure *kernel modules*, chosen
because their headers match the dispatcher's — a documentation match, which is the same
species of reasoning that produced the wrong-kernel board. So the **shipped submission
file itself** (sha `4da76db6…`) was run under its own permits on both dispatcher branches:

| shape | branch | shipped file | proxy | agreement |
| --- | --- | --- | --- | --- |
| 13 | megakernel, `d_model` ≤ 128 | **28.2849x** | 28.4098x | **0.44%** |
| 8 | fp16 stack, `d_model` > 128 | **2.016012x** | 2.016164x | **0.008%** |

Two shapes is the minimum pair that exercises both sides of the only branch the dispatcher
takes — shape 13 alone would show only that the file loads, shape 8 is what tests the
routing predicate. **The board is now a claim about the artifact that ships.**

**Mean delta −5.6%, scatter −22.4% to +21.6%, ten below and two above, uncorrelated with
baseline device idle** — so the spread is harness measurement variation, not bias. The
pre-gate board was **procedurally invalid but numerically close**: that is the honest
two-sided characterisation, and it is the one the report must carry.

### The two mechanism findings this campaign actually earned

1. **The megakernel wins by doing less work, not by recovering launch gaps.** Shape 5 has
   only **1.0% baseline device idle** — there are almost no launch gaps there to recover —
   and it still returned **9.1536x**. Keeping the whole block resident in registers deletes
   memory traffic; it does not merely close bubbles. This is why an idle-fraction argument
   can never bound this mechanism (LESSONS 34).
2. **The megakernel is much flatter across head count than k004.** Over 1 → 16 heads the
   megakernel spans 4.8355x → 12.6797x (a **2.62x** range, and only **1.93x** over the
   2 → 16 heads k004 was compared on), while k004 spans 1.1723x → 4.2433x. The single-head
   case is where k004 nearly vanishes (1.1723x, barely over its 1.03 threshold) and where
   the megakernel still returns **4.8355x**.

The k009-over-k004 ratio ranges **1.76x to 6.31x** — strongly shape-dependent, which is why
the "constant 1.77x" claim was wrong.

### Caveats that must travel with every one of these numbers

- **Screening lane. Nothing here is promotable.** These are characterisation runs; none is
  a champion and none went through the promotion gate.
- **Excludes shape 6 and shape 14** (not locally runnable). The geomean is over the twelve
  that are.
- **Different instrument from the pre-gate column.** These are paired event speedups from
  the trusted controller; the withdrawn 10.32x was the organizers' script against a
  baseline 6–63% off its own calibration. They agree to 6.2%, which is *evidence* the old
  board was roughly right — it is not the same measurement repeated.
- **Audit recording is still broken and owner-only.** No verdict is bound to any of these
  rows. See §1.

**Three of my own claims were refuted during this campaign:**

1. *"The pre-gate board was systematically inflated."* **Wrong.** Procedurally invalid and
   numerically close; I conflated those.
2. *"The shipped route beats k004 by a consistent ~1.77x."* **Wrong** — generalised from
   two points. The ratio spans 1.76x–6.31x.
3. *"2.94x is the corrected headline."* **Wrong**, and it briefly went into three
   judge-facing drafts — understating the shipped route by ~3.5x, the same error I accused
   the drafts of, inverted.

**Numeric prediction bands finished 0 for 26.** Shape 8 missed a 2% band centred on a
figure copied directly off the prior record, by 0.2%. Bands stay retired (LESSONS 35); the
field is filled only because the gate requires it and no conclusion is ever drawn from it.

## 00. ⛔⛔ I MEASURED THE WRONG KERNEL. The 2.94x board is NOT the submission.

Found 31 Aug ~01:05 SGT while auditing the tech report. **This is my error and it
supersedes the framing of every entry below.**

The whole twelve-shape post-LOCK board measured
**`Project/kernels/k004_graphed_triton.py`** — fp32 authored Triton attention plus a
whole-forward CUDA graph. **That is not what this submission ships.**
`Project/submission/dispatcher_region.py` routes:

- `d_model <= 128`, causal, no padding mask → **fused-block megakernel** (k009-class:
  LayerNorm+QKV fused, flash attention over all heads with the output projection folded
  into the head loop, residual + norm + GELU FFN in-register), CUDA-graphed. **That is
  11 of the 12 primary shapes.**
- larger `d_model` → **fp16 tensor-core stack** with fp32 accumulation. **That is shape 8.**

**How it happened:** runbook step 11's worked example profiles
`k004_graphed_triton.py`. I followed the example, made k004 the campaign's candidate, and
never checked it against the dispatcher. 22 attempts characterise a route that does not
ship.

**What survives, precisely:**
- The **12 calibrated noise floors and promotion thresholds** — candidate-independent.
- **Every baseline profile** (launch counts, idle fractions, kernel breakdowns) —
  measured on `k000_baseline.py`, unaffected.
- **The whole regime model** — head width, quadratic sequence, launch-bound small batch —
  because those are *baseline* weaknesses, not properties of k004.
- The method scorecard and all LESSONS.

**What does not survive:** 2.94x as a claim about this submission. It is a valid
controlled measurement of the wrong kernel.

**Correction propagated** to all three judge-facing drafts, which briefly carried my
2.94x as if it replaced the withdrawn 10.32x. They now say plainly that **the shipped
route has no valid post-LOCK measurement**. The video script's speed lower-third is
marked "leave blank".

**Next action (highest value, 38 attempts remain):** re-screen the *shipped* route —
`k009_fused_tuned.py` for `d_model <= 128` shapes and the fp16 stack for shape 8 —
using the same quiet-box screening protocol. The calibrations and baseline profiles are
already in place, so this is fast.

## 0. ⛔ READ FIRST — three judge-facing drafts carried a 3.5x overstatement

Found 31 Aug ~00:30 SGT, **19 hours before freeze**. `Project/drafts/` was written
30 Aug ~11:25, *before* the LOCK and before any of tonight's re-measurement, and three
deliverables quoted the dead pre-gate board as fact:

| file | claimed | measured tonight |
| --- | --- | --- |
| `tech_report_draft.md` §2.1, §2.2 | 10.32× and 10.95× geomean | **2.94×** |
| `track3_readme_draft.md` | 10.32× geomean | **2.94×** |
| `track3_video_script.md` | 10.3× geomean, 28.8× best shape | **2.94×**, best **8.11×** |

~~`devpost_description.md` is clean — it carries no numeric claims.~~

**FALSE. Corrected 31 Aug ~20:20.** `devpost_description.md:11` opens with
**"Geometric-mean 10.3× speedup"**, which is the rounded pre-gate `10.32×` —
the very figure this section withdraws. So the file was not clean, it was
carrying the withdrawn headline in its first sentence, and it was the only one
of the four never checked because this line said it did not need checking. The
number is now replaced with a `[PENDING]` and a stop banner. **A claim that a
file needs no audit is itself a claim and must be tested like any other**
(LESSONS 46 on meta-claims, LESSONS 48 on corrections that skip a file).

**All three now carry an unmissable WITHDRAWN block with the corrected table**, placed
above the stale text rather than replacing it, so the correction is auditable. The
original prose is retained unedited.

This is LESSONS 24 recurring — *"an unsourced number in an informal note becomes a claim
in the report"* — this time in the report itself. **Owner: the drafts still need a full
read-through before freeze; I corrected the headline figures I could verify, but I have
not audited every sentence of 23 KB of report prose for other pre-gate claims.**

## 1. THE ONE THING BLOCKING EVERYTHING — owner only

**Audit results cannot be recorded. Nothing can promote until this is fixed.**

Three audit attempts ran on entry `run-be8e56a55edd1926a84bf5d1efc0b154`, cost roughly
**$7.50** and about 20 minutes of GPU-idle waiting, and **all three failed to record**:

| attempt | failure |
| --- | --- |
| 1 | `verdict does not match full schema` — every required property reported missing |
| 2 | `auditor stdout must be exactly one duplicate-free JSON object with no banners` |
| 3 | `AUDITOR_PROCESS_ABANDONED_WITHOUT_TERMINAL_EVENT` |

The retry cap is exhausted and the entry is escalated to `owner_attention` permanently.
**The auditor itself worked** — both attempts 1 and 2 produced complete, high-quality
verdict documents which are durable at
`Project/authority/blobs/0b3fa1ce…audit-response.json` and `…/92b7c588….audit-response.json`.
Both returned integrity **RETEST** and technical **WEAK_DIAGNOSIS**, and both were correct
about a real attribution error. The failure is in the *recording* path, not the auditing.
`Project/tools/audit_champion.py` and the audit authority are inside the LOCK and
Write-denied to the agent, so this is owner work.

Consequence: **0 of 60 attempts have produced a promotable result, and none can until this
is fixed.** All measurement below is screening-lane, which cannot promote by design.

---

## 2. THE COMPLETE QUIET-BOX BOARD — all 12 primary shapes measured

k004 (authored Triton flash attention + whole-forward CUDA graph capture) against the
unmodified eager baseline. Every run: verified-quiet box (`champion_watch --dry-run`
showing `active: []`, SM 210 MHz, ~1–3% util, ~14 W), campaign timing protocol
(warmup 20, repeats 100, rounds 3), **`correct: true` on every seed**, screening lane,
**0 strikes**.

| shape | B | heads | head_dim | seq | baseline idle | **k004** |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 1 | 4 | 32 | 128 | 86.0% | **8.1115x** |
| 3 | 4 | 4 | 32 | 128 | 82.6% | **7.1845x** |
| 13 | 64 | 4 | 32 | 1024 | — | **5.8096x** |
| 11 | 64 | 16 | 8 | 128 | — | **4.2433x** |
| 7 | 64 | 4 | 8 | 128 | 3.2% | **3.4781x** |
| 12 | 64 | 4 | 32 | 32 | 69.8% | **3.2334x** |
| 4 | 16 | 4 | 32 | 128 | 49.2% | **2.7175x** |
| 5 | 128 | 4 | 32 | 128 | 1.0% | **2.1475x** |
| 1 | 64 | 4 | 32 | 128 | 3.4% | **2.1428x** |
| 10 | 64 | 2 | 64 | 128 | — | **1.5833x** |
| 9 | 64 | 1 | 128 | 128 | — | **1.1723x** |
| 8 | 64 | 4 | 256 | 128 | 0.2% | **1.1060x** |

**Geomean across these twelve: 2.94x.** Minimum 1.1060x — the candidate is **never slower
than the baseline on any measured shape**.

**Caveat on the headline:** the campaign's official scenario is `geomean-shapes-1-13`,
which includes **shape 6** (B=10000). Shape 6 is a dedicated side lane and is NOT in the
table above, so 2.94x is the geomean of the twelve primary shapes and is **not** the
official scenario figure. Do not quote it as such.

---

## 3. What explains the board — four measured baseline weaknesses

None of these is a strength of our kernel. They are all defects of the eager route that
our route simply does not have.

1. **Narrow head dimension is the biggest single lever.** Holding heads at 4 and narrowing
   head_dim from 32 to 8 moves 2.1428x (shape 1) → **3.4781x** (shape 7). Adding twelve
   more heads on top only moves it to 4.2433x (shape 11). *Head width dominates head
   count* — an earlier entry had this backwards and is corrected in DECISIONS.
2. **Quadratic sequence traffic.** At S=1024 the baseline materializes ~1.07 GB of score
   tensor per layer and spends **71.2%** of its device time on `masked_fill`, scale and
   softmax over it. Shape 13 → 5.8096x.
3. **Launch-bound small batch.** Baseline idle fraction orders these cleanly:
   1.0% → 2.1475x, 3.4% → 2.1428x, 49.2% → 2.7175x, 69.8% → 3.2334x, 82.6% → 7.1845x,
   86.0% → 8.1115x. No saturation at the extreme.
4. **Nothing to exploit.** Shape 8 (d=1024, head_dim 256, 99.8% device-busy, ~98% linear)
   has none of the above and lands at 1.1060x.

**Not a factor: problem size.** Shape 5 doubles shape 1's batch and moves the result by
0.2%.

---

## 4. Method finding, and it is the honest headline about process

| kind of claim | record |
| --- | --- |
| numeric prediction bands | **0 for 14** |
| qualitative regime hypotheses with preregistered falsifiers | **6 for 6** |

Every numeric band missed, including one derived from a measured per-kernel breakdown of
the very shape being predicted. Per a commitment preregistered on card C11, numeric bands
were **retired** mid-campaign (LESSONS 35); the gate requires the field, so it is now
filled as an explicitly low-confidence placeholder and no conclusion is drawn from it.

The six qualitative hypotheses all held, including the two hardest cases: one aimed at the
**low** end (shape 8 predicted to be worst — it was) and one **two-sided** bracket (shape 4
required to land inside 2.1428–3.2334 — it landed at 2.7175). Total strike cost of fourteen
consecutive numeric misses: **zero**, because every run was `--prediction-kind
characterization` in the scratch lane.

**I can classify regimes reliably. I cannot forecast magnitudes at all.** That distinction
determines which claims this project's evidence can carry.

---

## 5. Ledger

**CAMP-FINAL** (opened 31 Aug ~00:17 after CAMP-POSTLOCK was retired): ~22 of 60 attempts
spent, **0 promoted**, **0 strikes**, all 12 shapes calibrated in this campaign, 45 research
cycles, no permit armed, lock active and valid at 29 files, campaign not stalled.

Capability `/tmp/cap_grind.json` (`permit.issue`, 200 uses) expires **31 Aug ~21:37 SGT**.
`/tmp/cap_quar.json` (quarantine, 20 uses) expires **31 Aug ~20:46 SGT**. Both are inside
the recording window — **mint fresh ones before filming** (owner action 4).

## 6. Next actions

1. **Owner:** fix the audit recording path. It is still the only thing between this board
   and a promotable, verdict-bound result.
2. **Owner:** `git push -u origin grind-lastday`. `git push` is not on the post-LOCK
   command allowlist, so the agent cannot do it. The branch has never been pushed.
3. Once audit recording is fixed: re-run the strongest shapes in `--mode optimization` and
   audit them. The screening board gives well-founded expectations for every shape.
4. **Closed research target: head_dim 8 padding waste.** This was the open item here and it
   is now done — see §0. The winning route was not the `tl.sum` reduce the research note
   warned about, but lifting the head loop into its own kernel so the out-projection runs
   at full width. Shapes 7 and 11 gained **+26.4%** geomean between them.
5. Remaining open target: **sequence-persistent CTAs** for the launch-bound family
   (2, 3, 7, 12). Untouched, and the last un-attempted idea with a real arithmetic case.

## 7. Standing rules (unchanged)

Never touch frozen/protected files. Every benchmark goes through a permit and the trusted
controller. One GPU process at a time. Never benchmark while an audit runs. Never compare
absolute times across invocations (LESSONS 11). Trace every number to the artifact that
produced it (LESSONS 24). Guard etiquette: never put `clean`, `reset`, `restore` or
`checkout` after `git` in one command segment. Plain language. The owner's explicit "go"
is required before repo actions, and the owner's stop overrides everything, immediately.

## 8. Clock

CODE FREEZE 31 Aug 20:00 SGT → packaging to 1 Sep 02:00 → submission AND Devpost
registration close 1 Sep 12:00 GMT+8.
