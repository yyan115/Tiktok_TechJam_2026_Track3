TASK: Make Project/tools/ship_manifest.py verdict-aware. Run this ONLY
AFTER task 03 (its bugfixes) is merged — build on that version.

CONSTRAINTS (hard):
- NEVER modify: torch_transformer_benchmark.py, tensorflow_transformer_benchmark.py,
  README.md, Project/shapes.json, Project/manifest.json, Project/harness/**,
  Project/results/**, .claude/**, Project/loop/**, Project/tools/run_gate.py,
  Project/tools/audit_champion.py, Project/tools/champion_watch.py,
  Project/submission/**, Project/kernels/**, existing files in
  Project/results_side/. Read-only reference is fine.
- NO GPU execution, NO benchmark runs. `python3 -m py_compile` only. Do NOT
  regenerate SHIP_MANIFEST.json (freeze-time only).
- Commit when done. Never place the words 'clean', 'reset' or 'restore'
  after 'git' in any command or commit message.

WHY: an independent audit found the manifest selects the fastest promoted
candidate per shape and only ANNOTATES the audit verdict afterwards — a
RULE_VIOLATION entry could be shipped. Verdicts must gate selection.

SPEC:
1. When selecting each shape's entry, EXCLUDE any journal entry whose
   latest verdict in Project/audits/verdicts.jsonl (match by entry_id;
   latest "recorded" wins) is RULE_VIOLATION or RETEST — unless
   Project/loop/gate_log.jsonl contains a later step=="verdict_clear" row
   whose verdict_key starts with that entry_id + "|". (Read both files
   read-only; treat unreadable/malformed lines as excluding — fail closed.)
2. Prefer entries whose latest verdict is PASS over unaudited ones when
   both exist at equal speedup class; otherwise fastest eligible wins as
   today. Record "audit_verdict" as the latest verdict (or "unaudited").
3. If a shape ends up with NO eligible entry, put an explicit
   {"status": "NO-ELIGIBLE-EVIDENCE"} record for it and make the script
   exit nonzero with a clear message listing such shapes — the manifest
   must never silently drop a shape.
4. Add a --report flag that prints, per shape: chosen entry, its verdict,
   and any excluded-but-faster entries with their verdicts (transparency
   for the final review).
