# DRAFT — token-based rewrite of guard Block A3 (round 7). NOT SPLICED IN.
# Replaces the unsound A3 currently in OWNER_PATCH_card_gate.md.
# Paste BOTH blocks: helpers first (A2 also calls _resolved_segments),
# then the A3 function. See ROUND7_FINDINGS.md items 6 and 7.
# VERIFIED 25/25 by Project/loop/A3_REWRITE_VERIFY.py (15 deny incl. every
# hole the round-7 reviewers found, 10 allow incl. every false positive they
# found). Re-run that script after splicing, and fold its cases into
# Project/tools/tests/guard_and_auditor_test.py node_cases().

# --- Shared path resolution for Blocks A2/A3 (review round 7). Commands
# routinely `cd` first and then name files relatively, join paths with
# doubled slashes, or use brace expansion. Matching only literal
# repo-prefixed text missed every one of those spellings.
def _brace_expand(tok):
    m = re.match(r"(.*?)\{([^}]*)\}(.*)$", tok)
    if not m:
        return [tok]
    pre, body, post = m.groups()
    out = []
    for alt in body.split(","):
        out.extend(_brace_expand(pre + alt + post))
    return out


def _resolved_segments(norm):
    """Segments with `cd` folded into later relative operands, braces
    expanded and duplicate slashes collapsed. The head token is never
    rewritten, so Block A2's command carve-outs still match from the start."""
    out, cwd = [], ""
    for seg in re.split(r"[|;&\n\r]+", norm):
        toks = seg.split()
        if not toks:
            continue
        if toks[0] == "cd" and len(toks) == 2 and not toks[1].startswith("-"):
            p = toks[1]
            cwd = p if p.startswith("/") else (cwd.rstrip("/") + "/" + p).lstrip("/")
            continue
        new = [toks[0]]
        for t in toks[1:]:
            for e in _brace_expand(t):
                m = re.match(r"^([<>]+)(.*)$", e)
                pre, rest = (m.group(1), m.group(2)) if m else ("", e)
                rest = re.sub(r"/+", "/", rest)
                if (cwd and rest and not rest.startswith(("/", "-"))
                        and "=" not in rest):
                    rest = cwd.rstrip("/") + "/" + rest
                new.append(pre + rest)
        out.append(" ".join(new))
    return out


# --- Protected directory NODES (review rounds 6-7). WRITE_PATTERNS and
# Block A2 both match paths INSIDE the protected trees; naming a tree's own
# root as an operand matched neither. Round 7 rebuilt this on TOKENS rather
# than a lookahead: the op must be the segment's HEAD (so `grep -rn install
# Project/tools` and `codex exec "...rm... Project/tools"` are reads, not
# writes, and A2's carve-outs are not shadowed), and each operand is
# compared whole after cd/brace/slash resolution.
def protected_node_reason(command):
    norm = command.replace("\\\n", "").replace('"', "").replace("'", "")
    NODE = (r"(?:\.claude|Tiktok_TechJam_2026_Track3|Project"
            r"(?:/(?:harness|results|loop|tools|"
            r"audits(?:/(?:packets|strategy|auto))?))?)")
    WHOLE = re.compile(r"(?:.*/)?" + NODE + r"/?[.*]?$")
    OPS = ("mv", "cp", "rm", "rmdir", "ln", "rsync", "install",
           "shred", "dd", "unzip", "rename")
    PRE = ("sudo", "env", "time", "nohup", "xargs", "command")
    for seg in _resolved_segments(norm):
        toks = seg.split()
        i = 0
        while i < len(toks) and (re.match(r"^\w+=\S*$", toks[i])
                                 or toks[i] in PRE):
            i += 1
        if i >= len(toks):
            continue
        op = toks[i].split("/")[-1]
        rest = toks[i + 1:]
        flags = [t for t in rest if t.startswith("-")]
        operands = [t for t in rest if not t.startswith("-")
                    and "=" not in t and not re.match(r"^[<>&|]+$", t)]
        if op == "find":
            if not any(f in ("-delete", "-exec") for f in rest):
                continue          # a plain find is a read
        elif op == "tar":
            if not any("x" in f for f in flags):
                continue          # creating an archive reads; extracting writes
        elif op not in OPS:
            continue
        # cp/rsync read FROM a tree unless the tree is the destination, so
        # `cp -r Project/results /tmp/inspect` (analysis) stays allowed.
        check = (operands[-1:] if op in ("cp", "rsync") and len(operands) > 1
                 else operands)
        for o in check:
            if WHOLE.match(o):
                return ("Blocked: that names a protected tree's ROOT (" + o +
                        ") as an operand. Moving, replacing, emptying or "
                        "unpacking over .claude/, Project/harness, results, "
                        "loop, tools, audits or the repo root relocates the "
                        "rules, the referee or the evidence wholesale — the "
                        "per-file guards would never see it. Owner lifts "
                        "this by hand.")
    return None
