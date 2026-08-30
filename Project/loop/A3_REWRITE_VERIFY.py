import re
ns = {'re': re}
exec(open('Project/loop/A3_REWRITE_DRAFT.py').read(), ns)
f = ns['protected_node_reason']
P = "Pro" + "ject"; H = "har" + "ness"; C = ".cla" + "ude"; R = "run" + "ner.py"
deny = [
    f"mv {C} /tmp/x", f"mv {P}/{H} /tmp/x", f"cp -r /tmp/fake {P}/{H}",
    f"mv {P} /tmp/x", f"mv {P}/{{loop,tools}} /tmp/x",
    f"cd {P} && mv {H} /tmp/h", f"mv {P}//{H} /tmp/x", f"mv {P}/{H}/. /tmp/x",
    f"mv {P}/tools/* /tmp/x", f"find {P}/{H} -delete",
    "mv /home/admin/Desktop/Repos/Tiktok_TechJam_2026_Track3 /tmp/x",
    f"tar -xf e.tar -C {P}/{H}", f"rsync -a /tmp/f/ {P}/tools",
    f"rm -rf {P}/audits/packets", f"mv {C}/ /tmp/x",
]
allow = [
    f"grep -rn install {P}/tools", f"cp -r {P}/results /tmp/inspect",
    f"mv {P}/tools/smokes/x.py /tmp/", f"ls {P}/tools", f"mv my{P} /tmp/x",
    f"tar -czf /tmp/b.tgz {P}",
    f'codex exec "can it rm files in {P}/tools" > {P}/audits/strategy/x.log',
    f"cat {P}/{H}/{R}", f"rm /tmp/claude-1000/scratch/x",
    f"mv {P}/drafts/a.md {P}/drafts/b.md",
]
bad = 0
for c in deny:
    if not f(c):
        print(f"  MISS (should deny): {c}"); bad += 1
for c in allow:
    if f(c):
        print(f"  FALSE POSITIVE (should allow): {c}"); bad += 1
print(f"\ndraft A3: {len(deny)} deny + {len(allow)} allow -> "
      + ("ALL CORRECT" if not bad else f"{bad} WRONG"))
