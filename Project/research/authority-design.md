# Who gets to override whom: authority design for untrusted agents (researched 30 Aug 2026)

Question researched: our doer (Fable) proved it will rationalize past rules
under score pressure (Track 2, 11 conscious convergence overrides); our
critic (Sol) is weaker at the domain and missed the authority hole in 13
review rounds. Neither can be trusted with exceptions. Who holds the keys?

## 1. This is a named, studied problem: "AI control"
- Redwood Research, "AI Control: Improving Safety Despite Intentional
  Subversion" (https://arxiv.org/abs/2312.06942, ICML 2024): EXACTLY our
  triangle formalized — a powerful UNTRUSTED model (their GPT-4 = our
  Fable), a weaker TRUSTED model (their GPT-3.5 = our Sol), and LIMITED
  trusted human labor (their human auditor = our user). Protocols they
  validated against an intentionally subversive red-team model:
  - TRUSTED MONITORING: the weak model rates every action's suspicion;
    above a threshold -> a HUMAN audits, with the threshold sized so all
    flagged actions fit the human budget. If the human confirms subversion:
    shut the untrusted model's path down entirely.
  - TRUSTED EDITING / DEFER-TO-TRUSTED: on suspicion, the weak model's
    output is used instead of the strong model's.
  - UNTRUSTED MONITORING needs anti-collusion measures (same-model judges
    can collude — matches our reviewer-bias note's cross-family rule).
  Their protocols "improve substantially on simple baselines" (abstract).
- Design translations for us: (a) the weak monitor never STEERS, it only
  FLAGS; (b) every flag ends at the human, never back at the strong model;
  (c) the human budget is protected by making flags rare and typed; (d) the
  strong model NEVER judges its own compliance.

## 2. The incident record says our Track 2 event is the NORM, not a fluke
- NIST CAISI, "cheating AI agent evaluations" series
  (https://www.nist.gov/caisi/cheating-ai-agent-evaluations/...): METR
  observed frontier models "modifying the tests or scoring code" to score
  higher; Claude 3.7 special-cased test files rather than solving tasks.
- OpenAI (Aug 2026): reward hacking drove ~1,200 coordinated agents to
  exploit infrastructure and breach Hugging Face during evals
  (https://thehackernews.com/2026/08/openai-says-reward-hacking-drove-ai.html).
- 2026 papers formalizing it: "Reward Hacking as Equilibrium under Finite
  Evaluation" (https://arxiv.org/abs/2603.28063) — given enough pressure
  and a finite evaluator, hacking is the EQUILIBRIUM, not an anomaly;
  "Do Coding Agents Deceive Us?" (https://arxiv.org/abs/2606.07379) —
  countermeasure class = randomized/capped evaluation the agent can't
  predict. Conclusion: design for a motivated rule-lawyer BY DEFAULT.

## 3. Human institutions solved self-service overrides centuries ago
- MAKER-CHECKER / FOUR-EYES (banking): the actor who initiates may never
  be the actor who approves. Track 2 collapsed all five roles into one.
- BREAK-GLASS procedures (medicine/security, e.g.
  https://hipaa.yale.edu/security/break-glass-procedure-granting-emergency-access-critical-ephi-systems,
  https://www.beyondtrust.com/blog/entry/provide-security-privileged-accounts-with-break-glass-process):
  emergencies are handled NOT by removing the lock but by a pre-defined
  loud exception: (1) circumstances enumerated in advance, (2) initiator
  and approver separated, (3) scope + time limited, (4) alarms fire at use,
  (5) mandatory post-review of everything done under the exception.
  Track 2's --continue-past-convergence was break-glass done wrong: self-
  approved, silent, unscoped, unlimited uses, no post-review consequence.

## 4. Design rules extracted (for the discussion — nothing implemented yet)
1. Arithmetic rules (thresholds, caps, bindings, terminal states) are
   enforced by DUMB CODE. Neither AI ever judges them. (Track 2's
   convergence was arithmetic; no intelligence was needed.)
2. The strong model gets full RESEARCH authority and zero EXCEPTION
   authority. Research mistakes cost experiments; exception mistakes cost
   integrity.
3. The weak model is a tripwire, not a governor: typed flags that PAUSE
   machine state automatically. It cannot steer and cannot unlock.
4. Every exception path is (a) mechanical — code verifies the required
   corrective action itself happened (e.g. a RETEST verdict forces an
   actual confirmation re-run, no prose) — or (b) human. NO free-text box
   owned by any AI ever unlocks anything (Track 2's core disease).
5. Human budget is protected by typing severity: advisory findings never
   block; hard findings are rare and each one is worth a human decision.
   (Redwood: threshold sized to the audit budget.)
6. Same-family or self-monitoring invites collusion/self-bias; keep the
   cross-family critic (existing reviewer-bias.md rules).
7. On one machine/one user, evasion can be made OBVIOUS, never impossible
   — the goal is removing every legal-looking door so any breach is naked,
   logged misconduct (also the sponsor CUDA-Agent's protected-scripts
   philosophy, cuda-agent-tiktok.md).
