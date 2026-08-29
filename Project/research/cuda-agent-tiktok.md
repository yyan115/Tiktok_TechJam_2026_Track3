# CUDA Agent (ByteDance Seed + Tsinghua, arxiv 2602.24286) — researched 29 Aug PM

Source: https://cuda-agent.github.io/ · the SPONSOR-ADJACENT paper (organizer slides).
- ReAct loop: profile native torch -> implement -> compile in GPU sandbox ->
  iterate on feedback (5-input correctness vs reference; synchronized warm-up
  profiling; milestone rewards). 2.11x geomean vs torch.compile, 98.8% pass.
- ANTI-REWARD-HACKING DESIGN = OUR HARNESS, independently converged:
  protected verify/profile scripts the agent cannot modify; forbidden
  fallback calls; measurement the agent can't game. (Tech-report narrative:
  we built the same safeguards their RL system needed, before reading them.)
- Skills provided via a structured SKILL.md spec — matches the track's
  "submit the skills you used" deliverable; our loop docs serve this role.
- Copyable without RL: the harness design (have it), data synthesis (n/a),
  SKILL.md-style structured guidance (adopt for the skills deliverable).
