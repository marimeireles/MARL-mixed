"""Best-response (regret) and welfare (Pareto) analysis of played games.

Two optima per game, both by dynamic programming over the fixed opponent's
state machine and the same partner segments/horizon:
  * best response  — max of the MODEL's payoff  -> `captured`
  * social optimum — max of the JOINT payoff (model + opponent) -> `welfare_captured`.
    Since the opponent is fixed, the Pareto frontier the model can reach is
    a curve in (own, opponent) payoff space; the welfare optimum is its
    utilitarian point, e.g. sustained mutual cooperation against TFT.

For every logged game, compute the payoff an oracle best responder would
have earned against the same fixed strategy over the same horizon and
partner segments (w-swaps wipe the opponent's memory, so segments are
independent), via dynamic programming over the opponent's finite state
machine. Report the model's captured fraction of that optimum alongside
always-cooperate / always-defect baselines.

  python -m llm_dynamics.analysis llm_dynamics/results/<sweep_dir>
"""
from __future__ import annotations

import glob
import json
import statistics as S
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from . import strategies as st
from .matrix_games import GAMES, matrix_payoff

C, D = st.COOPERATE, st.DEFECT


# ── opponent finite-state machines: (init_state, move(state), next(state, model_action)) ──

def _fsm(strategy: str):
    if strategy == "always_cooperate":
        return (), lambda s: C, lambda s, a: s
    if strategy == "always_defect":
        return (), lambda s: D, lambda s, a: s
    if strategy == "tit_for_tat":
        return (C,), lambda s: s[0], lambda s, a: (a,)
    if strategy == "suspicious_tit_for_tat":
        return (D,), lambda s: s[0], lambda s, a: (a,)
    if strategy == "tit_for_two_tats":
        return (C, C), (lambda s: D if s == (D, D) else C), lambda s, a: (s[1], a)
    if strategy == "grim_trigger":
        return (False,), (lambda s: D if s[0] else C), lambda s, a: (s[0] or a == D,)
    if strategy == "random":
        return (), lambda s: "RANDOM", lambda s, a: s
    if strategy == "generous_tit_for_tat":
        # after a model D the partner cooperates w.p. GTFT_FORGIVE: expected payoff
        return (C,), (lambda s: C if s[0] == C else ("MIX", st.GTFT_FORGIVE)), lambda s, a: (a,)
    if strategy == "wsls":
        # state = (partner's own last, model's last); partner opens C
        def move(s):
            own, other = s
            if own is None:
                return C
            return own if other == C else (D if own == C else C)
        return (None, None), move, lambda s, a: (move(s), a)
    raise ValueError(strategy)


def _payoff_fn(row: dict):
    """Model's per-round payoff function (a_model, a_opp) -> float from row metadata."""
    if "b" in row:
        b, c = row["b"], row["c"]
        return lambda a, o: st.round_payoff(a, o, b, c, normalized=False)
    g = GAMES[row["game"]]
    return lambda a, o: matrix_payoff(a, o, g["R"], g["T"], g["S"], g["P"])


def _welfare_fn(row: dict):
    """Joint per-round payoff (model + opponent), symmetric games."""
    pay = _payoff_fn(row)
    return lambda a, o: pay(a, o) + pay(o, a)


def _expected(payoff, a, o):
    if o == "RANDOM":
        return 0.5 * payoff(a, C) + 0.5 * payoff(a, D)
    if isinstance(o, tuple) and o[0] == "MIX":
        return o[1] * payoff(a, C) + (1 - o[1]) * payoff(a, D)
    return payoff(a, o)


def best_response_value(strategy: str, horizon: int, payoff) -> tuple[float, str]:
    """Oracle optimum of `payoff` over a segment of `horizon` rounds and the
    optimal action sequence. Pass the model's payoff for the best response,
    or the joint payoff for the welfare (Pareto/social) optimum."""
    init, move, nxt = _fsm(strategy)

    @lru_cache(maxsize=None)
    def V(t, s):
        if t == horizon:
            return 0.0, ""
        best = None
        for a in (C, D):
            v, seq = V(t + 1, nxt(s, a))
            v += _expected(payoff, a, move(s))
            if best is None or v > best[0] + 1e-12:
                best = (v, a[0] + seq)
        return best

    return V(0, init)


def fixed_policy_value(strategy: str, horizon: int, payoff, action: str) -> float:
    init, move, nxt = _fsm(strategy)
    s, total = init, 0.0
    for _ in range(horizon):
        total += _expected(payoff, action, move(s))
        s = nxt(s, action)
    return total


# Normative "match the partner" reference: what a sensible reciprocator
# should do against each fixed strategy over the whole game. Conditional
# cooperators (TFT, suspicious TFT, TF2T, Grim) -> always cooperate (vs Grim a
# single D triggers permanent D, so all-C dominates all-D); unconditional
# partners are matched; random -> defect (unresponsive).
REFERENCE_ACTION = {
    "always_cooperate": C, "always_defect": D, "random": D,
    "tit_for_tat": C, "suspicious_tit_for_tat": C, "tit_for_two_tats": C,
    "grim_trigger": C, "wsls": C, "generous_tit_for_tat": C,
}


def analyze_game(rows: list[dict]) -> dict:
    rows = sorted(rows, key=lambda r: r["round"])
    strategy = rows[0]["opponent_strategy"]
    payoff = _payoff_fn(rows[0])
    pay_key = "payoff_raw" if "payoff_raw" in rows[0] else "payoff"
    # segments = runs with one partner (donors rows carry rounds_with_partner)
    segs, cur = [], []
    for r in rows:
        if r.get("rounds_with_partner", r["round"]) == 1 and cur:
            segs.append(cur)
            cur = []
        cur.append(r)
    segs.append(cur)
    welfare = _welfare_fn(rows[0])
    model = sum(r[pay_key] for r in rows)
    joint = sum(payoff(r["model_action"], r["opp_action"]) +
                payoff(r["opp_action"], r["model_action"]) for r in rows)
    br_vs = [best_response_value(strategy, len(s), payoff) for s in segs]
    so_vs = [best_response_value(strategy, len(s), welfare) for s in segs]
    br = sum(v for v, _ in br_vs)
    social = sum(v for v, _ in so_vs)
    br_coop = sum(seq.count("C") for _, seq in br_vs) / len(rows)
    social_coop = sum(seq.count("C") for _, seq in so_vs) / len(rows)
    allc = sum(fixed_policy_value(strategy, len(s), payoff, C) for s in segs)
    alld = sum(fixed_policy_value(strategy, len(s), payoff, D) for s in segs)
    br_seq = best_response_value(strategy, len(segs[0]), payoff)[1]
    ref_action = REFERENCE_ACTION[strategy]
    if strategy == "always_defect" and "game" in rows[0]:
        # game-aware: stage-game best response to a defector (C when S > P)
        ref_action = C if payoff(C, D) > payoff(D, D) else D
    reference = sum(fixed_policy_value(strategy, len(s), payoff, ref_action) for s in segs)
    agreement = sum(r["model_action"] == ref_action for r in rows) / len(rows)
    return dict(model=model, best_response=br, allc=allc, alld=alld,
                reference=reference, reference_action=ref_action,
                reference_captured=model / reference if reference else float("nan"),
                agreement=agreement,
                captured=model / br if br else float("nan"),
                regret_per_round=(br - model) / len(rows),
                joint=joint, social_optimum=social,
                welfare_captured=joint / social if social else float("nan"),
                br_coop=br_coop, social_coop=social_coop,
                br_seq_first_segment=br_seq, n_segments=len(segs))


def _mem_tag(r0: dict) -> str:
    m = r0.get("memory")
    return "full" if m is None else f"m{m}"


def _coop_rate(rows: list[dict]) -> float:
    valid = [r for r in rows if not r.get("parse_failed")]
    return (sum(r["model_action"] == C for r in valid) / len(valid)) if valid else float("nan")


def regret_table(results_dir: str | Path) -> dict:
    """Aggregate over seeds: {(strategy, *params): {...}}."""
    files = glob.glob(str(Path(results_dir) / "rounds" / "*.jsonl"))
    per = defaultdict(list)
    for f in files:
        rows = [json.loads(l) for l in open(f) if l.strip()]
        if not rows:
            continue
        r0 = rows[0]
        key = ((r0["opponent_strategy"],) +
               ((f"q={r0['q']:g}", f"w={r0['w']:g}", _mem_tag(r0)) if "q" in r0
                else (r0["game"], _mem_tag(r0))))
        a = analyze_game(rows)
        a["coop"] = _coop_rate(rows)
        per[key].append(a)
    out = {}
    for key, lst in sorted(per.items()):
        out[key] = {k: S.mean(x[k] for x in lst) for k in
                    ("model", "best_response", "allc", "alld", "captured",
                     "regret_per_round", "joint", "social_optimum",
                     "welfare_captured", "coop", "br_coop", "social_coop",
                     "reference", "reference_captured", "agreement")}
        out[key]["reference_action"] = lst[0]["reference_action"]
        out[key]["br_seq"] = lst[0]["br_seq_first_segment"]
        out[key]["n"] = len(lst)
    return out


def _fmt(v: dict) -> str:
    """agreement with reference policy | model/reference payoff | BR captured | welfare captured"""
    return (f"{v['agreement']:.2f} · {v['model']:.0f}/{v['reference']:.0f} · "
            f"{v['captured']:.2f} · {v['welfare_captured']:.2f}")


def comparison_markdown(results_dirs: list[str]) -> str:
    """Markdown tables 'coop rate / captured fraction of best response':
    donors: strategies x (w,q), one table per memory window;
    matrix:  strategies x games, one table per memory."""
    donors, matrix = {}, {}
    for d in results_dirs:
        for key, v in regret_table(d).items():
            (donors if key[1].startswith("q=") else matrix)[key] = v
    out = []
    if donors:
        mems = sorted({k[3] for k in donors}, key=lambda m: (m != "full", m))
        cols = sorted({(k[2], k[1]) for k in donors})
        strats = sorted({k[0] for k in donors})
        for mem in mems:
            out.append(f"\n### Donors game — LLM memory {mem}  (cell = agreement with reference · model/reference payoff · BR captured · welfare captured; reference = all-C vs AllC and conditional cooperators, stage-game best response vs AllD)\n")
            out.append("| vs | " + " | ".join(f"{w} {q}" for w, q in cols) + " |")
            out.append("|---|" + "---|" * len(cols))
            for st_ in strats:
                cells = [_fmt(donors[(st_, q, w, mem)]) if (st_, q, w, mem) in donors else "--"
                         for w, q in cols]
                out.append(f"| {st_} | " + " | ".join(cells) + " |")
    if matrix:
        mems = sorted({k[2] for k in matrix}, key=lambda m: int(m[1:]))
        games = sorted({k[1] for k in matrix})
        strats = sorted({k[0] for k in matrix})
        out.append("\n### Matrix games  (cell = agreement with reference · model/reference payoff · BR captured · welfare captured; reference = all-C vs AllC and conditional cooperators, stage-game best response vs AllD)\n")
        out.append("| vs | memory | " + " | ".join(games) + " |")
        out.append("|---|---|" + "---|" * len(games))
        for st_ in strats:
            for mem in mems:
                cells = [_fmt(matrix[(st_, g, mem)]) if (st_, g, mem) in matrix else "--"
                         for g in games]
                out.append(f"| {st_} | {mem} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    if sys.argv[1:2] == ["table"]:
        print(comparison_markdown(sys.argv[2:]))
        print(pool_markdown(sys.argv[2:]))
        # training-pool prior; strategies outside the pool get the mean pool weight
        mean_w = S.mean(TRAINING_POOL_WEIGHTS.values())
        wts = defaultdict(lambda: mean_w, TRAINING_POOL_WEIGHTS)
        print(pool_markdown(sys.argv[2:], wts,
                            "donorSim training-pool prior (AllD .30, TFT .20, TF2T .20, AllC .15, Random .10, Grim .05; others = mean)"))
        return
    for d in sys.argv[1:]:
        print(f"\n== {d} ==")
        print(f"{'cell':40s} {'model':>6s} {'best':>6s} {'AllC':>6s} {'AllD':>6s} "
              f"{'BRcapt':>7s} {'joint':>6s} {'social':>6s} {'Wcapt':>6s}  BR seq")
        for key, v in regret_table(d).items():
            cell = " ".join(key)
            print(f"{cell:40s} {v['model']:6.1f} {v['best_response']:6.1f} "
                  f"{v['allc']:6.1f} {v['alld']:6.1f} {v['captured']:7.2f} "
                  f"{v['joint']:6.1f} {v['social_optimum']:6.1f} "
                  f"{v['welfare_captured']:6.2f}  {v['br_seq']}")



# ── Uncertainty-aware ("unknown partner") reference ───────────────────────
#
# The per-strategy tables score against the optimum for a KNOWN opponent,
# so opening/probing cooperation against AllD counts as a loss there. Here
# a fixed set of meta-policies that do not know the partner is played
# against every strategy on the same partner segments; the best one under
# the pool prior is the uncertainty-aware yardstick, and the model's
# pool-averaged payoff is compared with it.

def _meta_policy(name: str):
    """Returns f(own_hist, opp_hist, t, horizon) -> action for the segment."""
    if name == "all_c":
        return lambda o, p, t, h: C
    if name == "all_d":
        return lambda o, p, t, h: D
    if name == "tft":
        return lambda o, p, t, h: (p[-1] if p else C)
    if name == "tft_last_d":       # TFT, but defect on the known last round
        return lambda o, p, t, h: D if t == h - 1 else (p[-1] if p else C)
    if name == "probe_once":       # open C; if partner opened D give up for good
        return lambda o, p, t, h: C if not p else (D if p[0] == D else (p[-1] if t < h - 1 else D))
    if name == "grim":
        return lambda o, p, t, h: D if D in p else C
    if name == "pavlov":
        return lambda o, p, t, h: C if not p else (o[-1] if p[-1] == C else (D if o[-1] == C else C))
    raise ValueError(name)


META_POLICIES = ["all_c", "all_d", "tft", "tft_last_d", "probe_once", "grim", "pavlov"]


def meta_policy_value(name: str, strategy: str, seg_lengths: list[int], payoff,
                      n_draws: int = 20) -> float:
    """Expected total payoff of meta-policy `name` vs `strategy` over the
    given partner segments (stochastic strategies averaged over draws)."""
    import random as _r
    pol = _meta_policy(name)
    stochastic = strategy in ("random", "generous_tit_for_tat")
    draws = n_draws if stochastic else 1
    total = 0.0
    for k in range(draws):
        rng = _r.Random(1000 + k)
        for h in seg_lengths:
            own, opp = [], []
            for t in range(h):
                a = pol(own, opp, t, h)
                o = st.opponent_first_move(strategy, rng) if not opp else \
                    st.opponent_response(strategy, own, rng, own_history=opp)
                total += payoff(a, o)
                own.append(a); opp.append(o)
    return total / draws


def pool_table(results_dirs: list[str], weights: dict | None = None) -> dict:
    """Per (w,q,memory) cell: model's pool-averaged payoff vs the best
    blind meta-policy vs the informed optimum (sum of best responses).
    weights: strategy -> prior weight (default: equal over strategies present)."""
    files = [f for d in results_dirs for f in glob.glob(str(Path(d) / "rounds" / "*.jsonl"))]
    games = defaultdict(lambda: defaultdict(list))   # cell -> strategy -> [rows...]
    for f in files:
        rows = [json.loads(l) for l in open(f) if l.strip()]
        if not rows or "q" not in rows[0]:
            continue
        r0 = rows[0]
        games[(f"w={r0['w']:g}", f"q={r0['q']:g}", _mem_tag(r0))][r0["opponent_strategy"]].append(rows)
    out = {}
    for cell, by_strat in sorted(games.items()):
        strats = sorted(by_strat)
        wts = {s_: (weights or {}).get(s_, 1.0) for s_ in strats}
        Z = sum(wts.values())
        model = informed = 0.0
        meta = {m: 0.0 for m in META_POLICIES}
        for s_, glist in by_strat.items():
            payoff = _payoff_fn(glist[0][0])
            pay_key = "payoff_raw"
            m_val = S.mean(sum(r[pay_key] for r in rows) for rows in glist)
            # segments of the first seed's game define the partner schedule
            segs = []
            cur = 0
            for r in sorted(glist[0], key=lambda r: r["round"]):
                if r.get("rounds_with_partner", r["round"]) == 1 and cur:
                    segs.append(cur); cur = 0
                cur += 1
            segs.append(cur)
            inf = sum(best_response_value(s_, h, payoff)[0] for h in segs)
            model += wts[s_] / Z * m_val
            informed += wts[s_] / Z * inf
            for m in META_POLICIES:
                meta[m] += wts[s_] / Z * meta_policy_value(m, s_, segs, payoff)
        best_meta = max(meta, key=meta.get)
        out[cell] = dict(strategies=strats, model=model, informed_optimum=informed,
                         blind_optimum=meta[best_meta], blind_policy=best_meta,
                         model_vs_blind=model / meta[best_meta],
                         model_vs_informed=model / informed, meta=meta)
    return out


TRAINING_POOL_WEIGHTS = dict(st.OPPONENT_POOL)


def pool_markdown(results_dirs: list[str], weights: dict | None = None,
                  title: str = "equal prior over strategies present") -> str:
    t = pool_table(results_dirs, weights)
    if not t:
        return ""
    out = [f"\n### Unknown-partner view — {title} (pool-averaged payoff per game)\n",
           "cell = model / best blind policy (which) / informed optimum; "
           "strategies in pool: " + ", ".join(next(iter(t.values()))["strategies"]) + "\n",
           "| memory | " + " | ".join(f"{w} {q}" for (w, q, m) in t if m == next(iter(t))[2]) + " |"]
    mems = sorted({m for (_, _, m) in t}, key=lambda m: (m != "full", m))
    cols = sorted({(w, q) for (w, q, _) in t})
    out[-1] = "| memory | " + " | ".join(f"{w} {q}" for w, q in cols) + " |"
    out.append("|---|" + "---|" * len(cols))
    for m in mems:
        cells = []
        for w, q in cols:
            v = t.get((w, q, m))
            cells.append("--" if v is None else
                         f"{v['model']:.0f} / {v['blind_optimum']:.0f} ({v['blind_policy']}) / {v['informed_optimum']:.0f}")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    main()
