"""Best-response (regret) analysis of played games.

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
    raise ValueError(strategy)


def _payoff_fn(row: dict):
    """Model's per-round payoff function (a_model, a_opp) -> float from row metadata."""
    if "b" in row:
        b, c = row["b"], row["c"]
        return lambda a, o: st.round_payoff(a, o, b, c, normalized=False)
    g = GAMES[row["game"]]
    return lambda a, o: matrix_payoff(a, o, g["R"], g["T"], g["S"], g["P"])


def _expected(payoff, a, o):
    if o == "RANDOM":
        return 0.5 * payoff(a, C) + 0.5 * payoff(a, D)
    return payoff(a, o)


def best_response_value(strategy: str, horizon: int, payoff) -> tuple[float, str]:
    """Oracle optimum over a segment of `horizon` rounds and a compact
    description of the optimal action sequence."""
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
    model = sum(r[pay_key] for r in rows)
    br = sum(best_response_value(strategy, len(s), payoff)[0] for s in segs)
    allc = sum(fixed_policy_value(strategy, len(s), payoff, C) for s in segs)
    alld = sum(fixed_policy_value(strategy, len(s), payoff, D) for s in segs)
    br_seq = best_response_value(strategy, len(segs[0]), payoff)[1]
    return dict(model=model, best_response=br, allc=allc, alld=alld,
                captured=model / br if br else float("nan"),
                regret_per_round=(br - model) / len(rows),
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
                     "regret_per_round", "coop")}
        out[key]["br_seq"] = lst[0]["br_seq_first_segment"]
        out[key]["n"] = len(lst)
    return out


def _fmt(v: dict) -> str:
    return f"{v['coop']:.2f} / {v['captured']:.2f}"


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
            out.append(f"\n### Donors game — LLM memory {mem}  (cell = coop rate / captured optimum)\n")
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
        out.append("\n### Matrix games  (cell = coop rate / captured optimum)\n")
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
        md = comparison_markdown(sys.argv[2:])
        print(md)
        return
    for d in sys.argv[1:]:
        print(f"\n== {d} ==")
        print(f"{'cell':44s} {'model':>7s} {'best':>7s} {'AllC':>7s} {'AllD':>7s} "
              f"{'captured':>9s} {'regret/rd':>9s}  best-response seq (1st segment)")
        for key, v in regret_table(d).items():
            cell = " ".join(key)
            print(f"{cell:44s} {v['model']:7.1f} {v['best_response']:7.1f} "
                  f"{v['allc']:7.1f} {v['alld']:7.1f} {v['captured']:9.2f} "
                  f"{v['regret_per_round']:9.2f}  {v['br_seq']}")


if __name__ == "__main__":
    main()
