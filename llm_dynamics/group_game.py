"""Group-selection stage evaluation (the environment the released RL run
actually trained on).

Mirrors donorSim@neurips-methodology `sample_scenario_group` +
`DonorGameInteraction` (group stage): the model is one member of a
K-player group competing with G-1 other groups, plays a fixed partner
(groupmate #0) for all N rounds, and each round emits `PREDICT: <g>` (its
forecast of the group's cooperation rate) and `DECISION: <a>`. Thinking is
ON in this stage (no /no_think suffix).

Groupmates and competing groups are scripted (Mode A): their actions come
from their strategies playing pairwise among themselves, their
predictions from a strategy prior + N(0, 0.05) noise, as in training. From
these we compute per round the CFE (Term 3), the model's own Brier score
(is its PREDICT calibrated?), the normalized payoff (Term 1), the
reciprocation signal rho (Term 2 envelope), and at the end the
group-survival bonus — so the trained model can be scored on the exact
reward it was trained on, and the base on the same scale.
"""
from __future__ import annotations

import math
import random
import re
from typing import Any, Optional

from . import strategies as st

_PREDICT_RE = re.compile(r"PREDICT\s*:\s*(-?[0-9]*\.?[0-9]+)", re.IGNORECASE)
STRAT_PRIOR = {"always_cooperate": 1.0, "always_defect": 0.0, "tit_for_tat": 0.85,
               "tit_for_two_tats": 0.90, "grim_trigger": 0.70, "random": 0.5}


def parse_prediction(text: str, default: Optional[float] = 0.5) -> Optional[float]:
    m = _PREDICT_RE.search(text or "")
    if not m:
        return default
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except ValueError:
        return default


def bern_kl(p, q, eps=1e-6):
    p = min(max(p, eps), 1 - eps); q = min(max(q, eps), 1 - eps)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def cfe(predictions: list[float], actions_coop: list[int]) -> float:
    """Paper Eq. (cfe), squared-error consensus term (current spec)."""
    K = len(predictions)
    g_bar = sum(predictions) / K
    c_bar = sum(actions_coop) / K
    return sum(bern_kl(p, g_bar) for p in predictions) / K + (g_bar - c_bar) ** 2


def group_prompt(b, c, w, q, rounds, group_size, num_groups, rep_text,
                 name, partner, members) -> str:
    rep_block = f"\n{rep_text}\n" if rep_text else ""
    return (
        f"You are {name}, a member of a {group_size}-player group competing against "
        f"{num_groups - 1} other groups.\n\n"
        f"Rules:\n"
        f"- Within your group, you play repeatedly against one fellow member for all {rounds} rounds.\n"
        f"- If you COOPERATE: you receive 0 points, your partner receives {b:.1f} points.\n"
        f"- If you DEFECT: you receive {c:.1f} points, your partner receives 0 points.\n"
        f"- {st.payoff_description(b, c)}\n"
        f"- {st.w_description(w)}\n"
        f"- {st.q_description(q)}\n"
        f"- After {rounds} rounds, the lowest-scoring group is eliminated.\n"
        f"- Within a surviving group, the highest-scoring individual wins overall.\n\n"
        f"Strategic context:\n"
        f"- Defecting raises your individual score relative to cooperating groupmates.\n"
        f"- BUT if your group's total score is lowest, your whole group is eliminated and you lose.\n"
        f"- Cooperating helps your group survive at a cost to your individual rank.\n\n"
        f"Each round, you will (1) predict how many of your group members "
        f"(including yourself) will cooperate, then (2) decide whether YOU "
        f"will cooperate. Reason briefly, then finish with exactly two "
        f"lines in this format (the second line must be one of the two "
        f"shown options, not both):\n"
        f"  PREDICT: <number between 0.0 and 1.0>\n"
        f"  DECISION: COOPERATE         (or: DECISION: DEFECT)\n\n"
        f"Your group: {name}, {members}\n"
        f"Your partner this round is {partner}.\n"
        f"{rep_block}"
        f"\nThis is round 1 of {rounds}. No previous interactions.\n\n"
        f"What is your prediction and decision for round 1?"
    )


def round_text(rnd, rounds, decision, partner, opp_move, payoff_raw, cum) -> str:
    if rnd == rounds:
        return (f"Round {rnd} result: you played {decision}, {partner} played {opp_move}. "
                f"You earned {payoff_raw:.1f} points this round (total: {cum:.1f}).\n\n"
                f"The game has ended after {rounds} rounds. Final total: {cum:.1f} points.")
    return (f"Round {rnd} result: you played {decision}, {partner} played {opp_move}. "
            f"You earned {payoff_raw:.1f} points this round (total so far: {cum:.1f}).\n\n"
            f"This is round {rnd + 1} of {rounds}. "
            f"First predict the fraction of your group members (including yourself) that will "
            f"COOPERATE this round, then give your decision.\n"
            f"End your reply with exactly two lines (second line must be one of the two options, not both):\n"
            f"  PREDICT: <number between 0.0 and 1.0>\n"
            f"  DECISION: COOPERATE          (or: DECISION: DEFECT)")


def sample_group_scenario(rng: random.Random) -> dict:
    """Same distribution as training's sample_scenario_group."""
    group_size = rng.choice([4, 6]); num_groups = rng.randint(2, 4)
    rounds = rng.randint(5, 10)
    b = rng.uniform(3.0, 6.0); c = b * rng.uniform(0.1, 1.0)
    w = 1.0; q = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])
    strategies, weights = zip(*st.OPPONENT_POOL)
    mates = [rng.choices(strategies, weights=weights, k=1)[0] for _ in range(group_size - 1)]
    competing = [[rng.choices(strategies, weights=weights, k=1)[0] for _ in range(group_size)]
                 for _ in range(num_groups - 1)]
    rep = st.build_reputation_text(mates[0], q, rng)
    return dict(group_size=group_size, num_groups=num_groups, rounds=rounds, b=b, c=c,
                w=w, q=q, groupmates=mates, competing=competing, rep_text=rep)


def _pair_play(strats, hist_self, hist_opp, rng):
    """One round of a scripted group playing pairwise among themselves."""
    k = len(strats); idx = list(range(k)); rng.shuffle(idx)
    acts = [None] * k
    for i in range(0, k - 1, 2):
        a, b_ = idx[i], idx[i + 1]
        ma = st.opponent_first_move(strats[a], rng) if not hist_opp[a] else \
            st.opponent_response(strats[a], hist_opp[a], rng, own_history=hist_self[a])
        mb = st.opponent_first_move(strats[b_], rng) if not hist_opp[b_] else \
            st.opponent_response(strats[b_], hist_opp[b_], rng, own_history=hist_self[b_])
        acts[a], acts[b_] = ma, mb
        hist_self[a].append(ma); hist_self[b_].append(mb)
        hist_opp[a].append(mb); hist_opp[b_].append(ma)
    if k % 2:  # odd sitter cooperates by default (kept for safety; groups are even)
        acts[idx[-1]] = st.COOPERATE
    return acts


def run_group_game(client, scenario: dict, *, seed: int = 0, temperature: float = 0.8,
                   max_tokens: int = 4096, lambda_tom: float = 0.1,
                   lambda_g: float = 0.5, log_content: bool = True) -> dict[str, Any]:
    rng = random.Random(seed)
    sc = scenario
    b, c, q, w, N = sc["b"], sc["c"], sc["q"], sc["w"], sc["rounds"]
    K = sc["group_size"]
    name = st.AGENT_NAMES[0]
    members = ", ".join(st.AGENT_NAMES[1:K])
    partner = st.AGENT_NAMES[1]
    mates = sc["groupmates"]                 # mates[0] is the partner
    messages = [{"role": "user", "content": group_prompt(
        b, c, w, q, N, K, sc["num_groups"], sc["rep_text"], name, partner, members)}]
    model_hist, opp_hist = [], []
    others = mates[1:]
    oth_self = [[] for _ in others]; oth_opp = [[] for _ in others]
    comp_hist = [([[] for _ in g], [[] for _ in g]) for g in sc["competing"]]
    comp_tot = [0.0] * len(sc["competing"])
    cum = 0.0; group_norm_total = 0.0
    rows = []
    for rnd in range(1, N + 1):
        reply = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
        text = reply["content"]
        decision = st.parse_decision(text); failed = decision is None
        decision = decision or "DEFECT"
        pred = parse_prediction(text, default=0.5)
        opp = st.opponent_first_move(mates[0], rng) if not opp_hist else \
            st.opponent_response(mates[0], model_hist, rng, own_history=opp_hist)
        pay_n = st.round_payoff(decision, opp, b, c, normalized=True)
        pay_raw = st.round_payoff(decision, opp, b, c, normalized=False)
        opp_pay_n = st.round_payoff(opp, decision, b, c, normalized=True)
        cum += pay_raw
        # rho: match partner's previous action (0 on round 1)
        rho = 0 if not opp_hist else (1 if decision == opp_hist[-1] else -1)
        r2 = lambda_tom * (4 * q + c / b) * rho
        # other groupmates play among themselves; predictions from prior
        oth_acts = _pair_play(others, oth_self, oth_opp, rng) if others else []
        preds = [pred] + [max(0, min(1, rng.gauss(STRAT_PRIOR.get(s, 0.5), 0.05)))
                          for s in mates]
        acts = [decision == st.COOPERATE, opp == st.COOPERATE] + [a == st.COOPERATE for a in oth_acts]
        cfe_val = cfe(preds, [int(a) for a in acts])
        realized = sum(acts) / len(acts)
        brier = (pred - realized) ** 2
        group_norm_total += pay_n + opp_pay_n + sum(
            st.round_payoff(a, o, b, c, normalized=True) for a, o in
            zip(oth_acts, [oth_opp[i][-1] for i in range(len(others))]) if o is not None)
        for gi, (g, (hs, ho)) in enumerate(zip(sc["competing"], comp_hist)):
            ga = _pair_play(g, hs, ho, rng)
            comp_tot[gi] += sum(st.round_payoff(a, ho[i][-1], b, c, normalized=True)
                                for i, a in enumerate(ga) if a is not None and ho[i])
        model_hist.append(decision); opp_hist.append(opp)
        rows.append(dict(round=rnd, model_action=decision, opp_action=opp, parse_failed=failed,
                         p_cooperate=reply.get("p_cooperate"), prediction=pred,
                         realized_group_coop=realized, brier=brier, cfe=cfe_val,
                         r1=pay_n, rho=rho, r2=r2, r3=-lambda_g * cfe_val,
                         payoff_raw=pay_raw, b=b, c=c, w=w, q=q, K=K, G=sc["num_groups"],
                         seed=seed, opponent_strategy=mates[0], groupmates=mates,
                         prev_own=(model_hist[-2] if len(model_hist) > 1 else None),
                         prev_opp=(opp_hist[-2] if len(opp_hist) > 1 else None),
                         rounds_with_partner=rnd, memory=None,
                         content=(text if log_content else None)))
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": round_text(rnd, N, decision, partner, opp, pay_raw, cum)})
    avg_comp = (sum(comp_tot) / len(comp_tot)) if comp_tot else 0.0
    bonus = max(-1.0, min(1.0, 0.5 * (group_norm_total - avg_comp) / (K * N)))
    R = (sum(r["r1"] + r["r2"] + r["r3"] for r in rows) + bonus) / N
    summary = dict(seed=seed, b=b, c=c, q=q, K=K, G=sc["num_groups"], rounds=N,
                   partner=mates[0], model=getattr(client, "model", "?"),
                   cooperation_rate=sum(r["model_action"] == st.COOPERATE for r in rows) / N,
                   mean_r1=sum(r["r1"] for r in rows) / N,
                   mean_rho=sum(r["rho"] for r in rows[1:]) / max(1, N - 1),
                   mean_r2=sum(r["r2"] for r in rows) / N,
                   mean_cfe=sum(r["cfe"] for r in rows) / N,
                   mean_brier=sum(r["brier"] for r in rows) / N,
                   bonus=bonus, trajectory_scalar=R,
                   parse_failures=sum(r["parse_failed"] for r in rows))
    return {"rows": rows, "summary": summary}
