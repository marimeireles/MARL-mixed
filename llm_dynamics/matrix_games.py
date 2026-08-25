"""Stateless memory-m matrix games for LLMs: PD, Chicken, Stag Hunt, Harmony.

Unlike the conversational donors harness, every round here is a FRESH
chat: system prompt with the rules + a user turn carrying only the last
m rounds of history. The LLM is therefore *genuinely* a memory-m policy
— its state space is exactly the state space of the CRLD counterpart
(`HistoryEmbedded(SocialDilemma(R,T,S,P), h=(m,m,m))`), which makes the
comparison between the LLM's realized dynamics and the CRLD flow field
exact rather than approximate.

The scripted opponent keeps FULL history internally (Grim really is
grim); only the LLM's window is truncated — memory is the treatment.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from . import strategies as st

# Same payoff tables as donors_crld.MATRIX_GAMES, duplicated here so this
# module stays jax-free (keep in sync).
GAMES: dict[str, dict] = {
    "ipd":      dict(label="Prisoner's Dilemma", R=3.0, T=5.0, S=0.0, P=1.0),
    "chicken":  dict(label="Chicken / Snowdrift", R=3.0, T=5.0, S=1.0, P=0.0),
    "staghunt": dict(label="Stag Hunt", R=5.0, T=3.0, S=0.0, P=1.0),
    "harmony":  dict(label="Harmony", R=5.0, T=3.0, S=2.0, P=0.0),
}


def matrix_payoff(my_action: str, their_action: str, R: float, T: float,
                  S: float, P: float) -> float:
    if my_action == st.COOPERATE:
        return R if their_action == st.COOPERATE else S
    return T if their_action == st.COOPERATE else P


def system_prompt(game: dict, memory: int, partner: str,
                  no_think: bool = True) -> str:
    R, T, S, P = game["R"], game["T"], game["S"], game["P"]
    suffix = " /no_think" if no_think else ""
    return (
        f"You are playing a repeated two-player game against {partner}. "
        f"Each round, you and {partner} simultaneously choose COOPERATE or DEFECT.\n\n"
        f"Payoffs each round:\n"
        f"- You COOPERATE, {partner} COOPERATES: you get {R:g}, they get {R:g}.\n"
        f"- You COOPERATE, {partner} DEFECTS: you get {S:g}, they get {T:g}.\n"
        f"- You DEFECT, {partner} COOPERATES: you get {T:g}, they get {S:g}.\n"
        f"- You DEFECT, {partner} DEFECTS: you get {P:g}, they get {P:g}.\n\n"
        f"You can only remember the outcomes of the last {memory} round(s) of play; "
        f"anything earlier is forgotten.\n\n"
        f"When asked for your decision, reason briefly, then on the final line "
        f"write exactly one of these two lines:\n"
        f"  DECISION: COOPERATE\n"
        f"  DECISION: DEFECT{suffix}"
    )


def history_block(visible: list[tuple[str, str]], partner: str,
                  first_visible_round: int) -> str:
    """Render the LLM's memory window (oldest first).

    `visible` holds (own, opp) pairs; `first_visible_round` is the round
    number of the oldest visible pair."""
    if not visible:
        return "You have no memory of any previous rounds with this partner."
    lines = ["Your memory of recent rounds (oldest first):"]
    for k, (own, opp) in enumerate(visible):
        lines.append(f"- Round {first_visible_round + k}: you played {own}, "
                     f"{partner} played {opp}.")
    return "\n".join(lines)


def round_user_prompt(rnd: int, num_rounds: int, visible: list[tuple[str, str]],
                      partner: str, first_visible_round: int) -> str:
    return (
        f"This is round {rnd} of {num_rounds}.\n\n"
        f"{history_block(visible, partner, first_visible_round)}\n\n"
        f"What is your decision for round {rnd}? End your reply with exactly "
        f"one of: `DECISION: COOPERATE` or `DECISION: DEFECT`."
    )


def build_messages(game_key: str, memory: int, rnd: int, num_rounds: int,
                   visible: list[tuple[str, str]], partner: str = "Bob",
                   no_think: bool = True) -> list[dict[str, str]]:
    game = GAMES[game_key]
    first_visible = rnd - len(visible)
    return [
        {"role": "system", "content": system_prompt(game, memory, partner, no_think)},
        {"role": "user", "content": round_user_prompt(
            rnd, num_rounds, visible, partner, first_visible)},
    ]


def run_matrix_game(client, *, game_key: str, memory: int, num_rounds: int,
                    opponent_strategy: str, seed: int = 0,
                    temperature: float = 0.6, max_tokens: int = 512,
                    no_think: bool = True,
                    parse_failure_action: str = "DEFECT") -> dict[str, Any]:
    """Play one memory-m matrix game; returns {'rows': [...], 'summary': {...}}."""
    game = GAMES[game_key]
    rng = random.Random(seed)
    partner = st.AGENT_NAMES[1]
    model_hist: list[str] = []
    opp_hist: list[str] = []
    rows: list[dict] = []

    for rnd in range(1, num_rounds + 1):
        visible = list(zip(model_hist, opp_hist))[-memory:]
        messages = build_messages(game_key, memory, rnd, num_rounds,
                                  visible, partner, no_think)
        reply = client.chat(messages, temperature=temperature,
                            max_tokens=max_tokens)
        decision = st.parse_decision(reply["content"])
        parse_failed = decision is None
        if parse_failed:
            decision = parse_failure_action

        if not opp_hist:
            opp_move = st.opponent_first_move(opponent_strategy, rng)
        else:
            opp_move = st.opponent_response(opponent_strategy, model_hist, rng, own_history=opp_hist)

        payoff = matrix_payoff(decision, opp_move, game["R"], game["T"],
                               game["S"], game["P"])
        state_label = "|".join(f"{o[0].lower()}{p[0].lower()}"
                               for o, p in visible) or "start"
        model_hist.append(decision)
        opp_hist.append(opp_move)

        rows.append(dict(
            round=rnd, model_action=decision, opp_action=opp_move,
            parse_failed=parse_failed,
            p_cooperate=reply.get("p_cooperate"),
            decision_logit_gap=reply.get("decision_logit_gap"),
            payoff=payoff, visible_state=state_label,
            game=game_key, memory=memory, seed=seed,
            opponent_strategy=opponent_strategy,
        ))

    valid = [r for r in rows if not r["parse_failed"]]
    coop = [1.0 if r["model_action"] == st.COOPERATE else 0.0 for r in valid]
    opp_coop = [1.0 if r["opp_action"] == st.COOPERATE else 0.0 for r in rows]
    summary = dict(
        game=game_key, memory=memory, seed=seed, num_rounds=num_rounds,
        opponent_strategy=opponent_strategy,
        model=getattr(client, "model", "?"),
        cooperation_rate=(sum(coop) / len(coop)) if coop else None,
        opp_cooperation_rate=sum(opp_coop) / len(opp_coop),
        parse_failures=sum(r["parse_failed"] for r in rows),
        mean_payoff=(sum(r["payoff"] for r in valid) / len(valid)) if valid else None,
    )
    return {"rows": rows, "summary": summary}


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
