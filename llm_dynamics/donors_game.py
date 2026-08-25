"""Dyadic donors game: one LLM vs one set strategy.

Faithful re-implementation of the *training rollout* environment from
donorSim@neurips-methodology (`donor_sim/verl_integration/
donor_game_interaction.py`, stage "direct_indirect"), so that RL-trained
checkpoints are evaluated in-distribution:

  * identical initial prompt (rules, payoff/w/q descriptions, optional
    q-gated reputation report, DECISION: format, optional /no_think),
  * identical per-round result turns ("Round k result: you played X,
    {partner} played Y. You earned ... points ..."),
  * identical simultaneous-move semantics (the scripted partner reacts to
    the model's PRIOR actions only),
  * identical w-gated partner swap (one rng draw per non-terminal round;
    on a swap the per-partner history and the HKB phase window reset and
    a fresh Bernoulli(q) reputation report is rolled),
  * identical HKB phase bookkeeping (window W=4, phi before appending).

The one deliberate deviation: `swap_mode="same"` (default) gives every
successor partner the SAME strategy as the original, so the dyad stays
strategy-controlled for the sweep; `swap_mode="pool"` reproduces the
training distribution (successor drawn from OPPONENT_POOL).
"""
from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path
from typing import Any, Optional

from . import strategies as st

HKB_WINDOW = 4


def initial_prompt(b: float, c: float, w: float, q: float, num_rounds: int,
                   rep_text: str, agent_name: str, partner_name: str,
                   no_think: bool = True) -> str:
    rep_block = f"\n{rep_text}\n" if rep_text else ""
    suffix = " /no_think" if no_think else ""
    return (
        f"You are {agent_name}, playing an iterated Donor's Game over {num_rounds} rounds.\n\n"
        f"Rules:\n"
        f"- Each round you and your partner simultaneously choose COOPERATE or DEFECT.\n"
        f"- If you COOPERATE: you receive 0 points, your partner receives {b:.1f} points.\n"
        f"- If you DEFECT: you receive {c:.1f} points, your partner receives 0 points.\n"
        f"- {st.payoff_description(b, c)}\n"
        f"- {st.w_description(w)}\n"
        f"- {st.q_description(q)}\n\n"
        f"When asked for your decision, reason briefly, then on the final line write exactly "
        f"one of these two lines:\n"
        f"  DECISION: COOPERATE\n"
        f"  DECISION: DEFECT\n\n"
        f"Your partner is {partner_name}.\n"
        f"{rep_block}"
        f"\nThis is round 1 of {num_rounds}. No previous direct interactions with this partner.\n\n"
        f"What is your decision for round 1? End your reply with exactly one of: "
        f"`DECISION: COOPERATE` or `DECISION: DEFECT`.{suffix}"
    )


def round_result_text(current_round: int, num_rounds: int, decision: str,
                      partner: str, opp_move: str, payoff_raw: float,
                      cumulative: float, swap_text: str,
                      no_think: bool = True) -> str:
    is_last = current_round == num_rounds
    if is_last:
        return (
            f"Round {current_round} result: you played {decision}, "
            f"{partner} played {opp_move}. "
            f"You earned {payoff_raw:.1f} points this round "
            f"(total: {cumulative:.1f}).\n\n"
            f"The game has ended after {num_rounds} rounds. "
            f"Final total: {cumulative:.1f} points."
        )
    next_round = current_round + 1
    suffix = " /no_think" if no_think else ""
    return (
        f"Round {current_round} result: you played {decision}, "
        f"{partner} played {opp_move}. "
        f"You earned {payoff_raw:.1f} points this round "
        f"(total so far: {cumulative:.1f})."
        f"{swap_text}\n\n"
        f"This is round {next_round} of {num_rounds}. "
        f"What is your decision for round {next_round}? End your "
        f"reply with exactly one of: `DECISION: COOPERATE` or "
        f"`DECISION: DEFECT`.{suffix}"
    )


def run_donors_game(client, *, b: float, c: float, w: float, q: float,
                    num_rounds: int, opponent_strategy: str, seed: int = 0,
                    swap_mode: str = "same", temperature: float = 0.6,
                    max_tokens: int = 512, no_think: bool = True,
                    parse_failure_action: str = "DEFECT",
                    memory: Optional[int] = None) -> dict[str, Any]:
    """Play one dyadic game; returns {'rows': [...], 'summary': {...}}.

    `memory` (None = full conversation, the training setting) truncates
    the dialogue the model sees to the rules prompt plus the last
    `memory` (assistant, round-result) turn pairs — a memory-m window in
    the donors framing.

    Each row carries the per-round observables used by the phase
    portraits: the committed action, the continuous p_cooperate read from
    the decision-token logprobs, the HKB phase phi, payoffs, and the
    visible state (own/partner last actions with the current partner)."""
    rng = random.Random(seed)
    agent_name, partner_name = st.AGENT_NAMES[0], st.AGENT_NAMES[1]
    partner_idx = 1

    rep_text = st.build_reputation_text(opponent_strategy, q, rng)
    messages = [{"role": "user", "content": initial_prompt(
        b, c, w, q, num_rounds, rep_text, agent_name, partner_name, no_think)}]

    strategy = opponent_strategy
    model_hist: list[str] = []   # vs current partner
    opp_hist: list[str] = []
    window: deque = deque(maxlen=HKB_WINDOW)
    cumulative = 0.0
    rows: list[dict] = []
    n_swaps = 0

    for rnd in range(1, num_rounds + 1):
        visible = messages
        if memory is not None and len(messages) > 1 + 2 * memory:
            visible = [messages[0]] + messages[-2 * memory:]
        reply = client.chat(visible, temperature=temperature,
                            max_tokens=max_tokens)
        decision = st.parse_decision(reply["content"])
        parse_failed = decision is None
        if parse_failed:
            decision = parse_failure_action

        if not opp_hist:
            opp_move = st.opponent_first_move(strategy, rng)
        else:
            opp_move = st.opponent_response(strategy, model_hist, rng, own_history=opp_hist)

        payoff = st.round_payoff(decision, opp_move, b, c, normalized=True)
        payoff_raw = st.round_payoff(decision, opp_move, b, c, normalized=False)
        cumulative += payoff_raw

        phi_now = st.hkb_phi_from_window(window)
        s_t = 1 if decision == opp_move else -1
        window.append(s_t)
        r2 = st.hkb_turn_reward(phi_now, q, b, c)

        state_own = model_hist[-1] if model_hist else None
        state_opp = opp_hist[-1] if opp_hist else None
        model_hist.append(decision)
        opp_hist.append(opp_move)

        rows.append(dict(
            round=rnd, model_action=decision, opp_action=opp_move,
            parse_failed=parse_failed,
            p_cooperate=reply.get("p_cooperate"),
            decision_logit_gap=reply.get("decision_logit_gap"),
            payoff_norm=payoff, payoff_raw=payoff_raw,
            phi=phi_now, s_t=s_t, r2=r2,
            prev_own=state_own, prev_opp=state_opp,
            partner=partner_name, partner_strategy=strategy,
            rounds_with_partner=len(model_hist),
            b=b, c=c, w=w, q=q, seed=seed,
            opponent_strategy=opponent_strategy, swap_mode=swap_mode,
            memory=memory,
        ))

        is_last = rnd == num_rounds
        swap_text = ""
        # One rng draw per non-terminal round, matching training.
        if not is_last:
            coin = rng.random()
            if w < 1.0 and coin >= w:
                n_swaps += 1
                old = partner_name
                partner_idx += 1
                names = [n for n in st.AGENT_NAMES if n != agent_name]
                partner_name = names[(partner_idx - 1) % len(names)]
                while partner_name == old:
                    partner_idx += 1
                    partner_name = names[(partner_idx - 1) % len(names)]
                if swap_mode == "pool":
                    strats, weights = zip(*st.OPPONENT_POOL)
                    strategy = rng.choices(strats, weights=weights, k=1)[0]
                new_rep = st.build_reputation_text(strategy, q, rng)
                model_hist, opp_hist = [], []
                window = deque(maxlen=HKB_WINDOW)
                swap_text = (
                    f"\n\n{old} has moved on and will not play with you again. "
                    f"Your new partner for round {rnd + 1} is {partner_name}."
                )
                if new_rep:
                    swap_text += f"\n{new_rep}"

        messages.append({"role": "assistant", "content": reply["content"]})
        messages.append({"role": "user", "content": round_result_text(
            rnd, num_rounds, decision, rows[-1]["partner"], opp_move,
            payoff_raw, cumulative, swap_text, no_think)})

    valid = [r for r in rows if not r["parse_failed"]]
    coop = [1.0 if r["model_action"] == st.COOPERATE else 0.0 for r in valid]
    opp_coop = [1.0 if r["opp_action"] == st.COOPERATE else 0.0 for r in rows]
    summary = dict(
        b=b, c=c, w=w, q=q, seed=seed, num_rounds=num_rounds,
        opponent_strategy=opponent_strategy, swap_mode=swap_mode,
        memory=memory,
        model=getattr(client, "model", "?"), n_swaps=n_swaps,
        cooperation_rate=(sum(coop) / len(coop)) if coop else None,
        opp_cooperation_rate=sum(opp_coop) / len(opp_coop),
        parse_failures=sum(r["parse_failed"] for r in rows),
        total_payoff_raw=cumulative,
        **st.nowak_thresholds(b, c, w, q),
    )
    return {"rows": rows, "summary": summary}


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
