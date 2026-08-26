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
                    memory: Optional[int | str] = None,
                    perturb_round: Optional[int] = None,
                    perturb_action: str = "DEFECT",
                    log_content: bool = True) -> dict[str, Any]:
    """Play one dyadic game; returns {'rows': [...], 'summary': {...}}.

    `memory` (None = full conversation, the training setting) truncates
    the dialogue the model sees to the rules prompt plus the last
    `memory` (assistant, round-result) turn pairs — a memory-m window in
    the donors framing. `memory="note2"` keeps the last 2 pairs and replaces
    everything older by a compaction-style summary note (the training-time
    context compaction, approximated).

    `perturb_round` forces the model's action to `perturb_action` on that
    round (the model's own decision is still logged as `intended_action`);
    used to measure repair time after a defection.

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

    note_mode = isinstance(memory, str) and memory.startswith("note")
    mem_n = int(memory[4:]) if note_mode else memory

    for rnd in range(1, num_rounds + 1):
        visible = messages
        if mem_n is not None and len(messages) > 1 + 2 * mem_n:
            visible = [messages[0]] + messages[-2 * mem_n:]
            if note_mode:
                # compaction note summarising the dropped turns with the current partner
                k = len(model_hist) - mem_n
                if k > 0:
                    mc = sum(a == st.COOPERATE for a in model_hist[:k])
                    oc = sum(a == st.COOPERATE for a in opp_hist[:k])
                    note = (f"[Memory note] Earlier rounds with {partner_name} (not shown): "
                            f"over {k} rounds you cooperated {mc} times and {partner_name} "
                            f"cooperated {oc} times. Your running total is {cumulative:.1f} points.")
                    visible = [messages[0], {"role": "user", "content": note},
                               {"role": "assistant", "content": "Noted."}] + messages[-2 * mem_n:]
        reply = client.chat(visible, temperature=temperature,
                            max_tokens=max_tokens)
        intended = st.parse_decision(reply["content"])
        parse_failed = intended is None
        if parse_failed:
            intended = parse_failure_action
        decision = perturb_action if (perturb_round is not None and rnd == perturb_round) else intended

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
            intended_action=intended, perturbed=(decision != intended),
            parse_failed=parse_failed,
            content=(reply.get("content") if log_content else None),
            p_cooperate=reply.get("p_cooperate"),
            decision_logit_gap=reply.get("decision_logit_gap"),
            payoff_norm=payoff, payoff_raw=payoff_raw,
            phi=phi_now, s_t=s_t, r2=r2,
            prev_own=state_own, prev_opp=state_opp,
            partner=partner_name, partner_strategy=strategy,
            rounds_with_partner=len(model_hist),
            b=b, c=c, w=w, q=q, seed=seed,
            opponent_strategy=opponent_strategy, swap_mode=swap_mode,
            memory=memory, perturb_round=perturb_round,
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

        assistant_text = reply["content"]
        if decision != intended:
            assistant_text = f"DECISION: {decision}"
        messages.append({"role": "assistant", "content": assistant_text})
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


def run_donors_selfplay(client_a, client_b, *, b: float, c: float, w: float,
                        q: float, num_rounds: int, seed: int = 0,
                        temperature: float = 0.6, max_tokens: int = 512,
                        no_think: bool = True,
                        parse_failure_action: str = "DEFECT") -> dict[str, Any]:
    """Two LLMs play each other (simultaneous moves), each with its own
    conversation in the same training-faithful framing. w-swaps are
    disabled (a swap would need a third player); q gates a reputation
    report that describes the other model as a 'tit_for_tat'-like player.
    Rows are per round with both sides' actions and p_cooperate."""
    rng = random.Random(seed)
    names = (st.AGENT_NAMES[0], st.AGENT_NAMES[1])
    reps = [st.build_reputation_text("tit_for_tat", q, rng) for _ in range(2)]
    msgs = [[{"role": "user", "content": initial_prompt(
        b, c, w, q, num_rounds, reps[i], names[i], names[1 - i], no_think)}]
        for i in range(2)]
    clients = (client_a, client_b)
    hist = [[], []]
    cum = [0.0, 0.0]
    rows = []
    for rnd in range(1, num_rounds + 1):
        replies = [clients[i].chat(msgs[i], temperature=temperature,
                                   max_tokens=max_tokens) for i in range(2)]
        acts, failed = [], []
        for r in replies:
            a = st.parse_decision(r["content"])
            failed.append(a is None)
            acts.append(a or parse_failure_action)
        pay = [st.round_payoff(acts[i], acts[1 - i], b, c, normalized=False) for i in range(2)]
        for i in range(2):
            cum[i] += pay[i]
        rows.append(dict(
            round=rnd, model_action=acts[0], opp_action=acts[1],
            a_action=acts[0], b_action=acts[1],
            a_p_cooperate=replies[0].get("p_cooperate"),
            b_p_cooperate=replies[1].get("p_cooperate"),
            p_cooperate=replies[0].get("p_cooperate"),
            a_payoff_raw=pay[0], b_payoff_raw=pay[1], payoff_raw=pay[0],
            parse_failed=failed[0] or failed[1],
            prev_own=(hist[0][-1] if hist[0] else None),
            prev_opp=(hist[1][-1] if hist[1] else None),
            b=b, c=c, w=w, q=q, seed=seed, rounds_with_partner=rnd,
            opponent_strategy=f"selfplay:{getattr(client_b, 'model', '?')}",
            a_model=getattr(client_a, "model", "?"), b_model=getattr(client_b, "model", "?"),
        ))
        for i in range(2):
            hist[i].append(acts[i])
            msgs[i].append({"role": "assistant", "content": replies[i]["content"]})
            msgs[i].append({"role": "user", "content": round_result_text(
                rnd, num_rounds, acts[i], names[1 - i], acts[1 - i], pay[i], cum[i], "", no_think)})
    coop = [[a == st.COOPERATE for a in hist[i]] for i in range(2)]
    return {"rows": rows, "summary": dict(
        b=b, c=c, w=w, q=q, seed=seed, num_rounds=num_rounds,
        a_model=rows[0]["a_model"], b_model=rows[0]["b_model"],
        a_cooperation_rate=sum(coop[0]) / num_rounds,
        b_cooperation_rate=sum(coop[1]) / num_rounds,
        a_total=cum[0], b_total=cum[1], mutual_c_rate=sum(
            x and y for x, y in zip(*coop)) / num_rounds)}


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
