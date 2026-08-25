"""Measure an LLM's conditional policy P(cooperate | history state).

The CRLD side of this project describes a learner as a point X[s, a] in
policy space. These probes measure the same object for an LLM: for every
memory-m history state we render the exact prompt the game harness would
render and read P(cooperate) off the decision-token logprobs. The result
is the LLM as a point in the same policy space the flow fields live in —
base model vs RL-trained checkpoint become two points (or two pinned
vector fields) in one portrait.

Two prompt framings:
  * matrix  — the stateless memory-m framing (exact state match to CRLD),
  * donors  — a conversational replay of the donors-game dialogue (forced
    assistant turns), matching the RL training distribution.
"""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Any, Optional

from . import strategies as st
from . import matrix_games as mg
from . import donors_game as dg

ACTIONS = (st.COOPERATE, st.DEFECT)


def enumerate_states(memory: int) -> list[tuple[tuple[str, str], ...]]:
    """All 4^m full-length histories, each a tuple of (own, opp) pairs,
    oldest -> newest."""
    pairs = list(itertools.product(ACTIONS, ACTIONS))
    return [tuple(h) for h in itertools.product(pairs, repeat=memory)]


def state_label(state: tuple[tuple[str, str], ...]) -> str:
    """(('COOPERATE','DEFECT'),) -> 'cd' ; two rounds -> 'cd|cc' (own,opp
    initials, oldest -> newest). Matches donors_crld.state_index rounds
    via [(own[0], opp[0]), ...] with the learner as agent 0."""
    return "|".join(f"{o[0].lower()}{p[0].lower()}" for o, p in state)


def _p_from_reply(reply: dict) -> Optional[float]:
    p = reply.get("p_cooperate")
    if p is not None:
        return float(p)
    action = st.parse_decision(reply.get("content", ""))
    if action is None:
        return None
    return 1.0 if action == st.COOPERATE else 0.0


def _probe_messages_matrix(state, game_key: str, memory: int,
                           probe_round: int, num_rounds: int,
                           no_think: bool = True) -> list[dict]:
    visible = [(own, opp) for own, opp in state]
    return mg.build_messages(game_key, memory, probe_round, num_rounds,
                             visible, no_think=no_think)


def probe_matrix_policy(client, *, game_key: str, memory: int,
                        probe_round: int = 10, num_rounds: int = 25,
                        temperature: float = 0.0, max_tokens: int = 512,
                        samples: int = 1, seed: int = 0,
                        no_think: bool = True) -> dict[str, Any]:
    """P(cooperate | s) for all 4^m memory-m states under the matrix
    framing. probe_round/num_rounds place the probe mid-game, away from
    first-round and end-game effects."""
    states = enumerate_states(memory)
    out: dict[str, Any] = {}
    for state in states:
        messages = _probe_messages_matrix(state, game_key, memory,
                                          probe_round, num_rounds, no_think)
        ps = []
        for k in range(samples):
            reply = client.chat(messages, temperature=temperature,
                                max_tokens=max_tokens,
                                seed=seed * 1000 + k if samples > 1 else None)
            p = _p_from_reply(reply)
            if p is not None:
                ps.append(p)
        out[state_label(state)] = dict(
            p_cooperate=(sum(ps) / len(ps)) if ps else None,
            n_samples=len(ps),
            state=[list(pair) for pair in state],
        )
    return dict(kind="matrix", game=game_key, memory=memory,
                probe_round=probe_round, num_rounds=num_rounds,
                temperature=temperature, samples=samples,
                model=getattr(client, "model", "?"), states=out)


def _probe_messages_donors(state, *, b: float, c: float, w: float, q: float,
                           num_rounds: int, rep_text: str,
                           no_think: bool = True) -> list[dict]:
    """Conversational replay: the game dialogue that puts the model into
    `state`, with forced assistant turns, ending on the next-round ask."""
    agent, partner = st.AGENT_NAMES[0], st.AGENT_NAMES[1]
    messages = [{"role": "user", "content": dg.initial_prompt(
        b, c, w, q, num_rounds, rep_text, agent, partner, no_think)}]
    cumulative = 0.0
    for k, (own, opp) in enumerate(state):
        rnd = k + 1
        cumulative += st.round_payoff(own, opp, b, c, normalized=False)
        messages.append({"role": "assistant", "content": f"DECISION: {own}"})
        messages.append({"role": "user", "content": dg.round_result_text(
            rnd, num_rounds, own, partner, opp,
            st.round_payoff(own, opp, b, c, normalized=False),
            cumulative, "", no_think)})
    return messages


def probe_donors_policy(client, *, b: float, c: float, w: float, q: float,
                        memory: int, num_rounds: int = 12,
                        with_reputation: Optional[bool] = None,
                        temperature: float = 0.0, max_tokens: int = 512,
                        samples: int = 1, seed: int = 0,
                        no_think: bool = True) -> dict[str, Any]:
    """P(cooperate | s) for all 4^m histories under the donors framing.

    `with_reputation` forces the q-gated reputation report on/off in the
    replayed initial prompt (None = roll Bernoulli(q) with `seed`,
    strategy prior taken as tit_for_tat)."""
    rng = random.Random(seed)
    if with_reputation is None:
        rep_text = st.build_reputation_text("tit_for_tat", q, rng)
    elif with_reputation:
        rep_text = st.build_reputation_text("tit_for_tat", 1.0, rng)
    else:
        rep_text = ""
    states = enumerate_states(memory)
    out: dict[str, Any] = {}
    for state in states:
        messages = _probe_messages_donors(
            state, b=b, c=c, w=w, q=q, num_rounds=num_rounds,
            rep_text=rep_text, no_think=no_think)
        ps = []
        for k in range(samples):
            reply = client.chat(messages, temperature=temperature,
                                max_tokens=max_tokens,
                                seed=seed * 1000 + k if samples > 1 else None)
            p = _p_from_reply(reply)
            if p is not None:
                ps.append(p)
        out[state_label(state)] = dict(
            p_cooperate=(sum(ps) / len(ps)) if ps else None,
            n_samples=len(ps),
            state=[list(pair) for pair in state],
        )
    return dict(kind="donors", b=b, c=c, w=w, q=q, memory=memory,
                num_rounds=num_rounds, temperature=temperature,
                samples=samples, reputation_text=rep_text,
                model=getattr(client, "model", "?"), states=out)


def uniform_state_p(probe: dict, own: str, opp: str) -> Optional[float]:
    """P(C) in the state where every remembered round was (own, opp) —
    e.g. ('c','c') -> after sustained mutual cooperation."""
    memory = probe["memory"]
    label = "|".join(f"{own}{opp}" for _ in range(memory))
    entry = probe["states"].get(label)
    return None if entry is None else entry["p_cooperate"]


def save_probe(path: str | Path, probe: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(probe, f, indent=1)


def load_probe(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
