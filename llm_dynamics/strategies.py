"""Fixed opponent strategies, donors-game payoffs, HKB phase, parsing.

Vendored from donorSim `donor_sim/game_helpers.py` on branch
neurips-methodology (merge-base e665f87) so that the LLM opponents here
behave exactly like the ones the GRPO training pipeline used. Pure
stdlib, no torch/jax. If donorSim's game_helpers changes, re-sync.
"""
from __future__ import annotations

import math
import random
import re
from typing import Optional

COOPERATE = "COOPERATE"
DEFECT = "DEFECT"

AGENT_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve",
    "Frank", "Grace", "Henry", "Iris", "Jack",
]

# Training-time distribution over fixed strategies (donorSim OPPONENT_POOL).
OPPONENT_POOL: list[tuple[str, float]] = [
    ("always_cooperate", 0.15),
    ("always_defect", 0.30),
    ("tit_for_tat", 0.20),
    ("tit_for_two_tats", 0.20),
    ("random", 0.10),
    ("grim_trigger", 0.05),
]

STRATEGIES = [name for name, _ in OPPONENT_POOL]


def opponent_first_move(strategy: str, rng: random.Random) -> str:
    if strategy == "always_defect":
        return DEFECT
    if strategy == "random":
        return COOPERATE if rng.random() < 0.5 else DEFECT
    return COOPERATE


def opponent_response(strategy: str, other_history: list[str], rng: random.Random) -> str:
    """Next move of a fixed strategy given the OTHER player's full history."""
    if strategy == "always_cooperate":
        return COOPERATE
    if strategy == "always_defect":
        return DEFECT
    if strategy == "random":
        return COOPERATE if rng.random() < 0.5 else DEFECT
    if strategy == "tit_for_tat":
        return other_history[-1] if other_history else COOPERATE
    if strategy == "tit_for_two_tats":
        if len(other_history) < 2:
            return COOPERATE
        if other_history[-1] == DEFECT and other_history[-2] == DEFECT:
            return DEFECT
        return COOPERATE
    if strategy == "grim_trigger":
        return DEFECT if DEFECT in other_history else COOPERATE
    return COOPERATE


def round_payoff(model_action: str, opponent_action: str, b: float, c: float,
                 normalized: bool = True) -> float:
    """Donors-game per-round payoff; normalized=True is paper Eq. 2, the
    value that drove the GRPO gradient: (pi + c) / (b + c) in [0, 1]."""
    own_kept = 0.0 if model_action == COOPERATE else c
    from_partner = b if opponent_action == COOPERATE else 0.0
    raw = own_kept + from_partner
    if normalized:
        denom = b + c if (b + c) > 1e-9 else 1.0
        return raw / denom
    return raw


def nowak_thresholds(b: float, c: float, w: float, q: float) -> dict:
    """Nowak's reciprocity conditions: direct favored iff w > c/b,
    indirect favored iff q > c/b."""
    cb = c / b if b > 0 else float("inf")
    return dict(c_over_b=cb,
                direct_reciprocity_favored=w > cb,
                indirect_reciprocity_favored=q > cb)


# ── HKB relative phase (Term 2 of the training reward) ────────────────────

def hkb_phi_from_window(window) -> float:
    """phi = pi*(1-mean(s))/2 from a window of s_t = a_model*a_opp in {+-1};
    pi/2 on an empty window (metastable initial state)."""
    items = list(window)
    if not items:
        return math.pi / 2.0
    s_bar = sum(items) / len(items)
    return math.pi * (1.0 - s_bar) / 2.0


def hkb_turn_reward(phi: float, q: float, b: float, c: float,
                    lambda_tom: float = 0.1, delta_omega: float = 0.0) -> float:
    """r_2 = lambda_tom * [dw*phi + (4q)*cos(phi) + (c/b)*cos(2phi)].
    Phase transition at q = c/b (Nowak's indirect-reciprocity threshold)."""
    if b <= 0:
        return 0.0
    return lambda_tom * (
        delta_omega * phi + 4.0 * q * math.cos(phi) + (c / b) * math.cos(2.0 * phi)
    )


# ── Reputation / prompt fragments (mirror the training prompts) ───────────

def strategy_reputation_rate(strategy: str) -> float:
    return {
        "always_cooperate": 1.0,
        "always_defect": 0.0,
        "tit_for_tat": 0.85,
        "tit_for_two_tats": 0.90,
        "grim_trigger": 0.70,
        "random": 0.5,
    }.get(strategy, 0.5)


def build_reputation_text(strategy: str, q: float, rng: random.Random) -> str:
    """With probability q, a natural-language reputation report; else ''."""
    if q <= 0 or rng.random() >= q:
        return ""
    coop_rate = strategy_reputation_rate(strategy)
    observed = max(0.0, min(1.0, coop_rate + rng.gauss(0, 0.1)))
    n_obs = rng.randint(3, 10)
    n_coop = round(observed * n_obs)
    return (
        f"Reputation report on your partner: in {n_obs} prior interactions "
        f"with other players, they cooperated {n_coop}/{n_obs} times "
        f"({observed*100:.0f}%)."
    )


def payoff_description(b: float, c: float) -> str:
    if c < b:
        return (
            f"Note: cooperation costs you {c:.1f} points (lost defection payoff) "
            f"but gives your partner {b:.1f} points. Since {b:.1f} > {c:.1f}, "
            f"mutual cooperation is socially beneficial in iterated play."
        )
    if c > b:
        return (
            f"Note: cooperation costs you {c:.1f} points (lost defection payoff) "
            f"but gives your partner only {b:.1f} points. Since {c:.1f} > {b:.1f}, "
            f"the cost of cooperating exceeds the benefit to your partner — "
            f"cooperation is net-negative even in repeated play."
        )
    return f"Note: cooperation cost ({c:.1f}) equals partner benefit ({b:.1f})."


def w_description(w: float) -> str:
    if w >= 0.99:
        return "You will encounter your partner again every round (guaranteed re-encounter)."
    if w <= 0.01:
        return ("After this round you will be paired with a random new player; "
                "you will likely never see this partner again.")
    return (f"After this round there is a {w*100:.0f}% chance you will be paired "
            f"with this same partner again next round.")


def q_description(q: float) -> str:
    if q >= 0.99:
        return "Your partner's full history of past actions with everyone is publicly known to you."
    if q <= 0.01:
        return "You have no information about your partner's reputation or past actions with others."
    return (f"There is a {q*100:.0f}% chance you receive a reputation report on "
            f"your partner's past behavior with other players.")


# ── Decision parsing ──────────────────────────────────────────────────────

_DECISION_RE = re.compile(r"DECISION\s*:\s*(COOPERATE|DEFECT)", re.IGNORECASE)


def parse_decision(text: str) -> Optional[str]:
    """Parse the LAST 'DECISION: COOPERATE|DEFECT' (models rehearse the
    format before committing). None if unparseable."""
    if not isinstance(text, str):
        return None
    matches = _DECISION_RE.findall(text)
    if matches:
        return matches[-1].upper()
    return None
