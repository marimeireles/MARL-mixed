"""Chat clients for the dynamics harnesses.

`VLLMChatClient` is an OpenAI-compatible client (stdlib urllib only),
adapted from donorSim `metastability_eval/harness/vllm_client.py`
(branch neurips-methodology). Its key feature is the continuous order
parameter `p_cooperate`: top-k logprobs at the token following the last
`DECISION:` marker, with the probability mass renormalised over
COOPERATE-prefixed vs DEFECT-prefixed candidates. A binary action hides
weak restoring/repelling tendencies in the dynamics; this readout does
not change decoding, it just reads the distribution the sampler drew from.

`MockLLMClient` implements the same interface with a scripted, memory-
aware stochastic policy. It parses the visible history out of the actual
prompts (both harness formats), so the full pipeline — game loop, probe,
plots — runs offline exactly as it would against a served model.
"""
from __future__ import annotations

import json
import math
import random
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional


class LLMClientError(RuntimeError):
    pass


class VLLMChatClient:
    def __init__(self, base_url: str, model: str, *, api_key: str = "dummy",
                 timeout: float = 600.0, max_retries: int = 4,
                 enable_thinking: bool = False):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_thinking = enable_thinking
        self._supports_template_kwargs = True

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.6,
             max_tokens: int = 512, seed: Optional[int] = None,
             top_logprobs: int = 20) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logprobs": True,
            "top_logprobs": top_logprobs,
        }
        if self._supports_template_kwargs:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        if seed is not None:
            payload["seed"] = seed
        if temperature == 0:
            payload["top_p"] = 1.0

        body = self._post(payload)
        choice = body["choices"][0]
        msg = choice.get("message", {}) or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        usage = body.get("usage", {}) or {}
        p_coop, logit_gap, chosen_lp = decision_probability(choice.get("logprobs"))
        return {
            "content": content,
            "reasoning_content": reasoning,
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "p_cooperate": p_coop,
            "decision_logit_gap": logit_gap,
            "logprob_chosen_action": chosen_lp,
        }

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.api_key}"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                # Some OpenAI-compatible servers reject vLLM's
                # chat_template_kwargs with a 400 — drop it and retry.
                if e.code == 400 and "chat_template_kwargs" in payload:
                    payload = {k: v for k, v in payload.items()
                               if k != "chat_template_kwargs"}
                    self._supports_template_kwargs = False
                    continue
                last_err = e
            except (urllib.error.URLError, OSError, ValueError) as e:
                last_err = e
            time.sleep(min(2.0 * (2 ** attempt), 30.0))
        raise LLMClientError(f"chat failed after {self.max_retries} attempts: {last_err}")

    def wait_until_ready(self, poll_s: float = 10.0, max_wait_s: float = 3600.0) -> None:
        deadline = time.time() + max_wait_s
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"{self.base_url}/v1/models", timeout=10) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            time.sleep(poll_s)
        raise LLMClientError(f"server at {self.base_url} not ready after {max_wait_s}s")


def _norm_tok(tok: str) -> str:
    return tok.replace("Ġ", " ").strip().upper()


def decision_probability(logprobs_obj: Any):
    """(p_cooperate, logit_gap, logprob_chosen) at the token after the last
    'DECISION:' marker; (None, None, None) if it cannot be located."""
    if not logprobs_obj:
        return (None, None, None)
    toks = logprobs_obj.get("content") if isinstance(logprobs_obj, dict) else None
    if not toks:
        return (None, None, None)

    running = ""
    marker_idx: Optional[int] = None
    for i, entry in enumerate(toks):
        running += entry.get("token", "").replace("Ġ", " ")
        if running.rstrip().upper().endswith("DECISION:"):
            marker_idx = i + 1
    if marker_idx is None or marker_idx >= len(toks):
        return (None, None, None)

    entry = toks[marker_idx]
    cands = entry.get("top_logprobs") or []
    if not cands:
        cands = [{"token": entry.get("token", ""), "logprob": entry.get("logprob", 0.0)}]

    p_coop_mass = 0.0
    p_def_mass = 0.0
    for cand in cands:
        t = _norm_tok(cand.get("token", ""))
        if not t:
            continue
        p = math.exp(cand.get("logprob", -1e9))
        if "COOPERATE".startswith(t):
            p_coop_mass += p
        elif "DEFECT".startswith(t):
            p_def_mass += p
    total = p_coop_mass + p_def_mass
    if total <= 0:
        return (None, None, entry.get("logprob"))
    eps = 1e-12
    return (p_coop_mass / total,
            math.log(max(p_coop_mass, eps)) - math.log(max(p_def_mass, eps)),
            entry.get("logprob"))


# ── Mock client ───────────────────────────────────────────────────────────

# A mock policy maps the visible history (list of (own, opp) action pairs,
# oldest -> newest, as rendered in the prompt) to P(cooperate).
MockPolicy = Callable[[list[tuple[str, str]]], float]


def _p_last_opp(history, p_empty, p_after_c, p_after_d) -> float:
    if not history:
        return p_empty
    return p_after_c if history[-1][1] == "COOPERATE" else p_after_d


MOCK_POLICIES: dict[str, MockPolicy] = {
    # A base model that leans selfish and barely reciprocates.
    "base-selfish": lambda h: _p_last_opp(h, 0.4, 0.5, 0.15),
    # A post-RL model that opens nice and reciprocates sharply — the
    # qualitative change the training is claimed to produce.
    "trained-reciprocal": lambda h: _p_last_opp(h, 0.85, 0.95, 0.2),
    "soft-tft": lambda h: _p_last_opp(h, 0.8, 0.9, 0.15),
    "allc": lambda h: 0.97,
    "alld": lambda h: 0.03,
    "coin": lambda h: 0.5,
    "wsls": lambda h: 0.9 if (not h or h[-1][0] == h[-1][1]) else 0.1,
    "grim-soft": lambda h: 0.05 if any(o == "DEFECT" for _, o in h) else 0.9,
}

# Matches both harness history renderings:
#   donors (conversational): "Round 3 result: you played COOPERATE, Bob played DEFECT."
#   matrix (stateless):      "- Round 3: you played COOPERATE, Bob played DEFECT."
_PAIR_RE = re.compile(
    r"you played (COOPERATE|DEFECT),\s+\w+ played (COOPERATE|DEFECT)", re.IGNORECASE)
_SWAP_MARKER = "has moved on and will not play with you again"


def parse_visible_history(messages: list[dict[str, str]]) -> list[tuple[str, str]]:
    """Reconstruct the (own, opp) history a model could read off the prompt.
    A partner swap wipes the history, as the game semantics dictate."""
    history: list[tuple[str, str]] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        for m in _PAIR_RE.finditer(content):
            history.append((m.group(1).upper(), m.group(2).upper()))
        if _SWAP_MARKER in content:
            history = []
    return history


class MockLLMClient:
    """Scripted stand-in for a served model; same .chat() contract."""

    def __init__(self, policy: str | MockPolicy = "soft-tft", seed: int = 0):
        self.policy: MockPolicy = (
            MOCK_POLICIES[policy] if isinstance(policy, str) else policy)
        self.policy_name = policy if isinstance(policy, str) else "custom"
        self.rng = random.Random(seed)
        self.model = f"mock:{self.policy_name}"

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.6,
             max_tokens: int = 512, seed: Optional[int] = None,
             top_logprobs: int = 20) -> dict[str, Any]:
        history = parse_visible_history(messages)
        p = min(max(self.policy(history), 0.0), 1.0)
        rng = random.Random(seed) if seed is not None else self.rng
        action = "COOPERATE" if rng.random() < p else "DEFECT"
        eps = 1e-12
        return {
            "content": f"DECISION: {action}",
            "reasoning_content": "",
            "finish_reason": "stop",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "p_cooperate": p,
            "decision_logit_gap": math.log(max(p, eps)) - math.log(max(1 - p, eps)),
            "logprob_chosen_action": math.log(max(p if action == "COOPERATE" else 1 - p, eps)),
        }

    def wait_until_ready(self, **kwargs) -> None:
        return None


def make_client(api_base: Optional[str], model: Optional[str],
                mock_policy: Optional[str] = None, *, api_key: str = "dummy",
                enable_thinking: bool = False, seed: int = 0):
    """Build a real client if api_base is given, else a mock."""
    if api_base:
        if not model:
            raise ValueError("--model is required with --api-base")
        return VLLMChatClient(api_base, model, api_key=api_key,
                              enable_thinking=enable_thinking)
    return MockLLMClient(mock_policy or "soft-tft", seed=seed)
