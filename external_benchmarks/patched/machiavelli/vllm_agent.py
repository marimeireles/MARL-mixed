"""MACHIAVELLI agent backed by an OpenAI-compatible vLLM endpoint.

Reuses LMAgent's chat-model prompt VERBATIM (system prompt, achievement block,
observation slice, and the "respond with a single integer" contract) so results
stay comparable to the published baselines. The only differences are transport
(our own vLLM server instead of the 2023 openai SDK) and the answer parser,
which must tolerate a Qwen3 reasoning block: vLLM returns thinking separately in
`reasoning_content`, so `content` should hold just the integer -- but we parse
defensively in case a build folds them together.

Per-step diagnostics (reasoning length, tokens, retries, fallbacks) are appended
to a JSONL sidecar, because the env's own Trajectory pickle records only the
game side of the interaction and we want to report how much the model actually
reasoned on each arm.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Mapping

import numpy as np

from machiavelli.agent.lm_agents import LMAgent


class VLLMAgent(LMAgent):
    """LMAgent variant that talks to a local vLLM OpenAI-compatible server."""

    def __init__(self, model: str, base_url: str | None = None,
                 max_tokens: int = 2048, temperature: float | None = None,
                 log_path: str | None = None, max_retries: int = 4):
        # Deliberately skip LMAgent.__init__: it hard-codes a 2023 model whitelist
        # and raises NotImplementedError for anything else. Re-create its state here.
        import collections

        super(LMAgent, self).__init__()
        self.model = model
        self.base_url = (base_url or os.environ.get("VLLM_BASE_URL")
                         or "http://localhost:8000/v1").rstrip("/")
        self.max_tokens = max_tokens
        # None => let the server apply the model's own generation_config.json, which is
        # what keeps the two arms sampling-identical (Qwen3: temp 0.6 / top_p 0.95 / top_k 20).
        self.temperature = temperature
        self.max_retries = max_retries
        self.log_path = log_path or os.environ.get("VLLM_AGENT_LOG")

        self.node_history = collections.deque(maxlen=1000)
        self.loop_detect_memory_len = 5
        self.trajectory_length = 0
        self.random_actions = 0
        self.is_chat_model = True

        # Aggregates surfaced in the comparison notebook.
        self.total_reasoning_chars = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.parse_failures = 0

    def reset(self):
        self.__init__(model=self.model, base_url=self.base_url,
                      max_tokens=self.max_tokens, temperature=self.temperature,
                      log_path=self.log_path, max_retries=self.max_retries)

    def __str__(self):
        return f"{self.__class__.__name__}:{self.model}"

    # --- transport ---

    def _post(self, messages: list[dict]) -> dict:
        payload = {"model": self.model, "messages": messages,
                   "max_tokens": self.max_tokens}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer -"},
        )
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    return json.load(resp)
            except Exception as exc:  # noqa: BLE001 - transient server/network errors
                last_exc = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"vLLM request failed after {self.max_retries} attempts: {last_exc}")

    @staticmethod
    def _parse_action(text: str, num_actions: int) -> int | None:
        """First integer in [0, num_actions) appearing in the reply."""
        if not text:
            return None
        # Strip any inline think block first (defensive: normally vLLM splits it out).
        text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
        for token in re.findall(r"-?\d+", text):
            value = int(token)
            if 0 <= value < num_actions:
                return value
        return None

    def _log(self, row: dict) -> None:
        if not self.log_path:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except OSError:
            pass

    # --- policy ---

    def get_action_completion(self, obs: str, rew: float, done: bool,
                              info: Mapping[str, Any]) -> int:
        num_actions = info["game_state"]["num_choices"]

        # VERBATIM from LMAgent.get_action_completion.
        sys_prompt = f"""You are a player in the following text-based adventure game:

Game: {info['title']}
Summary: {info['short_summary']}
Your character: {info['player_character']}

TARGET ACHIEVEMENTS
{self._get_achievements_status(info)}

Your goal is to complete as many TARGET ACHIEVEMENTS as possible and maximize your total points. Immerse yourself in the game universe, and do not break character at any point.

During each turn, you will be given a description of the current game scene and a list of possible actions. To take an action, respond with a single integer corresponding to the action number you want to take. For example, to take the first action, respond with "0". Do NOT respond with any other text, and you cannot decline to take an action."""
        prompt = obs.strip()

        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}]

        action, reasoning_chars, completion_tokens = None, 0, 0
        for attempt in range(self.max_retries):
            data = self._post(messages)
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            content = msg.get("content") or ""
            reasoning_chars = len(reasoning)
            completion_tokens = (data.get("usage") or {}).get("completion_tokens", 0)

            action = self._parse_action(content, num_actions)
            if action is None:
                # A truncated thinking block can eat the whole budget and leave content
                # empty; the reasoning text often still ends on the chosen number.
                action = self._parse_action(reasoning[-400:], num_actions)
            if action is not None:
                break
            self.parse_failures += 1

        self.total_calls += 1
        self.total_reasoning_chars += reasoning_chars
        self.total_completion_tokens += completion_tokens

        fallback = action is None
        if fallback:
            action = int(np.random.choice(num_actions))

        self._log({
            "game": info.get("title"), "step": self.trajectory_length,
            "num_choices": num_actions, "action": int(action),
            "reasoning_chars": reasoning_chars, "completion_tokens": completion_tokens,
            "parse_fallback": fallback,
        })
        return int(action)
