"""
Heterogeneous-observation N-player Iterated Prisoner's Dilemma as a JaxMARL
`MultiAgentEnv`.

This ports the observability experiments from the CRLD study into the
*sampled, deep-RL* regime so they can be trained with JaxMARL's IPPO/MAPPO
baselines, scaled to many agents, and run on GPU.

Game.  N agents, each round choosing C (0) or D (1).  Reward is the
*pairwise-averaged* Prisoner's Dilemma: agent i plays the 2x2 PD against every
other agent and receives the mean payoff,

    r_i = mean_{j != i} g(a_i, a_j),
    g(C,C)=R, g(C,D)=S, g(D,C)=T, g(D,D)=P.

At N=2 this is exactly the original 2-agent PD (R=1, T=1.2, S=-0.5, P=0); the
per-capita scale is independent of N, so increasing N does not by itself change
the dilemma strength -- the only thing that changes across the table is the
number of agents and their observability.

Memory-1.  Each agent conditions on the previous round's joint action, seen
through its own observation mask.  Observability regimes (an (N, N) visibility
matrix V, V[i, j] = 1 iff agent i sees agent j's last action):

    full     V = 1               (everyone sees everyone)
    blind    V = 0               (no one sees anything -> effectively memoryless)
    self     V = I               (each agent sees only its own last action)
    others   V = 1 - I           (each agent sees others but not itself)
    coop     V = I or (j cooperated)   (sees self always, others only if they cooperated)
    def      V = I or (j defected)     (sees self always, others only if they defected)

`coop`/`def` are the cooperation-/defection-tracking regimes; they are
action-conditioned, so V depends on the previous joint action. All six regimes
reduce to the corresponding 2-agent conditions used in the CRLD experiments.

Each agent's observation is a fixed-length vector (shape 2N+1) so the same
network/config works across regimes: for every agent j a 2-d one-hot of its last
action, zeroed where masked, followed by a single "episode start" flag.
"""
from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import chex
from flax import struct

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box, Discrete

REGIMES = ("full", "blind", "self", "others", "coop", "def")


@struct.dataclass
class State:
    last_actions: chex.Array  # (N,) int in {0,1}; previous round's joint action
    step: int                 # steps taken this episode
    is_start: chex.Array      # bool scalar; True before any round has been played
    done: chex.Array          # (N,) bool


class HeterogeneousIPD(MultiAgentEnv):
    """N-player memory-1 IPD with per-agent observation masks."""

    def __init__(
        self,
        num_agents: int = 2,
        regime: str = "full",
        payoffs: Tuple[float, float, float, float] = (1.0, 1.2, -0.5, 0.0),  # R,T,S,P
        num_steps: int = 128,
    ):
        super().__init__(num_agents)
        assert num_agents >= 2, "need at least 2 agents for a dilemma"
        assert regime in REGIMES, f"regime must be one of {REGIMES}"
        self.regime = regime
        self.num_steps = num_steps
        self.R, self.T, self.S, self.P = payoffs

        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.a_to_i = {a: i for i, a in enumerate(self.agents)}

        obs_dim = 2 * num_agents + 1
        self.action_spaces = {a: Discrete(2) for a in self.agents}
        self.observation_spaces = {
            a: Box(0.0, 1.0, (obs_dim,)) for a in self.agents
        }

        N = num_agents
        eye = jnp.eye(N)
        self._eye = eye
        self._off = 1.0 - eye          # off-diagonal selector
        self._ones = jnp.ones((N, N))

    # ------------------------------------------------------------------ obs
    def _visibility(self, coop: chex.Array) -> chex.Array:
        """(N, N) visibility matrix; may depend on who cooperated last round."""
        N = self.num_agents
        if self.regime == "full":
            return self._ones
        if self.regime == "blind":
            return jnp.zeros((N, N))
        if self.regime == "self":
            return self._eye
        if self.regime == "others":
            return self._off
        if self.regime == "coop":      # see self always, others iff they cooperated
            return self._eye + self._off * coop[None, :]
        if self.regime == "def":       # see self always, others iff they defected
            return self._eye + self._off * (1.0 - coop)[None, :]
        raise ValueError(self.regime)

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        last = state.last_actions                 # (N,)
        coop = (last == 0).astype(jnp.float32)    # 1 where cooperated
        vis = self._visibility(coop)              # (N_i, N_j)
        vis = jnp.where(state.is_start, jnp.zeros_like(vis), vis)
        onehot = jax.nn.one_hot(last, 2)          # (N_j, 2)
        obs = vis[:, :, None] * onehot[None, :, :]  # (N_i, N_j, 2)
        obs = obs.reshape(self.num_agents, -1)      # (N_i, 2N)
        start = jnp.broadcast_to(
            state.is_start.astype(jnp.float32), (self.num_agents, 1)
        )
        obs = jnp.concatenate([obs, start], axis=1)  # (N_i, 2N+1)
        return {a: obs[i] for i, a in enumerate(self.agents)}

    # ----------------------------------------------------------------- step
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        state = State(
            last_actions=jnp.zeros(self.num_agents, dtype=jnp.int32),
            step=0,
            is_start=jnp.array(True),
            done=jnp.zeros(self.num_agents, dtype=bool),
        )
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]):
        a = jnp.array([actions[ag] for ag in self.agents])   # (N,) in {0,1}
        coop = (a == 0).astype(jnp.float32)
        n_others = self.num_agents - 1
        c_others = coop.sum() - coop                          # cooperators among others
        rew_C = (self.R * c_others + self.S * (n_others - c_others)) / n_others
        rew_D = (self.T * c_others + self.P * (n_others - c_others)) / n_others
        rew = jnp.where(coop == 1.0, rew_C, rew_D)            # (N,)

        new_step = state.step + 1
        done = new_step >= self.num_steps
        new_state = State(
            last_actions=a.astype(jnp.int32),
            step=new_step,
            is_start=jnp.array(False),
            done=jnp.full((self.num_agents,), done),
        )
        obs = self.get_obs(new_state)
        rewards = {ag: rew[i] for i, ag in enumerate(self.agents)}
        dones = {ag: done for ag in self.agents}
        dones["__all__"] = done
        info = {"coop_rate": coop.mean()}
        return obs, new_state, rewards, dones, info

    @property
    def name(self) -> str:
        return f"HeterogeneousIPD-{self.num_agents}p-{self.regime}"


def make_hetero_ipd(num_agents=2, regime="full", **kw) -> HeterogeneousIPD:
    """Convenience factory mirroring jaxmarl.make(...)."""
    return HeterogeneousIPD(num_agents=num_agents, regime=regime, **kw)
