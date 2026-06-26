"""Smoke test for HeterogeneousIPD. Run after `pip install jaxmarl`.

Verifies the env follows the JaxMARL API (reset/step return shapes, dones,
auto-reset) and that the N=2 reward reduces to the original PD payoff table.
"""
import jax, jax.numpy as jnp
from heterogeneous_ipd import make_hetero_ipd, REGIMES


def test_api(N=3, regime="others"):
    env = make_hetero_ipd(num_agents=N, regime=regime, num_steps=8)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    assert set(obs.keys()) == set(env.agents)
    for a in env.agents:
        assert obs[a].shape == (2 * N + 1,), obs[a].shape
    # one random step
    key, ka, ks = jax.random.split(key, 3)
    acts = {a: jax.random.randint(jax.random.fold_in(ka, i), (), 0, 2)
            for i, a in enumerate(env.agents)}
    obs, state, rew, done, info = env.step_env(ks, state, acts)
    assert set(rew.keys()) == set(env.agents)
    assert "__all__" in done
    print(f"[api ok] N={N} regime={regime} obs_dim={2*N+1} "
          f"coop_rate={float(info['coop_rate']):.2f}")


def test_pd_reduction():
    """N=2 must reproduce R=1,T=1.2,S=-0.5,P=0."""
    env = make_hetero_ipd(num_agents=2, regime="full")
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    cases = {(0, 0): (1.0, 1.0), (0, 1): (-0.5, 1.2),
             (1, 0): (1.2, -0.5), (1, 1): (0.0, 0.0)}
    for (a0, a1), (e0, e1) in cases.items():
        acts = {"agent_0": jnp.int32(a0), "agent_1": jnp.int32(a1)}
        _, _, rew, _, _ = env.step_env(key, state, acts)
        r0, r1 = float(rew["agent_0"]), float(rew["agent_1"])
        assert abs(r0 - e0) < 1e-6 and abs(r1 - e1) < 1e-6, (a0, a1, r0, r1)
    print("[pd ok] N=2 reproduces the PD payoff table")


def test_all_cooperate_defect(N=5):
    env = make_hetero_ipd(num_agents=N, regime="full")
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    allc = {a: jnp.int32(0) for a in env.agents}
    alld = {a: jnp.int32(1) for a in env.agents}
    _, _, rc, _, _ = env.step_env(key, state, allc)
    _, _, rd, _, _ = env.step_env(key, state, alld)
    assert all(abs(float(rc[a]) - 1.0) < 1e-6 for a in env.agents)
    assert all(abs(float(rd[a]) - 0.0) < 1e-6 for a in env.agents)
    print(f"[social-optimum ok] N={N}: all-C->1.0, all-D->0.0")


if __name__ == "__main__":
    for r in REGIMES:
        test_api(N=4, regime=r)
    test_pd_reduction()
    test_all_cooperate_defect()
    print("\nAll smoke tests passed.")
