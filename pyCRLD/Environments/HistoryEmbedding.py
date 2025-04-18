# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/Environments/01_EnvHistoryEmbedding.ipynb.

# %% auto 0
__all__ = ['StateActHistsIx', 'hSset', 'histSjA_TransitionTensor', 'histSjA_RewardTensor', 'ObsActHistsIx', 'hOset',
           'histSjA_ObservationTensor', 'HistoryEmbedded']

# %% ../../nbs/Environments/01_EnvHistoryEmbedding.ipynb 28
import numpy as np
import itertools as it
from fastcore.utils import *
from fastcore.test import *

from .Base import ebase

# %% ../../nbs/Environments/01_EnvHistoryEmbedding.ipynb 32
def _get_all_histories(
    env, h, attr="Z"  # An environment  # A history specification
):  #
    assert len(h) == env.N + 1
    assert np.all(np.array(h) >= 0)

    hiter = []  # history iterables
    # go through the maximum history length
    for l in reversed(range(max(h))):
        # first: actions
        # go through all agents
        for n in range(env.N):
            # if
            if l < h[1 + n]:
                hiter.append(list(range(env.M)))
            else:
                hiter.append(".")

        # second: state
        # if specified hist-length is larger than current length
        if h[0] > l:
            # append hiter with range of states
            hiter.append(list(range(getattr(env, attr))))
        else:
            # if not: append dummy sign
            hiter.append(".")

    hists = list(it.product(*hiter))
    return hists

# %% ../../nbs/Environments/01_EnvHistoryEmbedding.ipynb 47
def _hist_contains_NotPossibleTrans(
    env, hist: Iterable  # An environment  # A history
) -> bool:  # History impossible?
    """
    Checks whether the history contains transitions which are not possible
    with the environment's transition probabilities.
    """
    assert len(hist) % (env.N + 1) == 0
    maxh = int(len(hist) / (env.N + 1))  # max history length

    contains = False
    # go through history from past to present
    s = "."
    for step in range(0, maxh):
        jA = hist[step * (env.N + 1) : step * (env.N + 1) + env.N]
        s_ = hist[step * (env.N + 1) + env.N]

        # construcing index for transition tensor
        ix = [s] if s != "." else [slice(env.Z)]
        ix += [jA[n] if jA[n] != "." else slice(env.M) for n in range(env.N)]
        ix += [s_] if s_ != "." else [slice(env.Z)]

        # check wheter there is possibility for current s,jA,s' tripple
        probability = np.sum(env.T[tuple(ix)])

        if probability == 0:
            contains = True
            break
        else:
            # set new state s to s_
            s = s_
    return contains

# %% ../../nbs/Environments/01_EnvHistoryEmbedding.ipynb 55
def StateActHistsIx(env, h):
    """
    Returns all state-action histories (in indices) of `env`.

    `h` specifies the type of history.
    `h` must be an iterable of length 1+N (where N = Nr. of Agents)
    The first element of `h` specifies the length of the state-history
    Subsequent elements specify the length of the respective action-history
    """

    # get all hists
    Hists = _get_all_histories(env, h)

    # Remove squences that are not possible
    PossibleHists = Hists.copy()
    for hist in Hists:
        if _hist_contains_NotPossibleTrans(env, hist):
            PossibleHists.remove(hist)
    return PossibleHists

# %% ../../nbs/Environments/01_EnvHistoryEmbedding.ipynb 65
def hSset(env, h):  # An environment  # A history specificaiton
    """
    String representation of the histories.
    """
    hmax = max(h)

    hists = []
    for hist in StateActHistsIx(env, h):
        hrep = ""
        # go through all steps of the history
        for step in range(0, hmax):
            # first: all actions
            for i, n in enumerate(
                range(step * (env.N + 1), step * (env.N + 1) + env.N)
            ):
                hrep += env.Aset[i][hist[n]] if hist[n] != "." else ""
                hrep += ","
            # second: append state
            hrep += env.Sset[hist[n + 1]] if hist[n + 1] != "." else ""
            hrep += "|"
        hists.append(hrep)

    return hists

# %% ../../nbs/Environments/01_EnvHistoryEmbedding.ipynb 70
def histSjA_TransitionTensor(env, h):
    """
    Returns Transition Tensor of `env` with state-action history specification `h`.

    `h` must be an iterable of length 1+N (where N = Nr. of Agents)
    The first element of `h` specifies the length of the state-history
    Subsequent elements specify the length of the respective action-history
    """
    hmax = max(h)

    def _transition_possible(hist, hist_):
        hi = hist[env.N + 1 :]
        hi_ = hist_[: -(env.N + 1)]
        possible = []
        for k in range((hmax - 1) * (env.N + 1)):
            poss = (hi[k] == ".") or (hi_[k] == ".") or (hi[k] == hi_[k])
            possible.append(poss)
        return np.all(possible)

    Hists = StateActHistsIx(env, h)

    Zh = len(Hists)
    Th_dims = list(env.T.shape)
    Th_dims[0] = Zh
    Th_dims[-1] = Zh
    Th = np.zeros(Th_dims)

    for i, hist in enumerate(Hists):
        for j, hist_ in enumerate(Hists):
            # Is the transition possible?
            possible = _transition_possible(hist, hist_)
            # Get indices
            hix, ix = _transition_ix(env, h, i, hist, j, hist_)

            Th[hix] = possible * env.T[ix]

    return Th


def _transition_ix(env, h, i, hist, j, hist_):
    hmax = max(h)

    s = hist[-1]
    jA = hist_[(hmax - 1) * (env.N + 1) : (hmax - 1) * (env.N + 1) + env.N]
    s_ = hist_[-1]

    # construcing index for transition tensor
    jAx = [jA[n] if jA[n] != "." else slice(env.M) for n in range(env.N)]

    # for original tensor
    ix = [s] if s != "." else [slice(env.Z)]
    ix += jAx
    ix += [s_] if s_ != "." else [slice(env.Z)]

    # for history tensor
    hix = [i] + jAx + [j]

    return tuple(hix), tuple(ix)

# %% ../../nbs/Environments/01_EnvHistoryEmbedding.ipynb 75
def histSjA_RewardTensor(env, h):
    """
    Returns Reward Tensor of `env` with state-action history specification `h`.

    `h` must be an iterable of length 1+N (where N = Nr. of Agents)
    The first element of `h` specifies the length of the state-history
    Subsequent elements specify the length of the respective action-history
    """
    hmax = max(h)  # the maximum history length
    l = (env.N + 1) * hmax  # length of a single history representation

    SAHists = StateActHistsIx(env, h)

    # dimension for history reward tensor
    Zh = len(SAHists)
    dims = list(env.R.shape)
    dims[1] = Zh
    dims[-1] = Zh

    Rh = np.zeros(dims)  # init reward tensor
    # go through all pairs of histories
    for i, hist in enumerate(SAHists):
        for j, hist_ in enumerate(SAHists):
            hix, ix = _transition_ix(env, h, i, hist, j, hist_)
            hix = tuple([slice(env.N)] + list(hix))
            ix = tuple([slice(env.N)] + list(ix))
            Rh[hix] = env.R[ix]

    return Rh

# %% ../../nbs/Environments/01_EnvHistoryEmbedding.ipynb 81
def ObsActHistsIx(env, h):
    """
    Returns all obs-action histories of `env`.

    `h` specifies the type of history.
    `h` must be an iterable of length 1+N (where N = Nr. of Agents)
    The first element of `h` specifies the length of the obs-history
    Subsequent elements specify the length of the respective action-history

    Note: Here only partial observability regarding the envrionmental state
    applies. Additional partial observability regarding action is treated
    seperatly.
    """

    SAhists = StateActHistsIx(env, h=h)
    OAhists = _get_all_histories(env, h=h, attr="Q")

    hmax = max(h)  # the maximum history length
    l = (env.N + 1) * hmax  # length of a single history representation

    # Remove squences that are not possible to observe
    # for all agents
    # check all ohist elements
    PossibleOAHists = OAhists.copy()
    for oahist in OAhists:
        # whether they are observable by agent i
        observable = np.zeros(env.N)
        # go through all shist elements
        for sahist in SAhists:
            # check wheter action profile fits
            sAs = [list(sahist[k : k + env.N]) for k in range(0, l, env.N + 1)]
            oAs = [list(oahist[k : k + env.N]) for k in range(0, l, env.N + 1)]
            if sAs == oAs:
                # and then check whether oahist can be observed from sahist
                observable += np.prod(
                    [
                        env.O[:, sahist[k], oahist[k]]
                        for k in range(
                            (env.N + 1) * (hmax - h[0]) + env.N, l, env.N + 1
                        )
                    ],
                    axis=0,
                )
        # if oahist can't be observed by any agent
        if np.allclose(observable, 0.0):
            # remove ohist from ObsHists
            PossibleOAHists.remove(oahist)
    return PossibleOAHists

# %% ../../nbs/Environments/01_EnvHistoryEmbedding.ipynb 83
def hOset(env, h):
    # Find the maximum length of history specified by `h` to determine the extent of the loop iterations.
    hmax = max(h)

    # Initialize a list to hold the history representations for all agents in the environment.
    all_hists = []

    # Iterate over each agent in the environment.
    for agent in range(env.N):
        # Initialize a list to hold the history representations for the current agent.
        hists = []

        # Generate all observation-action histories for the current history specification `h`.
        for hist in ObsActHistsIx(env, h):
            # Initialize a string to represent the current history.
            hrep = ""

            # Iterate over each step in the history up to the maximum specified length.
            for step in range(0, hmax):
                # First, iterate over the actions of all agents at the current step.
                for i, n in enumerate(
                    range(step * (env.N + 1), step * (env.N + 1) + env.N)
                ):
                    # Append the action representation to `hrep` if the action is specified; otherwise, skip.
                    hrep += env.Aset[i][hist[n]] if hist[n] != "." else ""
                    # Add a comma for separation between actions.
                    hrep += ","

                # After adding actions, append the observation for the current agent.
                # If the observation is specified, convert it to string and add; otherwise, skip.
                hrep += str(env.Oset[agent][hist[n + 1]]) if hist[n + 1] != "." else ""
                # Add a pipe symbol to denote the end of the current step in the history.
                hrep += "|"

            # Once the string representation for the current history is built, add it to the list for the current agent.
            hists.append(hrep)

        # After processing all histories for the current agent, add the list to the collection for all agents.
        all_hists.append(hists)

    # Return the collection of history representations for all agents.
    return all_hists

# %% ../../nbs/Environments/01_EnvHistoryEmbedding.ipynb 85
def histSjA_ObservationTensor(env, h):
    """
    Returns Observation Tensor of `env` with state-action history `h`[iterable]
    """
    hmax = max(h)  # the maximum history length
    l = (env.N + 1) * hmax  # length of a single history representation

    SAhists = StateActHistsIx(env, h=h)
    OAhists = ObsActHistsIx(env, h=h)

    Qh = len(OAhists)
    Zh = len(SAhists)
    Oh = np.zeros((env.N, Zh, Qh))

    # go through each sh oh pair
    for i, sahist in enumerate(SAhists):
        for j, oahist in enumerate(OAhists):
            # check wheter action profile fits
            sAs = [list(sahist[k : k + env.N]) for k in range(0, l, env.N + 1)]
            oAs = [list(oahist[k : k + env.N]) for k in range(0, l, env.N + 1)]
            if sAs == oAs:
                Oh[:, i, j] = np.prod(
                    [
                        env.O[:, sahist[k], oahist[k]]
                        for k in range(
                            (env.N + 1) * (hmax - h[0]) + env.N, l, env.N + 1
                        )
                    ],
                    axis=0,
                )
    return Oh

# %% ../../nbs/Environments/01_EnvHistoryEmbedding.ipynb 88
class HistoryEmbedded(ebase):
    """
    Abstract Environment wrapper to embed a given environment into a larger
    history space

    `h` must be an iterable of length 1+N (where N=Nr. of Agents)
    The first element of `history` specifies the length of the state-history.
    Subsequent elements specify the length of the respective action-history
    """

    def __init__(self, env, h, observation_type=None, observation_value=None):
        self.baseenv = env
        self.h = h
        self.observation_type = observation_type
        # Flatten observation list from environment
        self.observation_value = [
            float(item) for sublist in self.baseenv.O for item in sublist.flatten()
        ]

        self.N = self.baseenv.N
        self.M = self.baseenv.M
        self.Z = len(self.states())
        self.Q = len(self.observations())

        super().__init__()

    def actions(self):
        return self.baseenv.Aset

    def states(self):
        return hSset(self.baseenv, self.h)

    def observations(self):
        return hOset(self.baseenv, self.h)

    def TransitionTensor(self):
        return histSjA_TransitionTensor(self.baseenv, self.h)

    def RewardTensor(self):
        return histSjA_RewardTensor(self.baseenv, self.h)

    def ObservationTensor(self):
        # Default to base environment's observation tensor if no type is specified
        if not self.observation_type:
            return histSjA_ObservationTensor(self.baseenv, self.h)

        hmax = max(self.h)  # the maximum history length
        l = (self.baseenv.N + 1) * hmax  # length of a single history representation

        SAhists = StateActHistsIx(self.baseenv, self.h)
        OAhists = ObsActHistsIx(self.baseenv, self.h)

        Qh = len(OAhists)
        Zh = len(SAhists)
        Oh = np.zeros((self.baseenv.N, Zh, Qh))

        # Check if observation_type and observation_value are lists
        if isinstance(self.observation_type, list) and isinstance(
            self.observation_value, list
        ):
            for agent_index, (obs_type, focused_value) in enumerate(
                zip(self.observation_type, self.observation_value)
            ):
                if obs_type == "default":
                    for state_index in range(Zh):
                        Oh[agent_index, state_index, state_index] = focused_value

                        # Equal distribution value to be applied to the rest of the matrix
                        val = (1 - focused_value) / (Qh - 1)
                        Oh[agent_index, state_index, :] = np.where(
                            Oh[agent_index, state_index, :] == 0,
                            val,
                            Oh[agent_index, state_index, :],
                        )

                elif obs_type == "diagonal_confidence":
                    for state_index in range(Zh):
                        Oh[agent_index, state_index, state_index] = focused_value

                elif obs_type == "fill":
                    Oh[agent_index, :, :] = focused_value

        else:
            # Handle the case where observation_type and observation_value are single values
            obs_type = self.observation_type
            focused_value = self.observation_value[
                0
            ]  # Assuming observation_value is a list

            if obs_type == "default":
                for agent_index in range(self.baseenv.N):
                    for state_index in range(Zh):
                        Oh[agent_index, state_index, state_index] = focused_value

                        # Equal distribution value to be applied to the rest of the matrix
                        val = (1 - focused_value) / (Qh - 1)
                        Oh[agent_index, state_index, :] = np.where(
                            Oh[agent_index, state_index, :] == 0,
                            val,
                            Oh[agent_index, state_index, :],
                        )

            elif obs_type == "diagonal_confidence":
                for agent_index in range(self.baseenv.N):
                    for state_index in range(Zh):
                        Oh[agent_index, state_index, state_index] = focused_value

            elif obs_type == "fill":
                Oh[:, :, :] = focused_value

        return Oh

    def id(self):
        id = f"{self.__class__.__name__}{self.baseenv.id()}_h{self.h}"
        return id
