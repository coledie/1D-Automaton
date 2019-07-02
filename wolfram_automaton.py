"""
Fuctionality for elementary and totalistic 3 color cellular automaton.
"""
import numpy as np
from scipy import ndimage as ndi

import cv2  # Only for visualization, unnecessary w/ show=False.


def wolfram_1d(rule, init_state=None, iterations=500, show=True):
    """
    Generates 1D Wolfram automaton states over a number iterations.

    Parameters
    ----------
    rule: int [0, 255]
        Number associated with the next state rule.
    init_state: ndarray[0./1.], any size, default=[0, ... 1, ... 0]
        State to start simualation with.
    iterations: int, default=500
        Number of new states to solve for.
    show: bool, default=True
        Use opencv to display states or not.

    Returns
    -------
    ndarray[iterations+1, len(init_state] All calculated states, stacked.
    """
    def rule_map(rule):
        """
        Generate the map for the cell and its neighbors to the new cell state based
        on the given rule number.

        Parameters
        ----------
        rule: int
            Number associated with the next state rule.

        Returns
        -------
        dict, {'': 1/0 for all permutations of cell and neighbors.}
        """
        STATES = ['1', '0']
        reverse_binary = [x + y + z for x in STATES for y in STATES for z in STATES]

        rule_binary = []
        for i in range(7, -1, -1):
            rule_binary += [int(rule / 2**i >= 1)]
            rule %= 2**i

        return dict(zip(reverse_binary, rule_binary))

    state = np.zeros(shape=1001); state[state.size // 2 + 1] = 1
    if init_state is not None:
        state = init_state

    image = state.copy()

    state_map = rule_map(rule)

    RULE = lambda neighbors: state_map["".join([str(int(v)) for v in neighbors])]

    footprint = np.ones(shape=(3))

    for _ in range(iterations):
        state = ndi.generic_filter(state, RULE, footprint=footprint)

        image = np.vstack((image, state))

    if show:
        cv2.imshow(f"Wolfram 1D, Rule: {rule}", image)
        cv2.waitKey(0)

    return image


def wolfram_1d_3c(code, init_state=None, start_color=2, iterations=500, show=True):
    """
    Generates 2 color 1D Wolfram automaton states over a number iterations.

    Parameters
    ----------
    code: int [0, 255]
        Code associated with the next state rule.
    init_state: ndarray[0./1.], any size, default=[0, ... 1, ... 0]
        State to start simualation with.
    start_color: int[0, 1, 2], default=2
        If using default init state, can choose color of center value.
    iterations: int, default=500
        Number of new states to solve for.
    show: bool, default=True
        Use opencv to display states or not.

    Returns
    -------
    ndarray[iterations+1, len(init_state] All calculated states, stacked.
    """
    def code_map(code):
        """
        Generate the map for the cell and its neighbors to the new cell state based
        on the given code.

        Parameters
        ----------
        code: int
            Number associated with the next state rule.

        Returns
        -------
        dict, {'': 2/1/0 for all possible average values of cell and neighbors.}
        """
        averages = ['2.00', '1.67', '1.33', '1.00', '0.67', '0.33', '0.00']

        code_trinary = []
        for i in range(6, -1, -1):
            code_trinary += [code // 3**i]
            code %= 3**i

        return dict(zip(averages, code_trinary))

    state = np.zeros(shape=1001); state[state.size // 2 + 1] = start_color
    if init_state is not None:
        state = init_state

    image = state.copy()

    state_map = code_map(code)

    RULE = lambda neighbors: state_map[f"{sum(neighbors)/3:.2f}"]

    footprint = np.ones(shape=(3))

    for _ in range(iterations):
        state = ndi.generic_filter(state, RULE, footprint=footprint)

        image = np.vstack((image, state))

    if show:
        cv2.imshow(f"Wolfram 1D, Code: {code}", image / 2)
        cv2.waitKey(0)

    return image


if __name__ == '__main__':
    wolfram_1d(110)  ## Turing Complete Proof
    wolfram_1d(184, init_state=np.where(np.random.uniform(0, 1, size=1000) > .5, 1., 0.))
    wolfram_1d(90)  ## Serpinski Triangle
    wolfram_1d(30)  ## Chaos
    wolfram_1d(135)  ## Inverse of 30

    wolfram_1d_3c(1023)  ## Serpinski Triangle
    wolfram_1d_3c(219, start_color=1)  ## Pillars
    wolfram_1d_3c(1599, start_color=1)  ## Pattern
    wolfram_1d_3c(1041)  ## Chaos
