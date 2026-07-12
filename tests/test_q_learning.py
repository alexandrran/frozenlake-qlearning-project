import numpy as np

from frozen_lake_q import MAP_LAYOUTS, select_best_action, update_q_value


def test_standard_map_sizes():
    assert sum(len(row) for row in MAP_LAYOUTS['4x4']) == 16
    assert sum(len(row) for row in MAP_LAYOUTS['8x8']) == 64


def test_best_action_is_selected():
    q = np.array([[0.1, 0.8, 0.2, 0.3]])
    action = select_best_action(q, 0, np.random.default_rng(42))
    assert action == 1


def test_tied_best_actions_are_valid():
    q = np.array([[0.5, 0.1, 0.5, 0.2]])
    rng = np.random.default_rng(42)
    actions = {select_best_action(q, 0, rng) for _ in range(20)}
    assert actions <= {0, 2}
    assert actions == {0, 2}


def test_q_update_uses_future_reward():
    q = np.array([[0.0, 0.0], [0.5, 0.25]])
    update_q_value(q, 0, 1, 0.0, 1, False, alpha=0.1, gamma=0.9)
    assert np.isclose(q[0, 1], 0.045)


def test_terminal_q_update_does_not_bootstrap():
    q = np.array([[0.0, 0.0], [10.0, 5.0]])
    update_q_value(q, 0, 1, 1.0, 1, True, alpha=0.1, gamma=0.9)
    assert np.isclose(q[0, 1], 0.1)
