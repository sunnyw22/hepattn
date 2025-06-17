import pytest
import time

import numpy as np
import scipy
import lap1015


def solve_scipy(cost):
    _, col_idx = scipy.optimize.linear_sum_assignment(cost.T)
    return col_idx


def solve_1015_early(cost):
    return lap1015.lap_late(cost.T)


def solve_1015_late(cost):
    return lap1015.lap_late(cost.T)


solvers = {
    "scipy": solve_scipy,
    "1015_early": solve_1015_early,
    "1015_late": solve_1015_late,
}


def match_all(solver_name, cost, pad_mask=None):
    return solvers[solver_name](cost)


def match_padded(solver_name, cost, pad_mask):
    n_objects = np.sum(pad_mask)
    C = cost[:, :n_objects]
    pred_idx = solvers[solver_name](C)
    if solver_name == "scipy":
        default_idx = set(range(cost.shape[1]))
        full_col_idx = np.empty(cost.shape[1], dtype=int)
        full_col_idx[:n_objects] = pred_idx
        full_col_idx[n_objects:] = np.array(list(default_idx - set(pred_idx)), dtype=int)
        return full_col_idx
    return pred_idx


@pytest.mark.parametrize("solver_name", ["scipy", "1015_early", "1015_late"])
@pytest.mark.parametrize("scale", [1.0, 10.0, 100.0, 1000.0])
def test_matcher_equivalence(solver_name, scale: float):
    for _ in range(200):
        n_objects = np.random.randint(100, 250)
        n_valid_objects = np.random.randint(n_objects // 4, n_objects // 2)
        cost = np.random.rand(n_objects, n_objects) * scale
        pad_mask = np.zeros(n_objects, dtype=bool)
        pad_mask[:n_valid_objects] = True
        cost[~np.broadcast_to(pad_mask.reshape(1, -1), cost.shape)] = 1e8  # Set padding costs to infinity

        idx_all = match_all(solver_name, cost)
        idx_padded = match_padded(solver_name, cost, pad_mask)
        assert np.all(idx_all[:n_valid_objects] == idx_padded[:n_valid_objects]), (
            f"Solver {solver_name} mismatch for n_objects={n_objects}, n_valid_objects={n_valid_objects}"
        )
        assert len(set(idx_all)) == n_objects, (
            f"Solver {solver_name} produced duplicate indices for n_objects={n_objects}, n_valid_objects={n_valid_objects}"
        )
        assert len(set(idx_padded)) == n_objects, (
            f"Solver {solver_name} produced duplicate indices for padded n_objects={n_objects}, n_valid_objects={n_valid_objects}"
        )


@pytest.mark.parametrize("solver_name", ["scipy", "1015_early", "1015_late"])
def test_matcher_speed(solver_name):
    times_all = []
    times_padded = []
    for _ in range(1024):
        n_objects = np.random.randint(100, 250)
        n_valid_objects = np.random.randint(n_objects // 4, n_objects // 2)
        cost = np.random.rand(n_objects, n_objects) * 100
        pad_mask = np.zeros(n_objects, dtype=bool)
        pad_mask[:n_valid_objects] = True
        cost[~np.broadcast_to(pad_mask.reshape(1, -1), cost.shape)] = 1e8  # Set padding costs to infinity
        start_time = time.perf_counter()
        _ = match_all(solver_name, cost)
        times_all.append(time.perf_counter() - start_time)
        start_time = time.perf_counter()
        _ = match_padded(solver_name, cost, pad_mask)
        times_padded.append(time.perf_counter() - start_time)
    avg_time_all = np.mean(times_all)
    avg_time_padded = np.mean(times_padded)
    print(f"Solver: {solver_name}, Avg Time All: {avg_time_all:.6f}s, Avg Time Padded: {avg_time_padded:.6f}s")
