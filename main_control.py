import numpy as np
import matplotlib.pyplot as plt

from dynamic import DEFAULT_PARAMS, desired_upright_energy, total_energy
from energy_shaping import angle_error, simulate_energy_shaping
from lqr import design_acrobot_lqr, simulate_lqr
from visualize import animate_controlled_acrobot, plot_control_result


def should_switch_to_lqr(
    x,
    params=DEFAULT_PARAMS,
    energy_tol=1.0,
    q1_tol=0.10,
    q2_tol=0.20,
    vel_tol=1.0,
):
    x = np.asarray(x, dtype=float)

    q1_err = angle_error(x[0], np.pi)
    q2_err = angle_error(x[1], 0.0)
    q1_dot = x[2]
    q2_dot = x[3]
    energy_err = abs(total_energy(x, params) - desired_upright_energy(params))

    return (
        q1_err < 0.0
        and abs(q1_err) < q1_tol
        and abs(q2_err) < q2_tol
        and abs(q1_dot) < vel_tol
        and abs(q2_dot) < vel_tol
        and energy_err < energy_tol
    )


def simulate_hybrid_control(
    x0,
    tf=12.0,
    dt=0.01,
    k_energy=2.0,
    params=DEFAULT_PARAMS,
):
    t_energy, X_energy, U_energy, _ = simulate_energy_shaping(
        x0,
        tf=tf,
        dt=dt,
        k=k_energy,
        params=params,
    )

    switch_index = None
    for i, x in enumerate(X_energy):
        if should_switch_to_lqr(x, params=params):
            switch_index = i
            break

    if switch_index is None:
        mode = np.zeros(len(t_energy), dtype=int)
        return t_energy, X_energy, U_energy, mode

    t_switch = t_energy[switch_index]
    x_switch = X_energy[switch_index]
    remaining_time = tf - t_switch

    K, _, _, _, x_eq = design_acrobot_lqr(params=params)
    t_lqr, X_lqr, U_lqr = simulate_lqr(
        x_switch,
        K,
        x_eq,
        tf=remaining_time,
        dt=dt,
        params=params,
    )

    t = np.concatenate((t_energy[:switch_index], t_switch + t_lqr))
    X = np.vstack((X_energy[:switch_index], X_lqr))
    U = np.concatenate((U_energy[:switch_index], U_lqr))

    mode_energy = np.zeros(switch_index, dtype=int)
    mode_lqr = np.ones(len(t_lqr), dtype=int)
    mode = np.concatenate((mode_energy, mode_lqr))

    return t, X, U, mode


def run_main_control():
    x0 = np.array([0.0, 0.0, 0.0, 0.1], dtype=float)
    t, X, U, mode = simulate_hybrid_control(x0, tf=12.0, dt=0.01, k_energy=2.0)

    plot_control_result(t, X, U, mode)
    _, ani = animate_controlled_acrobot(t, X, U, mode)
    plt.show()

    final_state = X[-1]
    print("final state:", final_state)
    print("final energy:", total_energy(final_state))
    print("LQR activated:", np.any(mode == 1))
    return t, X, U, mode, ani


if __name__ == "__main__":
    run_main_control()
