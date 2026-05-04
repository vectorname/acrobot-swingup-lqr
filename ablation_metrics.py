import numpy as np

from energy_shaping import angle_error
from main_control import simulate_hybrid_control


def compute_metrics(t, X, U, mode):
    q1_err = angle_error(X[:, 0], np.pi)
    q2_err = angle_error(X[:, 1], 0.0)
    q1_dot = X[:, 2]
    q2_dot = X[:, 3]

    max_u = float(np.max(np.abs(U)))
    lqr_activated = bool(np.any(mode == 1))
    switch_time = float(t[np.argmax(mode)]) if lqr_activated else None

    inside_target = (
        (np.abs(q1_err) < 0.05)
        & (np.abs(q2_err) < 0.05)
        & (np.abs(q1_dot) < 0.10)
        & (np.abs(q2_dot) < 0.10)
    )

    settling_time = None
    for i in range(len(t)):
        if np.all(inside_target[i:]):
            settling_time = float(t[i])
            break

    final_error = float(
        np.linalg.norm([q1_err[-1], q2_err[-1], q1_dot[-1], q2_dot[-1]])
    )

    return {
        "lqr_activated": lqr_activated,
        "switch_time_s": switch_time,
        "settling_time_s": settling_time,
        "max_abs_torque_nm": max_u,
        "final_q1_error_rad": float(q1_err[-1]),
        "final_q2_error_rad": float(q2_err[-1]),
        "final_q1_dot_rad_s": float(q1_dot[-1]),
        "final_q2_dot_rad_s": float(q2_dot[-1]),
        "final_error_norm": final_error,
    }


def main():
    x0 = np.array([0.0, 0.0, 0.0, 0.1], dtype=float)
    t, X, U, mode = simulate_hybrid_control(x0, tf=12.0, dt=0.01, k_energy=2.0)
    metrics = compute_metrics(t, X, U, mode)

    print("Ablation metrics for current LQR setting")
    print("---------------------------------------")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
