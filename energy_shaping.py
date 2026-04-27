import numpy as np

from scipy.integrate import solve_ivp

from dynamic import DEFAULT_PARAMS, desired_upright_energy, dynamics, total_energy


def angle_error(angle, target):
    return (angle - target + np.pi) % (2.0 * np.pi) - np.pi


def energy_shaping_control(x, k=2.0, params=DEFAULT_PARAMS):
    x = np.asarray(x, dtype=float)
    q1 = x[0]
    q2 = x[1]
    q2_dot = x[3]

    energy = total_energy(x, params)
    energy_error_value = energy - desired_upright_energy(params)

    posture_weight = 1.0 / (1.0 + energy_error_value**2)
    posture_term = posture_weight * (
        -8.0 * angle_error(q2, 0.0)
        -3.0 * q2_dot
        -4.0 * angle_error(q1, np.pi)
    )

    u = -k * energy_error_value * q2_dot + posture_term
    u = np.clip(u, -params.u_max, params.u_max)
    return float(u)


def simulate_energy_shaping(x0, tf=10.0, dt=0.02, k=2.0, params=DEFAULT_PARAMS):
    num_steps = int(np.floor(tf / dt)) + 1
    t_eval = np.linspace(0.0, tf, num_steps)

    def rhs(t, x):
        u = energy_shaping_control(x, k=k, params=params)
        return dynamics(t, x, u, params)

    sol = solve_ivp(rhs, (0.0, tf), x0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
    if not sol.success:
        raise RuntimeError(sol.message)

    X = sol.y.T
    U = np.array([energy_shaping_control(x, k=k, params=params) for x in X])
    E = np.array([total_energy(x, params) for x in X])
    return sol.t, X, U, E


def _self_check():
    x0 = np.array([0.0, 0.0, 0.0, 0.1], dtype=float)
    t, X, U, E = simulate_energy_shaping(x0, tf=5.0, dt=0.02, k=2.0)

    print("initial state:", X[0])
    print("final state:", X[-1])
    print("initial energy:", E[0])
    print("final energy:", E[-1])
    print("max |u|:", np.max(np.abs(U)))
    print("num samples:", len(t))


if __name__ == "__main__":
    _self_check()
