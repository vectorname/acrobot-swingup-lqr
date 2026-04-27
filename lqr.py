import numpy as np

from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

from dynamic import DEFAULT_PARAMS, dynamics


def linearize_acrobot(x_eq, u_eq=0.0, eps=1e-6, params=DEFAULT_PARAMS):
    n = len(x_eq)
    A = np.zeros((n, n), dtype=float)
    B = np.zeros((n, 1), dtype=float)

    for i in range(n):
        dx = np.zeros(n, dtype=float)
        dx[i] = eps
        f_plus = dynamics(0.0, x_eq + dx, u_eq, params)
        f_minus = dynamics(0.0, x_eq - dx, u_eq, params)
        A[:, i] = (f_plus - f_minus) / (2.0 * eps)

    f_plus = dynamics(0.0, x_eq, u_eq + eps, params)
    f_minus = dynamics(0.0, x_eq, u_eq - eps, params)
    B[:, 0] = (f_plus - f_minus) / (2.0 * eps)

    return A, B


def lqr_gain(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K, P


def lqr_control(x, K, x_eq, params=DEFAULT_PARAMS):
    dx = np.asarray(x, dtype=float) - np.asarray(x_eq, dtype=float)
    dx[0] = (dx[0] + np.pi) % (2.0 * np.pi) - np.pi
    dx[1] = (dx[1] + np.pi) % (2.0 * np.pi) - np.pi

    u = -K @ dx
    u = np.clip(u.item(), -params.u_max, params.u_max)
    return float(u)


def simulate_lqr(x0, K, x_eq, tf=5.0, dt=0.01, params=DEFAULT_PARAMS):
    num_steps = int(np.floor(tf / dt)) + 1
    t_eval = np.linspace(0.0, tf, num_steps)

    def rhs(t, x):
        u = lqr_control(x, K, x_eq, params)
        return dynamics(t, x, u, params)

    sol = solve_ivp(rhs, (0.0, tf), x0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
    if not sol.success:
        raise RuntimeError(sol.message)

    X = sol.y.T
    U = np.array([lqr_control(x, K, x_eq, params) for x in X])
    return sol.t, X, U


def design_acrobot_lqr(
    Q=None,
    R=None,
    params=DEFAULT_PARAMS,
):
    x_eq = np.array([np.pi, 0.0, 0.0, 0.0], dtype=float)
    if Q is None:
        Q = np.diag([100.0, 100.0, 10.0, 10.0])
    if R is None:
        R = np.array([[1.0]], dtype=float)

    A, B = linearize_acrobot(x_eq, u_eq=0.0, params=params)
    K, P = lqr_gain(A, B, Q, R)
    return K, P, A, B, x_eq


def _self_check():
    K, P, A, B, x_eq = design_acrobot_lqr()
    x0 = np.array([np.pi + 0.03, -0.02, 0.0, 0.0], dtype=float)
    t, X, U = simulate_lqr(x0, K, x_eq, tf=5.0, dt=0.01)

    print("A shape:", A.shape)
    print("B shape:", B.shape)
    print("K:", K)
    print("P shape:", P.shape)
    print("initial state:", X[0])
    print("final state:", X[-1])
    print("max |u|:", np.max(np.abs(U)))
    print("num samples:", len(t))


if __name__ == "__main__":
    _self_check()
