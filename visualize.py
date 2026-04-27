import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

from dynamic import DEFAULT_PARAMS, dynamics, link_positions, total_energy


def simulate_acrobot(x0, tf=10.0, dt=0.02, u=0.0, params=DEFAULT_PARAMS):
    t_eval = np.arange(0.0, tf + dt, dt)

    def rhs(t, x):
        return dynamics(t, x, u, params)

    sol = solve_ivp(rhs, (0.0, tf), x0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.t, sol.y.T


def plot_states(t, X):
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(t, X[:, 0], label="q1")
    axes[0].plot(t, X[:, 1], label="q2")
    axes[0].set_ylabel("angle [rad]")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, X[:, 2], label="q1_dot")
    axes[1].plot(t, X[:, 3], label="q2_dot")
    axes[1].set_ylabel("angular velocity [rad/s]")
    axes[1].legend()
    axes[1].grid(True)

    energy = np.array([total_energy(x) for x in X])
    axes[2].plot(t, energy, label="total energy")
    axes[2].set_ylabel("energy [J]")
    axes[2].set_xlabel("time [s]")
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()
    return fig, axes


def plot_control_result(t, X, U, mode=None):
    energy = np.array([total_energy(x) for x in X])
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    axes[0].plot(t, X[:, 0], label="q1")
    axes[0].plot(t, X[:, 1], label="q2")
    axes[0].set_ylabel("angle [rad]")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, X[:, 2], label="q1_dot")
    axes[1].plot(t, X[:, 3], label="q2_dot")
    axes[1].set_ylabel("angular velocity [rad/s]")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t, U, label="u")
    axes[2].set_ylabel("torque [Nm]")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(t, energy, label="total energy")
    if mode is not None:
        mode = np.asarray(mode, dtype=float)
        axes[3].step(t, mode * np.max(energy), where="post", label="mode")
    axes[3].set_ylabel("energy")
    axes[3].set_xlabel("time [s]")
    axes[3].legend()
    axes[3].grid(True)

    fig.tight_layout()
    return fig, axes


def animate_acrobot(t, X, params=DEFAULT_PARAMS):
    fig, ax = plt.subplots(figsize=(6, 6))
    limit = params.l1 + params.l2 + 0.2

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Acrobot")

    line, = ax.plot([], [], "o-", lw=3)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text("")
        return line, time_text

    def update(frame):
        pts = link_positions(X[frame], params)
        line.set_data(pts[:, 0], pts[:, 1])
        time_text.set_text(f"t = {t[frame]:.2f} s")
        return line, time_text

    interval_ms = max(1, int(1000 * (t[1] - t[0]))) if len(t) > 1 else 20
    ani = FuncAnimation(
        fig,
        update,
        frames=len(t),
        init_func=init,
        interval=interval_ms,
        blit=True,
        repeat=True,
    )
    return fig, ani


def animate_controlled_acrobot(t, X, U, mode, params=DEFAULT_PARAMS):
    fig, ax = plt.subplots(figsize=(6, 6))
    limit = params.l1 + params.l2 + 0.2

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Acrobot Control")

    line, = ax.plot([], [], "o-", lw=3)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    mode_text = ax.text(0.02, 0.89, "", transform=ax.transAxes)
    torque_text = ax.text(0.02, 0.83, "", transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text("")
        mode_text.set_text("")
        torque_text.set_text("")
        return line, time_text, mode_text, torque_text

    def update(frame):
        pts = link_positions(X[frame], params)
        line.set_data(pts[:, 0], pts[:, 1])
        time_text.set_text(f"t = {t[frame]:.2f} s")
        mode_text.set_text("mode = LQR" if mode[frame] else "mode = energy shaping")
        torque_text.set_text(f"u = {U[frame]:.2f} Nm")
        return line, time_text, mode_text, torque_text

    interval_ms = max(1, int(1000 * (t[1] - t[0]))) if len(t) > 1 else 20
    ani = FuncAnimation(
        fig,
        update,
        frames=len(t),
        init_func=init,
        interval=interval_ms,
        blit=True,
        repeat=True,
    )
    return fig, ani


def run_acrobot_demo():
    x0 = np.array([0.2, -0.1, 0.0, 0.0], dtype=float)
    t, X = simulate_acrobot(x0, tf=10.0, dt=0.02, u=0.0)

    plot_states(t, X)
    _, ani = animate_acrobot(t, X)
    plt.show()
    return ani


if __name__ == "__main__":
    run_acrobot_demo()
