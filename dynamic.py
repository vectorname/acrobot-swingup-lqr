from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AcrobotParams:
    """Physical parameters of the Acrobot."""

    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    lc1: float = 0.5
    lc2: float = 0.5
    I1: float = 1.25
    I2: float = 1.25
    g: float = 9.81
    u_max: float = 10.0


DEFAULT_PARAMS = AcrobotParams()

def acrobot_matrices(
    x: np.ndarray,
    params: AcrobotParams = DEFAULT_PARAMS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return M(q), C(q, qdot), G(q), and B.

    The returned terms satisfy:

        M(q) qddot + C(q, qdot) + G(q) = B u

    C is returned as a 2-vector containing Coriolis and centrifugal terms,
    not as a matrix.
    """

    q1, q2, q1_dot, q2_dot = np.asarray(x, dtype=float)

    m1, m2 = params.m1, params.m2
    l1, lc1, lc2 = params.l1, params.lc1, params.lc2
    I1, I2, g = params.I1, params.I2, params.g

    c2 = np.cos(q2)
    s2 = np.sin(q2)

    M11 = I1 + I2 + m2 * l1**2 + 2.0 * m2 * l1 * lc2 * c2
    M12 = I2 + m2 * l1 * lc2 * c2
    M22 = I2
    M = np.array([[M11, M12], [M12, M22]], dtype=float)

    C1 = -2.0 * m2 * l1 * lc2 * s2 * q1_dot * q2_dot - m2 * l1 * lc2 * s2 * q2_dot**2
    C2 = m2 * l1 * lc2 * s2 * q1_dot**2
    C = np.array([C1, C2], dtype=float)

    G1 = (m1 * lc1 + m2 * l1) * g * np.sin(q1) + m2 * lc2 * g * np.sin(q1 + q2)
    G2 = m2 * lc2 * g * np.sin(q1 + q2)
    G = np.array([G1, G2], dtype=float)

    B = np.array([0.0, 1.0], dtype=float)
    return M, C, G, B


def qddot(
    x: np.ndarray,
    u: float,
    params: AcrobotParams = DEFAULT_PARAMS,
) -> np.ndarray:
    """Compute joint accelerations for state x and input torque u."""

    u = float(np.clip(u, -params.u_max, params.u_max))
    M, C, G, B = acrobot_matrices(x, params)
    return np.linalg.solve(M, B * u - C - G)


def dynamics(
    t: float,
    x: np.ndarray,
    u: float = 0.0,
    params: AcrobotParams = DEFAULT_PARAMS,
) -> np.ndarray:
    """Continuous-time state equation xdot = f(t, x, u)."""

    del t
    x = np.asarray(x, dtype=float)
    q1_ddot, q2_ddot = qddot(x, u, params)
    return np.array([x[2], x[3], q1_ddot, q2_ddot], dtype=float)


def kinetic_energy(
    x: np.ndarray,
    params: AcrobotParams = DEFAULT_PARAMS,
) -> float:
    """Return kinetic energy."""

    x = np.asarray(x, dtype=float)
    q_dot = x[2:4]
    M, _, _, _ = acrobot_matrices(x, params)
    return float(0.5 * q_dot @ M @ q_dot)


def potential_energy(
    x: np.ndarray,
    params: AcrobotParams = DEFAULT_PARAMS,
) -> float:
    """Return potential energy with the hanging-down state as zero."""

    q1, q2 = np.asarray(x, dtype=float)[:2]

    y1 = -params.lc1 * np.cos(q1)
    y2 = -params.l1 * np.cos(q1) - params.lc2 * np.cos(q1 + q2)

    y1_bottom = -params.lc1
    y2_bottom = -params.l1 - params.lc2

    V1 = params.m1 * params.g * (y1 - y1_bottom)
    V2 = params.m2 * params.g * (y2 - y2_bottom)
    return float(V1 + V2)


def total_energy(
    x: np.ndarray,
    params: AcrobotParams = DEFAULT_PARAMS,
) -> float:
    """Return total mechanical energy."""

    return kinetic_energy(x, params) + potential_energy(x, params)


def desired_upright_energy(params: AcrobotParams = DEFAULT_PARAMS) -> float:
    """Return the total energy of the upright equilibrium."""

    x_top = np.array([np.pi, 0.0, 0.0, 0.0], dtype=float)
    return total_energy(x_top, params)


def link_positions(
    x: np.ndarray,
    params: AcrobotParams = DEFAULT_PARAMS,
) -> np.ndarray:
    """Return Cartesian positions of base, elbow, and tip.

    The output has shape (3, 2):
        [[base_x, base_y], [elbow_x, elbow_y], [tip_x, tip_y]]
    """

    q1, q2 = np.asarray(x, dtype=float)[:2]

    base = np.array([0.0, 0.0])
    elbow = np.array([
        params.l1 * np.sin(q1),
        -params.l1 * np.cos(q1),
    ])
    tip = elbow + np.array([
        params.l2 * np.sin(q1 + q2),
        -params.l2 * np.cos(q1 + q2),
    ])

    return np.vstack((base, elbow, tip))


def _self_check() -> None:
    """Run lightweight consistency checks when this file is executed."""

    bottom = np.array([0.0, 0.0, 0.0, 0.0])
    top = np.array([np.pi, 0.0, 0.0, 0.0], dtype=float)

    print("bottom energy:", total_energy(bottom))
    print("top energy:", total_energy(top))
    print("bottom positions:")
    print(link_positions(bottom))
    print("top positions:")
    print(link_positions(top))
    print("xdot at bottom, u=0:", dynamics(0.0, bottom, 0.0))
    print("xdot at top, u=0:", dynamics(0.0, top, 0.0))


if __name__ == "__main__":
    _self_check()
