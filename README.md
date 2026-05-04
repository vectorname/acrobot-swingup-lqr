# Acrobot Swing-up and Balance Control  

![example picture](picture/image1.png)


This repository contains a pure Python implementation for an underactuated robotics course:

- Acrobot dynamics modeling
- energy shaping swing-up control
- LQR stabilization around the upright equilibrium
- switching from swing-up to balance
- visualization, state plots, control input plots, and animation

The official website of the course：https://underactuated.mit.edu/   


## Environment

Recommended Python version:

```text
Python 3.10 
```

Create and activate a virtual environment:(recommend conda)  
activate the environment   

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Enter the project directory:

```bash
cd underactuated_project
```

Run the full Acrobot control demo:

```bash
python main_control.py
```

This will:

- initialize the Acrobot near the hanging-down configuration
- use energy shaping to pump energy into the system
- switch to LQR near the upright equilibrium
- display state plots, control input plots, and animation

## Control Pipeline

### 1. Dynamics

The Acrobot is modeled in the standard manipulator form:

```text
M(q) qddot + C(q, qdot) + G(q) = B u
```

State definition:

```text
x = [q1, q2, q1_dot, q2_dot]
```

Angle convention:

- `q1 = 0`: first link points straight down
- `q2 = 0`: second link is aligned with the first link
- upright equilibrium: `[pi, 0, 0, 0]`

### 2. Energy Shaping

The swing-up controller uses the total mechanical energy:

```text
E = T + V
```

The controller pumps energy until the Acrobot approaches the desired upright energy. A small posture-shaping term is also included to help bring the configuration closer to the upright pose before switching.

### 3. LQR

Near the upright equilibrium, the nonlinear dynamics are numerically linearized:

```text
xdot = A (x - x*) + B (u - u*)
```

An LQR controller is designed from the linearized model to stabilize the system locally.

### 4. Switching

The controller switches from energy shaping to LQR when the state is close enough to the upright equilibrium in:

- angle error
- angular velocity
- energy error

