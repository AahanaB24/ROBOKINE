# ROBOKINE
### A Free Python-Based Universal Platform for Serial Robot Kinematics, Trajectory Planning, Workspace Analysis, Singularity Evaluation, and Machine Learning-Based Inverse Kinematics

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-success)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-orange.svg)

</p>

---

## Overview

ROBOKINE is a free, Python-based graphical software platform developed for configurable serial robot kinematics, trajectory planning, workspace analysis, singularity evaluation, and machine learning-based inverse kinematics.

Unlike traditional robotics software that requires programming knowledge, ROS configuration, or proprietary licenses, ROBOKINE provides an intuitive graphical interface where users can build and analyze custom robot manipulators with up to six degrees of freedom.

The platform enables users to:

- Configure custom serial manipulators
- Visualize robot motion in real time
- Perform Forward Kinematics (FK)
- Solve Numerical Inverse Kinematics (IK)
- Generate robot workspaces
- Evaluate singularities
- Plan trajectories through multiple via points
- Compare different trajectory planning methods
- Export animations and graphs
- Explore Machine Learning-based Inverse Kinematics using multiple regression models

ROBOKINE is intended for:

- Robotics education
- Undergraduate laboratories
- Graduate research
- Early-stage robot design
- Preliminary manipulator validation

---

# Features

| Feature | Description |
|----------|-------------|
| Configurable Robot Architecture | Supports custom robots from 1–6 DOF |
| Mixed Joint Types | Revolute, Prismatic, and Fixed joints |
| Motion Axis Selection | +X, -X, +Y, -Y, +Z, -Z |
| Link Length Configuration | Individual link dimensions |
| Joint Constraints | User-defined limits |
| Home Position Management | Save, reset, and return home |
| Forward Kinematics | Real-time pose computation |
| Numerical Inverse Kinematics | Jacobian-based IK solver |
| Machine Learning IK | Neural Network, Random Forest, and KNN |
| Workspace Generation | Monte Carlo sampling |
| Singularity Analysis | Jacobian rank, manipulability, condition number |
| Via Point Planning | Add, edit, reorder, delete via points |
| Trajectory Planning | Linear, Parabolic Blend, Cubic, Quintic |
| Real-Time Graphs | Position, Velocity, Acceleration |
| Animation | Interactive trajectory playback |
| Export | PNG, CSV, GIF, MP4 |

---

# Software Preview

<p align="center">
<img src="assets/main_gui.png" width="900">
</p>

The interface provides an integrated workflow for robot configuration, forward and inverse kinematics, workspace generation, singularity analysis, trajectory planning, and visualization.

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/ROBOKINE.git
```

```bash
cd ROBOKINE
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Launch ROBOKINE

```bash
python main.py
```

---

# Quick Start

## Step 1 — Configure the Robot

Choose:

- Number of DOF
- Joint Type
- Motion Axis
- Link Length
- Joint Limits
- Home Position

Preset robot configurations are also available for quick setup.

---

## Step 2 — Forward Kinematics

Move the joint sliders to:

- Rotate revolute joints
- Extend prismatic joints
- Visualize robot motion
- Observe end-effector movement
- View the transformation matrix

---

## Step 3 — Inverse Kinematics

Enter a desired target position.

ROBOKINE computes the required joint variables using an iterative Jacobian-based numerical inverse kinematics solver.

Features include:

- Target Position
- IK Tolerance
- Maximum Iterations
- Damping
- Joint Limit Enforcement

---

## Step 4 — Workspace Generation

Generate the reachable workspace using Monte Carlo sampling.

The workspace visualization helps verify:

- Reachable positions
- Robot limitations
- Via-point feasibility

---

## Step 5 — Singularity Analysis

Evaluate robot configurations using:

- Jacobian Rank
- Manipulability
- Singular Values
- Condition Number

This allows users to identify singular or near-singular configurations before executing trajectories.

---

## Step 6 — Trajectory Planning

ROBOKINE supports multiple trajectory planning algorithms:

- Fifth-Order Polynomial
- Third-Order Polynomial
- Second-Order Parabolic Blend

Users can:

- Add Via Points
- Edit Via Points
- Reorder Via Points
- Animate Motion
- Compare Trajectory Profiles

---

## Step 7 — Visualization

Real-time trajectory plots include:

- Joint Displacement
- Joint Velocity
- Joint Acceleration

Trajectory graphs can be downloaded for reports or publications.

---

## Step 8 — Export

ROBOKINE supports exporting:

- Robot Images
- Workspace CSV Files
- Trajectory Graphs
- GIF Animations
- MP4 Videos

---

# Software Modules

## Robot Configuration

Configure custom serial manipulators with up to six degrees of freedom.

Supported Joint Types:

- Revolute
- Prismatic
- Fixed

---

## Forward Kinematics

Compute:

- End-effector position
- Transformation matrix
- Robot pose

Real-time sliders allow immediate visualization.

---

## Numerical Inverse Kinematics

Jacobian-based iterative solver supporting:

- Mixed joint architectures
- Joint constraints
- User-defined targets

---

## Workspace Analysis

Generate reachable workspace through random sampling within user-defined joint constraints.

Export workspace data as CSV.

---

## Singularity Evaluation

Evaluate robot posture using:

- Jacobian Rank
- Manipulability Index
- Singular Values
- Condition Number

---

## Trajectory Planning

Supported methods:

| Method | Characteristics |
|----------|----------------|
| Linear | Simple interpolation |
| Parabolic Blend | Constant velocity segments |
| Cubic Polynomial | Smooth position and velocity |
| Quintic Polynomial | Smooth position, velocity, and acceleration |

---

## Animation

Visualize robot motion through planned trajectories.

Adjust:

- Playback speed
- Animation delay

Export animations as:

- GIF
- MP4

---

# Machine Learning Inverse Kinematics

ROBOKINE includes a separate Machine Learning GUI that allows users to train inverse kinematics models using forward-kinematics-generated datasets.

Supported Models

- Neural Network
- Random Forest
- K-Nearest Neighbors

Workflow:

1. Generate Dataset
2. Train Model
3. Evaluate Performance
4. Save Model
5. Predict Joint Angles

The Neural Network achieved the highest prediction accuracy during validation.

---

# Example Applications

- Robot Kinematics Education
- Undergraduate Robotics Courses
- Graduate Robotics Research
- Manipulator Prototyping
- Workspace Validation
- Motion Planning Demonstrations
- Machine Learning Research
- Robotics Laboratory Exercises

---

# Repository Structure

```
ROBOKINE/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── assets/
│
├── docs/
│   ├── installation.md
│   ├── quick_start.md
│   ├── tutorials/
│   └── screenshots/
│
├── robokine/
│   ├── gui/
│   ├── kinematics/
│   ├── trajectory/
│   ├── workspace/
│   ├── singularity/
│   ├── ml_ik/
│   ├── utils/
│   └── main.py
│
├── examples/
│
├── datasets/
│
└── exported_examples/
```

---

# Case Studies

ROBOKINE has been validated using four representative case studies.

| Case Study | Configuration | Validation |
|------------|--------------|------------|
| Case 1 | 4-DOF R-R-R-P | Fifth-order trajectory planning |
| Case 2 | 5-DOF P-R-R-R-R | Third-order trajectory planning |
| Case 3 | 6-DOF R-R-R-R-R-R | Parabolic Blend trajectory |
| Case 4 | ML-Based IK | Neural Network, Random Forest, KNN |

---

# Documentation

Additional documentation includes:

- Installation Guide
- User Manual
- Tutorial Videos
- Research Paper
- Example Projects

---

# Tutorial Videos

| Tutorial | Description |
|-----------|-------------|
| Installation | Software setup |
| Quick Start | First robot |
| Forward Kinematics | FK walkthrough |
| Inverse Kinematics | IK tutorial |
| Workspace Analysis | Reachability |
| Singularity Analysis | Jacobian evaluation |
| Trajectory Planning | Motion generation |
| Machine Learning IK | Model training |

---

# Future Development

Planned features include:

- CAD/STL Import
- Collision Detection
- Robot Dynamics
- Torque Analysis
- Path Planning (RRT, PRM, A*)
- ROS Integration
- Arduino Integration
- Raspberry Pi Support
- Jetson Support
- Cloud Version
- Web-Based GUI
- Automatic Report Generation

---

# Research

If you use ROBOKINE in your research, please cite:

```bibtex
@article{robokine2026,
  title={ROBOKINE: A Free Python-Based Universal Platform for Serial Robot Kinematics, Trajectory Planning, Workspace Analysis, Singularity Evaluation, and Machine Learning-Based Inverse Kinematics},
  author={Authors},
  journal={Under Review},
  year={2026}
}
```

---

# Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Added feature"
```

4. Push to GitHub

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# License

This project is released under the MIT License.

See the LICENSE file for details.

---

# Contact

For questions, suggestions, or collaboration opportunities, please open a GitHub Issue or contact the project maintainers.

---

## Acknowledgements

ROBOKINE was developed to provide an accessible and open-source platform for robotics education, research, and early-stage manipulator design. We thank the robotics research community and the contributors whose open-source tools and publications inspired this work.
