# n-Body Solar System Perturbation Simulator

A numerical gravitational n-body simulator implemented in Python, built for an investigation into the perturbative effects of Solar System planets on Earth's orbital eccentricity.

## Modes

### 1. Interactive Simulator (`main.py`)

A real-time interactive n-body gravitational simulator with a 3D visualisation and live Hamiltonian energy monitoring.

```bash
cd src
python3 main.py
```

Bodies, masses, and initial conditions can be configured through the UI at startup. The simulation uses an adaptive RKF45 integrator and displays trajectories and energy deviation in real time.

### 2. Perturbation Investigation (`investigate.py`)

Runs the nine simulation configurations used in the investigation, comparing Earth's trajectory against an analytical circular reference orbit and writing results to CSV files and figures.

```bash
cd src
python3 investigate.py
```

Outputs are written to `src/results/`:
- `timeseries_<config>.csv` — per-timestep orbital quantities for Earth
- `summary.csv` — max/mean phase-space distance and eccentricity per configuration
- `trajectory_<config>.png` — 2D orbital trajectory plots
- `energy_error_all.png` — Hamiltonian error across all configurations
- `phase_dist_all.png` — phase-space distance from analytical reference
- `eccentricity_all.png` — instantaneous orbital eccentricity (selected configs)
- `eccentricity_summary.png` — final eccentricity bar chart

A sensitivity analysis can also be run:

```bash
python3 sensitivity.py
```

Outputs are written to `src/results/sensitivity/`.

## Requirements

```
Python 3.13+
numpy
matplotlib
```

Install dependencies:

```bash
pip install numpy matplotlib
```

## Units

All simulations use astronomical units (AU), solar masses (M☉), and years (yr), with G = 4π² AU³ yr⁻² M☉⁻¹.

## Project Structure

```
src/
├── main.py              # Interactive simulator
├── investigate.py       # Perturbation investigation script
├── sensitivity.py       # Sensitivity analysis script
├── data_types.py        # Vector3, Body dataclasses
├── utils.py             # State vector packing/unpacking
├── physics/
│   ├── dynamics.py      # NBodyDynamics — derivatives, energies
│   └── system.py        # NBodySystem — mass array, derived properties
├── integration/
│   └── rk45.py          # AdaptiveRKF45Integrator
├── simulation/
│   ├── runner.py        # SimulationRunner
│   ├── frame.py         # SimulationFrame dataclass
│   ├── history.py       # SimulationHistory deque
│   └── statistics.py    # SimulationStatistics diagnostics
├── ui/                  # Interactive UI (main.py only)
├── visualisation/       # Real-time plotting (main.py only)
└── results/             # Output directory (generated on run)
```