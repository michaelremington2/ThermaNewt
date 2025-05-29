# ThermaNewt

**ThermaNewt** is a Python module designed to simulate ectotherm thermoregulation using microhabitat temperature preferences and biophysical heat transfer equations. The model can be used to predict body temperature trajectories and microhabitat switching behavior over time, based on environmental conditions and organismal thermal preferences.

This tool is especially useful in eco-physiological and behavioral simulations of reptiles (e.g., snakes) in fluctuating thermal landscapes.

## Features

- 2-state thermal simulation using Newton’s Law of Cooling
- Microhabitat switching behavior based on:
  - Random flipping
  - Preferred temperature (Topt)-based probability
  - Thermal boundary limits
- Models both k-based and h-based cooling equations
- Debounce logic to ignore noise in microhabitat decisions
- Surface area estimation for more realistic energy transfer modeling

## Installation

Clone the repository:

```bash
git clone https://github.com/michaelremington2/ThermaNewt.git
cd ThermaNewt
````

(Optional) Create a conda environment:

```bash
conda env create -f environment.yaml
conda activate therma_env
```

## Quick Start

```python
from ThermaNewt.sim_snake_tb import ThermalSimulator

# Initialize the simulator
sim = ThermalSimulator(flip_logic="preferred", t_pref_min=22, t_pref_max=35, t_pref_opt=28)

# Run simulation
burrow_vector = [24, 23, 25, 24, 23]
open_vector = [30, 31, 29, 32, 33]
initial_tb = 27
k = 0.03
dt = 1  # time step in hours

usage, tb_series = sim.tb_simulator_2_state_model_wrapper(
    k=k,
    t_initial=initial_tb,
    delta_t=dt,
    burrow_temp_vector=burrow_vector,
    open_temp_vector=open_vector,
    return_tbody_sim=True
)

print(tb_series)
print(usage)
```

## Behavior Logic Options

Set `flip_logic` to one of the following:

* `"random"` — random microhabitat changes
* `"preferred"` — based on distance from Topt
* `"boundary"` — switch when beyond Tpref\_min or Tpref\_max

## Project Structure

```
ThermaNewt/
│
├── src/sim_snake_tb.py       # Main thermal simulation module
├── environment.yaml      # Optional: Conda environment file
├── README.md             # This file
└── examples/             # (Optional) Example Jupyter notebooks or CSVs
```

## Applications

* Modeling snake microhabitat use under changing climates
* Exploring thermal performance trade-offs
* Testing behavioral thermoregulation strategies in ectotherms
* Integrating with agent-based models for more complex simulations

## License

MIT License. See `LICENSE` for details.

## Contributing

Pull requests and issues are welcome. Please open a discussion or issue if you want to extend this model or clarify its usage.

```

---

Let me know if you’d like to add a usage example with real data or include citation information for a manuscript.
```
