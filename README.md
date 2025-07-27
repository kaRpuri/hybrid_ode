# Hybrid Neural ODE Vehicle Dynamics

This repository implements a JAX/Flax-based Hybrid Neural ODE pipeline for learning and evaluating vehicle dynamics models. It supports efficient training, validation, and visualization of multi-step trajectory predictions.

## Workflow
1. **Prepare Data**
   - Place raw NPZ data files in the `data/` directory.
   - Run `data_processing.py` to process and split data into train/val/test sets.

2. **Train Model**
   - Edit `config.yaml` to set training parameters.
   - Run `train.py` to train the Hybrid ODE model. Model parameters are saved to `results/model_params.pkl`.

3. **Evaluate Model**
   - Run `test_model.py` to evaluate the trained model on test data. Results are saved in `evaluation_results/`.

4. **Visualize Results**
   - Run `visualize_results.py` to generate trajectory and metrics plots in `evaluation_plots/`.

## Main Files
- `models.py` — HybridODE model and RK4 integration
- `train_new.py` — Training loop and config handling
- `test_model.py` — Model evaluation and metrics
- `visualize_results.py` — Visualization of results
- `data_processing.py` — Data loading, normalization, and sample creation
- `config.yaml` — All training and model parameters


## Quick Start
```bash
# 1. Process data
python data_processing.py

# 2. Train model
python train_new.py

# 3. Evaluate model
python test_model.py

# 4. Visualize results
python visualize_results.py
```

## TODO Checklist
- [x] Make 3D jnp arrays
- [x] GPU data preloading
- [x] fix yaw angle
- [x] fix yaw error
- [x] Larger model/ batch size
- [ ] tune training parameters
- [ ] evaluate model on test_data
- [ ] write inference
