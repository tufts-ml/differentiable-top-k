# Perturbed Top-K Optimization

Here we present the Perturbed Top-K Optimization objective with extensions to the synthetic data experiments in the IMLH 2023 paper [Learning where to intervene with a differentiable top-K operator:
Towards data-driven strategies to prevent fatal opioid overdoses]().

## Requirements:
Numpy, Tensorflow 2.0+, [perturbations.py from this repository](https://github.com/google-research/google-research/tree/master/perturbations)

## Code overview
- `Perturbed Top-K Demonstration.ipynb` A notebook demonstrating the creation of synthetic data and model training
- `create_data.py` Generates and saves the synthetic data
- `create_model.py` Creates a tensorflow model that can utilize the Perturbed-Top-K module, or without
- `data_utils.py` Miscellaneous utilities for generating and visualizing the synthetic data