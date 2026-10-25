# Sample Data Directory

This directory contains the bundled reproducible synthetic retail campaign dataset for EconoCausal:

* **`synthetic_retail_campaign.csv`**: 2,500 synthetic customer records with documented non-linear confounding, generated via `src/data/generator.py` (seed 42).
* **`DATASET_METADATA.md`**: Detailed technical specification of the Data Generating Process (DGP), logit formulations, parameter weights, and variable dictionary.

For general instructions on data schemas, custom dataset placement, and generation instructions, see [`data/README.md`](../data/README.md).
