# Project Overview

This project predicts SME Financial Health Index from structured tabular data across Eswatini, Lesotho, Malawi, and Zimbabwe.

## Business framing

The model aims to capture broader business resilience rather than pure revenue. Features therefore combine:

- savings and informal finance signals
- formal financial service access
- debt and repayment proxy behavior
- resilience and perceived future risk
- business scale and age

## Engineering principles

- All transformations are code-based and reproducible
- Raw data is preserved in `data/raw`
- Feature engineering is centralized in `src/features/engineering.py`
- Training and inference share the same engineered feature logic
- Outputs are written into `outputs/` and `artifacts/` for easy reuse

## F1 optimization choices

- Stratified cross-validation for stable multiclass estimates
- Class-weighted models where appropriate
- Native categorical boosting models included for mixed-type data
- Ensemble comparison against the best single model
- Per-class reporting to watch rare-class degradation
