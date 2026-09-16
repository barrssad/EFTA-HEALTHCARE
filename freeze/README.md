# Part 10 Scientific Freeze

This directory is the additive protocol-freeze package for the EFTA healthcare XAI research project.

The freeze records the canonical protocol configuration, preregistration, environment, dataset and source manifests, AI-use log, persistent split references, and checksum manifest. It contains no final confirmatory research results. The existing two-seed smoke artifacts remain under `../ml/results/smoke/` and are explicitly labelled `execution_mode: smoke`.

The frozen runtime source of truth remains `ml/config.py` and the existing implementation modules. `config.yaml` is the serialized, reviewable freeze record and does not override runtime code.
