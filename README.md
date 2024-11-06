# Fingerprint for Large Language Models

## Table of Contents
- [Data](#data)
- [dpo_tuning_models](#dpo_tuning_models)
- [figs](#figs)
- [instruction_tuning_models](#instruction_tuning_models)
- [logs](#logs)
- [Scripts](#scripts)
  - [download.py](#downloadpy)
  - [dpo_finetuning.py](#dpo_finetuningpy)
  - [evaluation.py](#evaluationpy)
  - [fig_plot.py](#fig_plotpy)
  - [finetuning.py](#finetuningpy)
  - [generation.py](#generationpy)
  - [metrics.py](#metricspy)
  - [model_list.py](#model_listpy)
  - [pre_experiments.py](#pre_experimentspy)
  - [test_prompt.py](#test_promptpy)
  - [trigger_optimize.py](#trigger_optimizepy)
  - [trigger.py](#triggerpy)
  - [utils.py](#utilspy)
- [Result](#result)

## Data
Contains datasets used for training and evaluation.

seed_trigger_set: contains 600 prompt.

trajectory_set: record LLM's output towards the seed trigger set. It contains 'tokens', 'prompt', 'output', 'token probs', 'mean_entropy', 'entropy'.

final_trigger_set: We use the optimization method to obtain a subset of the seed trigger set.

## dpo_tuning_models
Contains model configurations and checkpoints for DPO tuning.

## figs
Folder for generated figures.

## instruction_tuning_models
Contains configurations for instruction tuning.

## logs
Logs for training and evaluation processes.

## Scripts
### download.py
Script to download required datasets and model weights.

### dpo_finetuning.py
Script for fine-tuning models using DPO.

### evaluation.py
Code to evaluate model performance.

### fig_plot.py
Generates plots for results.

### finetuning.py
General fine-tuning script for different configurations.

### generation.py
Script to handle text generation tasks.

### metrics.py
Implements evaluation metrics.

### model_list.py
Lists available models and configurations.

### pre_experiments.py
Code for preliminary experiments.

### test_prompt.py
Tests prompt effectiveness for fine-tuned models.

### trigger_optimize.py
Optimization code for finding best trigger prompts.

### trigger.py
Code related to trigger-based methods.

### utils.py
Utility functions for common tasks.

## Result
Contains output images and visualizations, like result.png.

