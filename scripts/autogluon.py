#!/usr/bin/env python
# coding: utf-8

from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import numpy as np

trial = 3

# load train data
train_data = TabularDataset('./train.csv')
#print(train_data.head())

# check train labels
label = 'label'
#print(train_data[label].describe())

# 82800 is 23 hours, slurm job is 24 hours.
predictor = TabularPredictor(
   label=label,
   eval_metric="roc_auc",
   problem_type="binary",
   path=f"../koa_scratch/models{trial}"
).fit(
   train_data,
   presets='best_quality',
   #presets='experimental_quality',
   #num_gpus=2,
   num_cpus=48,
   time_limit=82800,
   #hyperparameter_tune_kwargs={
   #    'searcher': 'random',
   #    'scheduler': 'local',
   #    'num_trials': 5
   #},
)

# to load from an existing save
predictor = TabularPredictor.load(f"/home/echung32/koa_scratch/models{trial}")

# print results
results = predictor.fit_summary()

# load test data
test_data = TabularDataset('./test.csv')
# print(test_data.head())

# make predictions
y_pred_proba = predictor.predict_proba(test_data)
# print(y_pred_proba.head())

# save predictions
submission = np.hstack((np.arange(50000).reshape(-1, 1), y_pred_proba.iloc[:, 1].values.reshape(-1, 1)))
# print(submission[:5])

np.savetxt(fname=f'submission{trial}.csv', X=submission, header='Id,Predicted', delimiter=',', comments='')
