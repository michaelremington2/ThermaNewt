#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import math
import os
import csv
from itertools import chain
from pprint import pprint
### Rattle_Newton
import rattle_newton.sim_snake_tb as therma_sim
#ML
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn import metrics
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier


## Functions
def in_out_binary(in_out_value):
    if in_out_value=='In':
        val = 0
    else:
        val = 1
    return val

#### Load Data


train_fp = 'Data/imp_therma_data_2024_05_06.csv'
output_file_name = 'results/rf_imp_15.csv'

train = pd.read_csv(train_fp)
train['Date'] = pd.to_datetime(train['Date'])
train['Year'] = train['Date'].dt.year
train['Month'] = train['Date'].dt.month
train['Day'] = train['Date'].dt.day
train['Hour'] = train['Date'].dt.hour
train['Minute'] = train['Date'].dt.minute
## Filter out Winter
overwintering = [10, 11, 12, 1, 2, 3, 4]
filter_condition = ~train['Date'].dt.month.isin(overwintering)
train = train[filter_condition]
train =train.dropna().sort_values(by=['Study_Site', 'Snake_Name', 'Date']).reset_index(drop=True)

sim_para_fp = 'Data/parameters_MMSE.csv'
sim_para = pd.read_csv(sim_para_fp)



#### Make Training Labels
# rand np
simulated_t_body_rand_np = []
burrow_usage_rand_np = []
# rand p
simulated_t_body_rand_p = []
burrow_usage_rand_p = []
# bound np
simulated_t_body_bound_np = []
burrow_usage_bound_np = []
# Bound p
simulated_t_body_bound_p = []
burrow_usage_bound_p = []
# pref np
simulated_t_body_pref_np = []
burrow_usage_pref_np = []
# pref p
simulated_t_body_pref_p = []
burrow_usage_pref_p = []

snake_names = train['Snake_Name'].unique()


for snake in snake_names:
    condition1 = train['Snake_Name']==snake
    temp_df = train.loc[condition1]
    # Get Parameters
    para_condition = sim_para['Snake_Name']==snake
    temp_para = sim_para[para_condition]
    ## Set Parameters
    k = 0.01
    t_initial= 25
    t_pref_min = 18
    t_pref_max = 32
    t_opt = 28
    delta_t = 15
    # Estimated Parameters
    t_pref_min_est = float(temp_para['t_pref_min'].iloc[0])
    t_pref_max_est = float(temp_para['t_pref_max'].iloc[0])
    t_opt_est = float(temp_para['t_opt'].iloc[0])

    #Run Simulation w/out parameters
    ## Rand np
    ts_rand_np = therma_sim.ThermalSimulator(flip_logic='random',
                                 t_pref_min=t_pref_min,
                                 t_pref_max=t_pref_max,
                                 t_pref_opt=t_opt, seed=42)
    ss_burrow_usage_rand_np, tb_sim_rand_np = ts_rand_np.tb_simulator_2_state_model_wrapper(
                            k=k,
                            t_initial=t_initial,
                            delta_t=delta_t,
                            burrow_temp_vector=temp_df['Burrow'],
                            open_temp_vector=temp_df['Open'],
                            return_tbody_sim=True)
    burrow_usage_rand_np.append(ss_burrow_usage_rand_np)
    simulated_t_body_rand_np.append(tb_sim_rand_np)

    ## Boundary np
    ts_bound_np = therma_sim.ThermalSimulator(flip_logic='boundary',
                                 t_pref_min=t_pref_min,
                                 t_pref_max=t_pref_max,
                                 t_pref_opt=t_opt, seed=42)
    ss_burrow_usage_bound_np, tb_sim_bound_np = ts_bound_np.tb_simulator_2_state_model_wrapper(
                            k=k,
                            t_initial=t_initial,
                            delta_t=delta_t,
                            burrow_temp_vector=temp_df['Burrow'],
                            open_temp_vector=temp_df['Open'],
                            return_tbody_sim=True)
    burrow_usage_bound_np.append(ss_burrow_usage_bound_np)
    simulated_t_body_bound_np.append(tb_sim_bound_np)

    ## Pref
    ts_pref_np = therma_sim.ThermalSimulator(flip_logic='preferred',
                                 t_pref_min=t_pref_min,
                                 t_pref_max=t_pref_max,
                                 t_pref_opt=t_opt, seed=42)
    ss_burrow_usage_pref_np, tb_sim_pref_np = ts_pref_np.tb_simulator_2_state_model_wrapper(
                            k=k,
                            t_initial=t_initial,
                            delta_t=delta_t,
                            burrow_temp_vector=temp_df['Burrow'],
                            open_temp_vector=temp_df['Open'],
                            return_tbody_sim=True)
    burrow_usage_pref_np.append(ss_burrow_usage_pref_np)
    simulated_t_body_pref_np.append(tb_sim_pref_np)

    ######
    # Run Simulation w/ parameters
    ######

    ## Rand p 
    ts_rand_p = therma_sim.ThermalSimulator(flip_logic='random',
                                 t_pref_min=t_pref_min_est,
                                 t_pref_max=t_pref_max_est,
                                 t_pref_opt=t_opt_est, seed=42)
    ss_burrow_usage_rand_p, tb_sim_rand_p = ts_rand_p.tb_simulator_2_state_model_wrapper(
                            k=k,
                            t_initial=t_initial,
                            delta_t=delta_t,
                            burrow_temp_vector=temp_df['Burrow'],
                            open_temp_vector=temp_df['Open'],
                            return_tbody_sim=True)
    burrow_usage_rand_p.append(ss_burrow_usage_rand_p)
    simulated_t_body_rand_p.append(tb_sim_rand_p)
    ## Bound p 
    ts_bound_p = therma_sim.ThermalSimulator(flip_logic='boundary',
                                 t_pref_min=t_pref_min_est,
                                 t_pref_max=t_pref_max_est,
                                 t_pref_opt=t_opt_est, seed=42)
    ss_burrow_usage_bound_p, tb_sim_bound_p = ts_bound_p.tb_simulator_2_state_model_wrapper(
                            k=k,
                            t_initial=t_initial,
                            delta_t=delta_t,
                            burrow_temp_vector=temp_df['Burrow'],
                            open_temp_vector=temp_df['Open'],
                            return_tbody_sim=True)
    burrow_usage_bound_p.append(ss_burrow_usage_bound_p)
    simulated_t_body_bound_p.append(tb_sim_bound_p)

    ## Pref p 
    ts_pref_p = therma_sim.ThermalSimulator(flip_logic='preferred',
                                 t_pref_min=t_pref_min_est,
                                 t_pref_max=t_pref_max_est,
                                 t_pref_opt=t_opt_est, seed=42)
    ss_burrow_usage_pref_p, tb_sim_pref_p = ts_pref_p.tb_simulator_2_state_model_wrapper(
                            k=k,
                            delta_t=delta_t,
                            t_initial=t_initial,
                            burrow_temp_vector=temp_df['Burrow'],
                            open_temp_vector=temp_df['Open'],
                            return_tbody_sim=True)
    burrow_usage_pref_p.append(ss_burrow_usage_pref_p)
    simulated_t_body_pref_p.append(tb_sim_pref_p)


## rand
burrow_usage_rand_np = list(chain.from_iterable(burrow_usage_rand_np))
simulated_t_body_rand_np = list(chain.from_iterable(simulated_t_body_rand_np))
    
train['Burrow_Usage_rand_np'] = burrow_usage_rand_np
train['tb_sim_rand_np'] = simulated_t_body_rand_np
train['Burrow_Usage_rand_np'] = [in_out_binary(i) for i in train['Burrow_Usage_rand_np']]

    
burrow_usage_rand_p = list(chain.from_iterable(burrow_usage_rand_p))
simulated_t_body_rand_p = list(chain.from_iterable(simulated_t_body_rand_p))
    
train['Burrow_Usage_rand_p'] = burrow_usage_rand_p
train['tb_sim_rand_p'] = simulated_t_body_rand_p
train['Burrow_Usage_rand_p'] = [in_out_binary(i) for i in train['Burrow_Usage_rand_p']]

## Bound
burrow_usage_bound_np = list(chain.from_iterable(burrow_usage_bound_np))
simulated_t_body_bound_np = list(chain.from_iterable(simulated_t_body_bound_np))
    
train['Burrow_Usage_bound_np'] = burrow_usage_bound_np
train['tb_sim_bound_np'] = simulated_t_body_bound_np
train['Burrow_Usage_bound_np'] = [in_out_binary(i) for i in train['Burrow_Usage_bound_np']]

    
burrow_usage_bound_p = list(chain.from_iterable(burrow_usage_bound_p))
simulated_t_body_bound_p = list(chain.from_iterable(simulated_t_body_bound_p))
    
train['Burrow_Usage_bound_p'] = burrow_usage_bound_p
train['tb_sim_bound_p'] = simulated_t_body_bound_p
train['Burrow_Usage_bound_p'] = [in_out_binary(i) for i in train['Burrow_Usage_bound_p']]

## Pref
burrow_usage_pref_np = list(chain.from_iterable(burrow_usage_pref_np))
simulated_t_body_pref_np = list(chain.from_iterable(simulated_t_body_pref_np))
    
train['Burrow_Usage_pref_np'] = burrow_usage_pref_np
train['tb_sim_pref_np'] = simulated_t_body_pref_np
train['Burrow_Usage_pref_np'] = [in_out_binary(i) for i in train['Burrow_Usage_pref_np']]

    
burrow_usage_pref_p = list(chain.from_iterable(burrow_usage_pref_p))
simulated_t_body_pref_p = list(chain.from_iterable(simulated_t_body_pref_p))
    
train['Burrow_Usage_pref_p'] = burrow_usage_pref_p
train['tb_sim_pref_p'] = simulated_t_body_pref_p
train['Burrow_Usage_pref_p'] = [in_out_binary(i) for i in train['Burrow_Usage_pref_p']]


#### Do this after training labels
per_site_data = {}
unique_sites = train['Study_Site'].unique()
for site in unique_sites:
    condition = train['Study_Site']==site
    temp_df = train.loc[condition]
    per_site_data[site] = temp_df
per_site_data['total'] = train



simulation_labels = {'random': {'np': 'Burrow_Usage_rand_np',
                                'p': 'Burrow_Usage_rand_p'},
                    'boundary': {'np': 'Burrow_Usage_bound_np',
                                 'p': 'Burrow_Usage_bound_p'},
                    'preference': {'np': 'Burrow_Usage_pref_np',
                                   'p': 'Burrow_Usage_pref_p'}}

val_fp = 'Data/validation.csv'
val = pd.read_csv(val_fp)
val['datetime'] = pd.to_datetime(val['datetime'])

with open(output_file_name, mode='w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)
    # Write Header
    writer.writerow(["site_label", "rbp_label", "para_est_label", 
                    "accuracy_test", "precision_test", "recall_test", "f1_test",
                    "TP_test", "FP_test", "TN_test", "FN_test", "auc_test",
                    "accuracy_val", "precision_val", "recall_val", "f1_val",
                    "TP_val", "FP_val", "TN_val", "FN_val", "auc_val"]) 
    for site_label, site_data in per_site_data.items():
        for rbp_label, para_est_type in simulation_labels.items():
            for para_est_label, burrow_label in para_est_type .items():
                X = site_data[['Open','Burrow','Month','Day','Hour','Minute']] 
                y = site_data[burrow_label]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                RF = RandomForestClassifier(n_estimators = 1800,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            max_features=None,
                            max_depth=20,
                            bootstrap=False,
                            random_state=42)
                RF.fit(X_train, y_train)
                y_pred = RF.predict(X_test)
                # results test set
                cm_test = confusion_matrix(y_test, y_pred)
                TP_test = cm_test[1, 1]
                FP_test = cm_test[0, 1]
                TN_test = cm_test[0, 0]
                FN_test = cm_test[1, 0]
                accuracy_test = accuracy_score(y_test, y_pred)
                precision_test = precision_score(y_test, y_pred)
                recall_test = recall_score(y_test, y_pred)
                f1_test = f1_score(y_test, y_pred)
                y_pred_proba = RF.predict_proba(X_test)[::,1]
                auc_test = metrics.roc_auc_score(y_test, y_pred_proba)
                # Results validation set
                emp_val = site_data.merge(val, left_on=['Snake_Name', 'Year', 'Month', 'Day', 'Hour'],right_on=['Snake_Name', 'year', 'month', 'day', 'hour'], how='inner')
                emp_val = emp_val.dropna(subset=[burrow_label, 'In_Out_Burrow', 'Burrow', 'Open'])
                X_emp = emp_val[['Open','Burrow','Month','Day','Hour','Minute']] #,'Year'
                y_emp = emp_val['In_Out_Burrow']
                y_pred_emp = RF.predict(X_emp)
                y_pred_proba_emp = RF.predict_proba(X_emp)[::,1]
                try:
                    auc_val = metrics.roc_auc_score(y_emp, y_pred_proba_emp)
                    cm_val = confusion_matrix(y_emp, y_pred_emp)
                    TP_val = cm_val[1, 1]
                    FP_val = cm_val[0, 1]
                    TN_val = cm_val[0, 0]
                    FN_val = cm_val[1, 0]
                    accuracy_val = accuracy_score(y_emp, y_pred_emp)
                    precision_val = precision_score(y_emp, y_pred_emp)
                    recall_val = recall_score(y_emp, y_pred_emp)
                    f1_val = f1_score(y_emp, y_pred_emp)
                except ValueError:
                    auc_val = 0
                    cm_val = 0
                    TP_val = 0
                    FP_val = 0
                    TN_val = 0
                    FN_val = 0
                    accuracy_val = 0
                    precision_val = 0
                    recall_val = 0
                    f1_val = 0
                # Write Data
                writer.writerow([site_label, rbp_label, para_est_label, 
                                accuracy_test, precision_test, recall_test, f1_test,
                                TP_test, FP_test, TN_test, FN_test, auc_test,
                                accuracy_val, precision_val, recall_val, f1_val,
                                TP_val, FP_val, TN_val, FN_val, auc_val])



