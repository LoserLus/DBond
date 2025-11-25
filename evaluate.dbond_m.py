import torch
import os
from typing import Dict
from pprint import pprint
import pandas

best_model_dir = '/workspace/best_model/dbond_m'
metric_dir = '/workspace/result/multi_label_metric/dbond_m'
best_model_name = [file for file in os.listdir(best_model_dir)]
for i in range(len(best_model_name)):
    print(best_model_name[i])
    best_model_path = os.path.join(best_model_dir,best_model_name[i])
    metric_path = os.path.join(metric_dir,best_model_name[i]+'.metric.csv')

    best_model_check_point:Dict[str,Dict]
    best_model_check_point = torch.load(best_model_path)
    metrics_dict = best_model_check_point['metric']
    pprint(metrics_dict)
    metric_df = pandas.DataFrame(list(metrics_dict.items()), columns=['metric', 'value'])
    metric_df.to_csv(metric_path,index=False)