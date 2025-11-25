import torch
from dbond_s import Model as Net
from dbond_s import focal_loss
from data_utils_dbond_s import PepDataset,collate_callback
from sklearn.metrics import recall_score,precision_score,accuracy_score,confusion_matrix,ConfusionMatrixDisplay,f1_score,roc_auc_score,average_precision_score
import tqdm
from typing import List,Dict,Callable,Tuple
import numpy
import datetime
import argparse
import random
import os
import matplotlib.pyplot as plt
import pandas
import yaml
import re
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)
MODEL = 'dbond_s'
tag = 'default'
pattern = f'.*{tag}_\\d{{1,2}}'
pattern =re.compile(pattern)
# Beijing clock
def get_beijing_time():
    return datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=8)

now = datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=8)
# format
run_time = now.strftime("%Y_%m_%d_%H_%M")
pad_len = 36-1
best_model_name = [file for file in os.listdir(f'./best_model/{MODEL}') if pattern.match(file)][0]

model_weight_path = f'./best_model/{MODEL}/{best_model_name}'
pred_result_path = f'./result/{MODEL}/{best_model_name}.pred.csv'
config_path = f'./{MODEL}_config/{tag}.yaml'
multi_label_metric_path = f'./result/multi_label_metric/{MODEL}/{tag}_metric.csv'
#####
fbr_csv_path = './dataset/dataset.fbr.csv'
condition_col = ['rt','intensity','scan_num','seq','nce','charge']
tb_col = 'tb'
######
best_model_check_point:Dict[str,Dict]
best_model_check_point = torch.load(model_weight_path)

with open(str(config_path), 'r') as stream:
    config = yaml.safe_load(stream)


label = [0,1]
dataset_path = config['csv']['test_dataset_path']
print('='*10+'train args'+'='*10)
for k,v in best_model_check_point['train_args'].items():
    print(f'{k:15}\t{v}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('='*10+str(device)+'='*10)


torch.manual_seed(config['train_args']['seed'])
torch.cuda.manual_seed_all(config['train_args']['seed'])
numpy.random.seed(config['train_args']['seed'])
random.seed(config['train_args']['seed'])

dataset = PepDataset(config,False)


dataloader = torch.utils.data.DataLoader(
                    dataset,
                    pin_memory=True,
                    shuffle=False,
                    batch_size=config['train_args']['batch_size'],
                    collate_fn=collate_callback,
                    num_workers=config['train_args']['dataloader_workers'])

model = Net(config)

model.load_state_dict(best_model_check_point['model_state_dict'])
print(str(model))
model.to(device)

loss_func:Callable

if config['train_args']['loss_type'].lower() == 'ce':
    loss_func = lambda logits,labels:torch.nn.functional.cross_entropy(logits,labels,reduction='mean',**config['train_args']['loss_args'])
elif config['train_args']['loss_type'].lower() == 'focal':
    loss_func = lambda logits,labels:focal_loss(logits,labels,reduction='mean',**config['train_args']['loss_args'])



def evaluate(dataloader:torch.utils.data.DataLoader)->Tuple[dict,List[int],List[int],List[float],List[float],Dict[str,List[float]]]:

    model.eval()
    
    loss_sum = []
    preds = []
    preds_probs = []
    trues = []

    seq_index_batch	:torch.Tensor
    seq_padding_mask_batch	:torch.Tensor
    bond_index_batch	:torch.Tensor
    bond_vec_batch	:torch.Tensor
    state_vec_batch	:torch.Tensor
    env_vec_batch	:torch.Tensor
    label_real_batch: torch.Tensor
    


    with tqdm.tqdm(dataloader, total =len(dataloader),unit='batch') as loop:
          for seq_index_batch,seq_padding_mask_batch,bond_index_batch,bond_vec_batch,state_vec_batch,env_vec_batch,label_real_batch in loop:
            loop.set_description(f'EVAL')
     
            seq_index_batch=seq_index_batch.to(device)
            seq_padding_mask_batch=seq_padding_mask_batch.to(device)
            bond_index_batch=bond_index_batch.to(device)
            bond_vec_batch=bond_vec_batch.to(device)
            state_vec_batch=state_vec_batch.to(device)
            env_vec_batch=env_vec_batch.to(device)
            label_real_batch = label_real_batch.to(device)
           
            label_predict_batch = model.forward(seq_index_batch=seq_index_batch,
                                                                seq_padding_mask_batch=seq_padding_mask_batch,
                                                                bond_index_batch=bond_index_batch,
                                                                bond_vec_batch=bond_vec_batch,
                                                                state_vec_batch=state_vec_batch,
                                                                env_vec_batch=env_vec_batch)
            loss:torch.Tensor = loss_func(label_predict_batch,label_real_batch)
    

            loss_sum.append(label_real_batch.shape[0]*loss.item())

            # print(label_predict_batch.shape)
            label_prob_batch = torch.nn.functional.softmax(label_predict_batch,dim=1)

            label_predict_batch = label_predict_batch.argmax(dim=1)

  
            
            preds.extend(label_predict_batch.detach().cpu().numpy())
            preds_probs.extend(label_prob_batch[:,1].detach().cpu().numpy())
            trues.extend(label_real_batch.detach().cpu().numpy())
            loop.set_postfix({'loss':loss.item()} )
            # break
            
    mean_loss = numpy.sum(loss_sum)/len(dataset)
	
    sklearn_accuracy = accuracy_score(trues, preds)
    sklearn_auc = roc_auc_score(trues,preds_probs)
    sklearn_ap = average_precision_score(trues,preds_probs)

    sklearn_precision_label_0,sklearn_precision_label_1 = precision_score(trues, preds, average=None)
    sklearn_recall_label_0,sklearn_recall_label_1 = recall_score(trues, preds, average=None)
    sklearn_f1_label_0,sklearn_f1_label_1 = f1_score(trues, preds, average=None)
    
    cm = confusion_matrix(trues,preds,labels=label)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=label)
    disp.plot()
    

    # 保存图为svg格式，即矢量图格式
    plt.savefig(f"evaluate_{MODEL}.svg", dpi=300,format="svg")
    
    metrics_dict = {
        'Loss':mean_loss,
        'accuracy':sklearn_accuracy,
        'AUC':sklearn_auc,
        'AP':sklearn_ap,
        'precision':(sklearn_precision_label_0+sklearn_precision_label_1)/2,
        'recall':(sklearn_recall_label_0+sklearn_recall_label_1)/2,
        'f1':(sklearn_f1_label_0+sklearn_f1_label_1)/2,
        'Label 0: precision':sklearn_precision_label_0,
        'Label 1: precision':sklearn_precision_label_1,
        'Label 0: recall':sklearn_recall_label_0,
        'Label 1: recall':sklearn_recall_label_1,
        'Label 0: f1':sklearn_f1_label_0,
        'Label 1: f1':sklearn_f1_label_1,
    }
    
    # Tuple[dict,List[int],List[int],List[float],List[float],Dict[str,List[float]]]
    return metrics_dict,trues,preds
   
    
    

    
   

metric,trues,preds = evaluate(dataloader=dataloader)

print('='*10+'evaluate metric'+'='*10)
for k,v in metric.items():
    print(f'{k:30}\t{v:.4}')

bond_csv_df = pandas.read_csv(dataset_path,na_filter=None)
bond_csv_df['true'] = trues
bond_csv_df['pred'] = preds
print(f'bond predict finish')
bond_csv_df.to_csv(pred_result_path,index=False)

fbr_csv_df:pandas.DataFrame = pandas.read_csv(fbr_csv_path,na_filter=None)
pep_list = dataset.df['seq'].unique()
fbr_csv_df = fbr_csv_df[fbr_csv_df['seq'].isin(pep_list)]
def process_row(row:pandas.Series):
    sub_df:pandas.DataFrame = bond_csv_df
    for k in condition_col:
        condition = sub_df[k].values==row[k]
        sub_df = sub_df[condition]
    assert len(sub_df) == row[tb_col],f'condition is not unique !\n{condition_col}\ntb = {row[tb_col]}\nlen(sub_df) = {len(sub_df)}'
    sub_df.sort_values(by='bond_pos')
    label_true_list = sub_df['true'].tolist()
    label_pred_list = sub_df['pred'].tolist()
    multi_label_true_str=';'.join(map(str, label_true_list))
    multi_label_pred_str=';'.join(map(str, label_pred_list))
    return multi_label_true_str,multi_label_pred_str

fbr_csv_df['true_multi'],fbr_csv_df['pred_multi'] =   zip(*fbr_csv_df.parallel_apply(lambda row: process_row(row),axis=1))

multi_label_true_str_list = fbr_csv_df['true_multi'].tolist()
multi_label_pred_str_list = fbr_csv_df['pred_multi'].tolist()
print(f'compute multi label metric')
def masked_metric(multi_label_true_str_list:List[str],multi_label_pred_str_list:List[str]):

    import numpy as np
    multi_label_true_list_list = [list(map(int,item.split(';'))) for item in multi_label_true_str_list]
    multi_label_true_list_list = [item+[0]*(pad_len-len(item)) for item in multi_label_true_list_list]
    multi_label_true_list_list = np.array(multi_label_true_list_list)
    multi_label_pred_list_list = [list(map(int,item.split(';'))) for item in multi_label_pred_str_list]
    multi_label_pred_list_list = [item+[0]*(pad_len-len(item)) for item in multi_label_pred_list_list]
    multi_label_pred_list_list = np.array(multi_label_pred_list_list)
    n = len(multi_label_true_list_list)
    m = len(multi_label_true_list_list[0])
    print(f'n: {n}\nm:{m}')
   
    gt = multi_label_true_list_list
    predict = multi_label_pred_list_list
    import multi_label_metrics
    subset_acc =multi_label_metrics.example_subset_accuracy(gt, predict)
    ex_acc =multi_label_metrics.example_accuracy(gt, predict)
    ex_precision =multi_label_metrics.example_precision(gt, predict)
    ex_recall =multi_label_metrics.example_recall(gt, predict)
    ex_f1 =multi_label_metrics.example_f1(gt, predict)

    lab_acc_ma =multi_label_metrics.label_accuracy_macro(gt, predict)
    lab_acc_mi =multi_label_metrics.label_accuracy_micro(gt, predict)
    lab_precision_ma =multi_label_metrics.label_precision_macro(gt, predict)
    lab_precision_mi =multi_label_metrics.label_precision_micro(gt, predict)
    lab_recall_ma =multi_label_metrics.label_recall_macro(gt, predict)
    lab_recall_mi =multi_label_metrics.label_recall_micro(gt, predict)
    lab_f1_ma =multi_label_metrics.label_f1_macro(gt, predict)
    lab_f1_mi =multi_label_metrics.label_f1_micro(gt, predict)

    metric['subset_acc'] = subset_acc
    metric['ex_acc'] = ex_acc
    metric['ex_precision'] = ex_precision
    metric['ex_recall'] = ex_recall
    metric['ex_f1'] = ex_f1
    metric['lab_acc_ma'] = lab_acc_ma
    metric['lab_acc_mi'] = lab_acc_mi
    metric['lab_precision_ma'] = lab_precision_ma
    metric['lab_precision_mi'] = lab_precision_mi
    metric['lab_recall_ma'] = lab_recall_ma
    metric['lab_recall_mi'] = lab_recall_mi
    metric['lab_f1_ma'] = lab_f1_ma
    metric['lab_f1_mi'] = lab_f1_mi
    
    print("subset acc\t\t%.4f" %subset_acc)
    print("example acc\t\t%.4f" %ex_acc)
    print("example precision\t\t%.4f" %ex_precision)
    print("example recall\t\t%.4f" %ex_recall)
    print("example f1\t\t%.4f" %ex_f1)
    print("label acc macro\t\t%.4f" %lab_acc_ma)
    print("label acc micro\t\t%.4f" %lab_acc_mi)
    print("label prec macro\t\t%.4f" %lab_precision_ma)
    print("label prec micro\t\t%.4f" %lab_precision_mi)
    print("label rec macro\t\t%.4f" %lab_recall_ma)
    print("label rec micro\t\t%.4f" %lab_recall_mi)
    print("label f1 macro\t\t%.4f" %lab_f1_ma)
    print("label f1 micro\t\t%.4f" %lab_f1_mi)

masked_metric(multi_label_true_str_list,multi_label_pred_str_list)
metric_df = pandas.DataFrame(list(metric.items()), columns=['metric', 'value'])
metric_df.to_csv(multi_label_metric_path,index=False)