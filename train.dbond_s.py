import torch
import yaml
from dbond_s import Model as Net
from dbond_s import focal_loss
from data_utils_dbond_s import PepDataset,collate_callback
from sklearn.metrics import recall_score,precision_score,accuracy_score,confusion_matrix,ConfusionMatrixDisplay,f1_score,roc_auc_score,average_precision_score
from torch.utils.tensorboard import SummaryWriter
import tqdm
from typing import List,Callable
import numpy
import datetime
import argparse
import random
import os



MODEL = 'dbond_s'
# Beijing clock
def get_beijing_time():
    return datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=8)

now = datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=8)
# format
run_time = now.strftime("%Y_%m_%d_%H_%M")

tensorboard_log_pattern = './tensorboard/{model}/{time}_{status}_{tag}'
checkpoint_path_pattern = './checkpoint/{model}/{time}_{tag}_{epoch}.pt'
model_weight_path_pattern = './best_model/{model}/{time}_{tag}_{epoch}.pt'
model_weight_dir_pattern =  './best_model/{model}'
model_weight_dir = model_weight_dir_pattern.format(model = MODEL)

parser = argparse.ArgumentParser()
parser.add_argument('--config',type=str,action='store',default='/workspace/dbond_s_config/default.yaml',help='path to config')

args = parser.parse_args()
print('='*10+'Args'+'='*10)
for k,v in vars(args).items():
    print(f'{k:15}\t{v}')

with open(str(args.config), 'r') as stream:
    config = yaml.safe_load(stream)

label = [0,1]
best_test_auc = 0
best_test_model_path = ''
epoch_cnt_to_save = int(config['train_args']['save_per_epoch'])

# best_loss = 0
train_writer = SummaryWriter(tensorboard_log_pattern.format(model=MODEL,time=run_time,status = 'train',tag = config['tag']))
test_writer = SummaryWriter(tensorboard_log_pattern.format(model=MODEL,time=run_time,status = 'test',tag = config['tag']))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('='*10+str(device)+'='*10)

torch.manual_seed(config['train_args']['seed'])
torch.cuda.manual_seed_all(config['train_args']['seed'])
numpy.random.seed(config['train_args']['seed'])
random.seed(config['train_args']['seed'])

train_dataset = PepDataset(config)
test_dataset = PepDataset(config,False)


train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    shuffle=False,
                    pin_memory=True,
                    batch_size=config['train_args']['batch_size'],
                    collate_fn=collate_callback,
                    num_workers=config['train_args']['dataloader_workers'])

test_dataloader = torch.utils.data.DataLoader(
                    test_dataset,
                    pin_memory=True,
                    shuffle=False,
                    batch_size=config['train_args']['batch_size'],
                    collate_fn=collate_callback,
                    num_workers=config['train_args']['dataloader_workers'])

model = Net(config)

print(str(model))
model.to(device)
optimizer:torch.optim.Optimizer

if config['train_args']['optimizer'].lower() == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(),**config['train_args']['optimizer_args'])
elif config['train_args']['optimizer'].lower() == 'adam':
    optimizer = torch.optim.Adam(model.parameters(),**config['train_args']['optimizer_args'])
loss_func:Callable

if config['train_args']['loss_type'].lower() == 'ce':
    loss_func = lambda logits,labels:torch.nn.functional.cross_entropy(logits,labels,reduction='mean',**config['train_args']['loss_args'])
elif config['train_args']['loss_type'].lower() == 'focal':
    loss_func = lambda logits,labels:focal_loss(logits,labels,reduction='mean',**config['train_args']['loss_args'])



def process(epoch:int, status:str,writer:SummaryWriter,dataloader:torch.utils.data.DataLoader)->dict|None:

    if status.lower() == 'train':
        model.train()
    elif status.lower() == 'test':
        model.eval()
    else:
        return None
    
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
            loop.set_description(f"{status.capitalize()} Epoch [{epoch}/{config['train_args']['epoch']}]")
     
            seq_index_batch=seq_index_batch.to(device)
            seq_padding_mask_batch=seq_padding_mask_batch.to(device)
            bond_index_batch=bond_index_batch.to(device)
            bond_vec_batch=bond_vec_batch.to(device)
            state_vec_batch=state_vec_batch.to(device)
            env_vec_batch=env_vec_batch.to(device)
            label_real_batch = label_real_batch.to(device)
            if status.lower() == 'train':
                model.zero_grad()
            label_predict_batch = model.forward(seq_index_batch=seq_index_batch,
                                                                seq_padding_mask_batch=seq_padding_mask_batch,
                                                                bond_index_batch=bond_index_batch,
                                                                bond_vec_batch=bond_vec_batch,
                                                                state_vec_batch=state_vec_batch,
                                                                env_vec_batch=env_vec_batch)
            loss:torch.Tensor = loss_func(label_predict_batch,label_real_batch)
            
            if status.lower() == 'train':
                loss.backward()
                optimizer.step()

            loss_sum.append(label_real_batch.shape[0]*loss.item())

            # print(label_predict_batch.shape)
            label_prob_batch = torch.nn.functional.softmax(label_predict_batch,dim=1)

            label_predict_batch = label_predict_batch.argmax(dim=1)
            
            preds.extend(label_predict_batch.detach().cpu().numpy())
            preds_probs.extend(label_prob_batch[:,1].detach().cpu().numpy())
            trues.extend(label_real_batch.detach().cpu().numpy())
            loop.set_postfix({'loss':loss.item()} )
            # break
            

    mean_loss:float
    if status.lower() == 'train':
        mean_loss = numpy.sum(loss_sum)/len(train_dataset)
    elif status.lower() == 'test':
        mean_loss = numpy.sum(loss_sum)/len(test_dataset)
	
    sklearn_accuracy = accuracy_score(trues, preds)
    sklearn_auc = roc_auc_score(trues,preds_probs)
    sklearn_ap = average_precision_score(trues,preds_probs)

    sklearn_precision_label_0,sklearn_precision_label_1 = precision_score(trues, preds, average=None)
    sklearn_recall_label_0,sklearn_recall_label_1 = recall_score(trues, preds, average=None)
    sklearn_f1_label_0,sklearn_f1_label_1 = f1_score(trues, preds, average=None)
    
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
    for k,v in metrics_dict.items():
         writer.add_scalar(k,v,epoch)
   
    return metrics_dict



def early_stopping(patience=5, delta=1e-4)->Callable[[float],bool]:
    best_metric = None
    counter = 0
    early_stop = False

    def check_stop(metric:float)->bool:
        nonlocal best_metric, counter, early_stop

        if best_metric is None:
            best_metric = metric
        elif metric < best_metric - delta:
            counter += 1
            if counter >= patience:
                early_stop = True
        else:
            best_metric = metric
            counter = 0

        return early_stop

    return check_stop

early_stop = early_stopping(**config['train_args']['early_stopping'])

def save_checkpoint(save_path,metric,status):
        checkpoint_dict:dict = {}
        checkpoint_dict.update(config)
        checkpoint_dict['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint_dict['model_state_dict'] = model.state_dict()
        checkpoint_dict['train_args']['save_epoch'] = epoch
        checkpoint_dict['metric'] = metric
        checkpoint_dict['status'] = status
        torch.save(checkpoint_dict,save_path)
        return 

for epoch in range(config['train_args']['epoch']):
    
    
    train_metrics_dict =  process(epoch=epoch,status='train',writer=train_writer,dataloader=train_dataloader)
    test_metrics_dict = process(epoch=epoch,status='test',writer=test_writer,dataloader=test_dataloader)

    if epoch % epoch_cnt_to_save == 0:
        save_path = checkpoint_path_pattern.format(model =MODEL, time = get_beijing_time().strftime("%Y_%m_%d_%H_%M"), tag = config['tag'],epoch=epoch) 
        save_checkpoint(save_path,test_metrics_dict,'test')
        print(f'save checkpoint: {save_path}')
        
    if early_stop(test_metrics_dict['AUC']):
        print(f"{'#'*10} early stop {'#'*10}")
        print(f"{'#'*10} epoch: [{epoch}/{config['train_args']['epoch']}] {'#'*10}")
        print(f"{'#'*10} best AUC {best_test_auc:.4} {'#'*10}")
        print(f"{'#'*10} test AUC {test_metrics_dict['AUC']:.4} {'#'*10}")
        break
    if  test_metrics_dict['AUC'] > best_test_auc:
        best_test_auc = test_metrics_dict['AUC']
        save_path = model_weight_path_pattern.format(model =MODEL, time = get_beijing_time().strftime("%Y_%m_%d_%H_%M"), tag = config['tag'],epoch=epoch)
        save_checkpoint(save_path,test_metrics_dict,'test')
        print(f'save model weight: {save_path}')
        if best_test_model_path != '':
            try:
                os.remove(best_test_model_path)
                print(f'remove success: {best_test_model_path}')
            except Exception as e:
                print(f'remove failed: {best_test_model_path}\nerror: {e}')
        best_test_model_path = save_path
train_writer.close()
test_writer.close()