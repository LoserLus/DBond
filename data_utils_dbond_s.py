from torch.utils.data import Dataset,default_collate, dataloader
from typing import Tuple,List

import pandas
import numpy
from torch import tensor,Tensor,LongTensor



def collate_callback(data):
    if isinstance(data,tuple):
        # from getitems: Tuple[Tensor,Tensor,Tensor,Tensor]
        # batch data 
        return data
    else:
        # single data to batch
        return default_collate(data)


class PepDataset(Dataset):
    def __init__(self,config:str,is_train:bool=True) -> None:
        super().__init__()
   
        self.csv_path = config['csv']['train_dataset_path'] if is_train else  config['csv']['test_dataset_path']
        self.df:pandas.DataFrame = pandas.read_csv(self.csv_path,na_filter=False)
        self.dataset_len = len(self.df)
        # GB
        self.mem_usage = self.df.memory_usage().sum()/ (1024**3)
        self.seq_col_name = config['csv']['seq_col_name'] 
        self.bond_index_col_name = config['csv']['bond_index_col_name']
        self.bond_var_col_name_list:List[str] = config['csv']['bond_var_col_name'] 
        self.state_var_col_name_list:List[str] = config['csv']['state_var_col_name'] 
        self.env_var_col_name_list:List[str] = config['csv']['env_var_col_name'] 
        self.label_col_name = config['csv']['label_col_name'] 
        self.pad_char = config['seq']['pad_char']
        self.alphabet:List[str] = list(config['seq']['alphabet'])
        self.max_len = int(config['seq']['max_len'])
        self.alphabet_pos_dict:dict[str,int] = {}
        for i,c in enumerate(self.alphabet):
            self.alphabet_pos_dict[c] = i

    
    
    def __getitems__(self,index:List[int])->Tuple[Tensor,Tensor,Tensor,Tensor]:
        # print('called')
        if isinstance(index,list):
            return self.__getitem__(index)
        else:
            raise TypeError(f'__getitems__ not support index type: {type(index)}')

    def __getitem__(self, index:int|List[int])->Tuple[Tensor,Tensor,Tensor,Tensor]:
 
        if isinstance(index,int):
            row:pandas.Series = self.df.iloc[index]
            seq:str = row[self.seq_col_name]
            bond_index = int(row[self.bond_index_col_name])
            bond_vars: numpy.ndarray = row[self.bond_var_col_name_list].values.astype(numpy.float32)
            state_vars: numpy.ndarray = row[self.state_var_col_name_list].values.astype(numpy.float32)
            env_vars:numpy.ndarray = row[self.env_var_col_name_list].values.astype(numpy.float32)
            label = int(row[self.label_col_name])
            seq_index = self.seq2index(seq)
            seq_mask = self.seq2mask(seq)
            return tensor(seq_index),tensor(seq_mask),tensor(bond_index),tensor(bond_vars),tensor(state_vars),tensor(env_vars),tensor(label)
        elif isinstance(index,list):
            # seq_index_batch : [batch_len,max_seq_len]
            # seq_padding_mask_batch : [batch_len,max_seq_len]
            # bond_index_batch : [batch_len,]
            # bond_vars: [batch_len,UI_DIM]
            # bi_vec_batch: [batch_len,BI_DIM]
            # env_vec_batch: [batch_len,ENV_DIM]
            # label: [batch,]
            
            df_sub:pandas.DataFrame = self.df.iloc[index]
            seq_index_list:List[List[int]] = df_sub[self.seq_col_name].apply(self.seq2index).to_list()
            seq_mask_list:List[List[int]] = df_sub[self.seq_col_name].apply(self.seq2mask).to_list()
            bond_index_ndarray:numpy.ndarray = df_sub[self.bond_index_col_name].values.astype(numpy.int32)
            bond_vars_ndarray:numpy.ndarray = df_sub[self.bond_var_col_name_list].values.astype(numpy.float32)
            state_vars_ndarray:numpy.ndarray = df_sub[self.state_var_col_name_list].values.astype(numpy.float32)
            env_vars_ndarray:numpy.ndarray = df_sub[self.env_var_col_name_list].values.astype(numpy.float32)
            label_ndarray:numpy.ndarray = df_sub[self.label_col_name].values.astype(numpy.int32)
            
            return tensor(seq_index_list),tensor(seq_mask_list),tensor(bond_index_ndarray),tensor(bond_vars_ndarray),tensor(state_vars_ndarray),tensor(env_vars_ndarray),LongTensor(label_ndarray)
            
    def __len__(self) -> int:
        return self.dataset_len
    
    def seq2index(self,seq:str)->List[int]:
        # assert check_seq(seq),f'{seq} is too long, max length is {SEQ_MAX_LEN}'
        seq_pad =  seq.ljust(self.max_len,self.pad_char)
        seq_index = [ self.alphabet_pos_dict[aa] for aa in seq_pad]
        return seq_index
    def seq2mask(self,seq:str)->List[bool]:
        l = [False]*self.max_len
        l[len(seq):] = [True] *  (self.max_len - len(seq))
        return l
