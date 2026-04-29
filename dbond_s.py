import torch
import torch.nn as nn
import torch.functional as F
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from typing import List
import yaml


class AttentionBlock(nn.Module):
    def __init__(self,hidden_dim, num_heads, dropout, forward_expansion,**kwargs):
        super(AttentionBlock, self).__init__()
        self.attentionLayer = nn.MultiheadAttention(
            embed_dim = hidden_dim,
            num_heads = num_heads,
            dropout = dropout,
            batch_first=True,
            **kwargs
        )
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, forward_expansion * hidden_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self,query:torch.Tensor,key:torch.Tensor,value:torch.Tensor,**kwargs)->torch.Tensor:
        attn_output_batch,attn_weight_batch = self.attentionLayer.forward(query=query,key=key,value=value,**kwargs)
        x =   self.layer_norm_1(query+self.dropout(attn_output_batch))
        res_x = x
        x = self.ffn(x)
        out = self.layer_norm_2(res_x + self.dropout(x))
        return out,attn_weight_batch    

class MyAttentionBlock(AttentionBlock):
    pass


class Scalar2VecBlock(nn.Module):
    def __init__(self,hidden_dim, dropout):
        super(Scalar2VecBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self,scalar:torch.Tensor)->torch.Tensor:
        # scalar: [batch,1]
        # vec: [batch, hidden_dim]
        vec = self.block(scalar)
        return vec


class StateEncoder(nn.Module):
    def __init__(self,state_dim,hidden_dim, dropout):
        super(StateEncoder, self).__init__()
        self.scalar2vec = nn.ModuleList([Scalar2VecBlock(hidden_dim,dropout) for _ in range(state_dim)])
   
        self.embed_dim = hidden_dim
    def forward(self, state_vec_batch:torch.Tensor)->torch.Tensor:
        # state_vec_batch: [batch_len,state_dim]
        state_vec_batch = state_vec_batch.float()
        cnt = 0
        #  x: [batch_len,state_dim,HIDDEN_DIM]
        x = torch.empty((state_vec_batch.shape[0],state_vec_batch.shape[1],self.embed_dim),dtype=torch.float,device=state_vec_batch.device)
        for scalar2vec in self.scalar2vec:
            x[:,cnt,:] = scalar2vec(state_vec_batch[:,cnt].unsqueeze(1))
            cnt+=1
        return x 
    
class EnvEncoder(nn.Module):
    def __init__(self,env_dim,hidden_dim, dropout):
        super(EnvEncoder, self).__init__()
        self.scalar2vec = nn.ModuleList([Scalar2VecBlock(hidden_dim,dropout) for _ in range(env_dim)])
   
        self.embed_dim = hidden_dim
    def forward(self, env_vec_batch:torch.Tensor)->torch.Tensor:
        # env_vec_batch: [batch_len,ENV_DIM]

        env_vec_batch = env_vec_batch.float()
        cnt = 0
        #  x: [batch_len,env_dim,HIDDEN_DIM]
        x = torch.empty((env_vec_batch.shape[0],env_vec_batch.shape[1],self.embed_dim),dtype=torch.float,device=env_vec_batch.device)
        for scalar2vec in self.scalar2vec:
            x[:,cnt,:] = scalar2vec(env_vec_batch[:,cnt].unsqueeze(1))
            cnt+=1
        return x 

class BondEncoder(nn.Module):
    def __init__(self,bond_dim,hidden_dim, dropout, ):
        super(BondEncoder, self).__init__()
        self.scalar2vec = nn.ModuleList([Scalar2VecBlock(hidden_dim,dropout) for _ in range(bond_dim)])

        self.embed_dim = hidden_dim
    def forward(self,bond_vec_batch:torch.Tensor)->tuple[torch.Tensor,List[torch.Tensor]]:
        # bond_vec_batch: [batch_len,bond_dim]
        bond_vec_batch = bond_vec_batch.float()
        cnt = 0
        #  x: [batch_len,bond_dim,HIDDEN_DIM]
        x = torch.empty((bond_vec_batch.shape[0],bond_vec_batch.shape[1],self.embed_dim),dtype=torch.float,device=bond_vec_batch.device)
        for scalar2vec in self.scalar2vec:
            x[:,cnt,:] = scalar2vec(bond_vec_batch[:,cnt].unsqueeze(1))
            cnt+=1
       
        
        return x

class SeqEncoder(nn.Module):
    def __init__(self,hidden_dim,num_heads, dropout, forward_expansion,attention_layer_num):
        super(SeqEncoder, self).__init__()
        self.self_attention = nn.ModuleList([
                AttentionBlock(hidden_dim, num_heads, dropout, forward_expansion) 
                for _ in range(attention_layer_num)
                                        ])
    
    def forward(self, seq_embedding_batch:torch.Tensor,seq_padding_mask_batch:torch.Tensor)->tuple[torch.Tensor,List[torch.Tensor]]:
        # seq_embedding_batch : [batch_len,max_seq_len,HIDDEN_DIM]
        # seq_padding_mask_batch : [batch_len,max_seq_len],true for pad pos
        attn_output_batch = seq_embedding_batch
        # attn_output_batch : [batch_len,max_seq_len,HIDDEN_DIM]
        # attn_output_vec_batch : [batch_len,1,HIDDEN_DIM]
        # attn_weight_batch : [batch_len,max_seq_len,max_seq_len]
        # attn_weight_batch_list : [attention_layer_num,batch_len,max_seq_len,max_seq_len]
        attn_weight_batch_list = []
        for layer in self.self_attention:
            attn_output_batch,attn_weight_batch = layer.forward(attn_output_batch,attn_output_batch,attn_output_batch,key_padding_mask=seq_padding_mask_batch)
            attn_weight_batch_list.append(attn_weight_batch)
        # attn_output_vec_batch = self.masked_mean(attn_output_batch,~seq_padding_mask_batch)
        return attn_output_batch,attn_weight_batch_list

class BondSeqEnvEncoder(nn.Module):
    def __init__(self,env_dim,hidden_dim,num_heads, dropout, forward_expansion,attention_layer_num):
        super(BondSeqEnvEncoder, self).__init__()
        self.scalar2vec = nn.ModuleList([Scalar2VecBlock(hidden_dim,dropout) for _ in range(env_dim)])
        self.bond_seq_env_block = nn.ModuleList([
                MyAttentionBlock(hidden_dim, num_heads, dropout, forward_expansion) 
                for _ in range(attention_layer_num)
                                        ])
        self.embed_dim = hidden_dim
    def forward(self, bond_info_vec_batch:torch.Tensor,seq_vec_batch:torch.Tensor,env_vec_batch:torch.Tensor)->tuple[torch.Tensor,List[torch.Tensor]]:
        # bond_info_vec_batch : [batch_len,1,HIDDEN_DIM]
        # seq_vec_batch : [batch_len,1,HIDDEN_DIM]
        # env_vec_batch: [batch_len,ENV_DIM]
        env_vec_batch = env_vec_batch.float()
        cnt = 0
        #  x: [batch_len,state_dim,HIDDEN_DIM]
        x = torch.empty((env_vec_batch.shape[0],env_vec_batch.shape[1],self.embed_dim),dtype=torch.float,device=env_vec_batch.device)
        for scalar2vec in self.scalar2vec:
            x[:,cnt,:] = scalar2vec(env_vec_batch[:,cnt].unsqueeze(1))
            cnt+=1
        # K: [batch_len,1+1+env_dim,HIDDEN_DIM]
        K = torch.concat((bond_info_vec_batch,seq_vec_batch,x),dim=1)
        
        # attn_output_batch : [batch_len,1,HIDDEN_DIM]
        # attn_weight_batch : [batch_len,1,1+1+env_dim]
        # attn_weight_batch_list: [attention_layer_num,batch_len,1,1+1+env_dim]
        attn_weight_batch_list = []
        attn_output_batch = bond_info_vec_batch
        for layer in self.bond_seq_env_block:
            attn_output_batch,attn_weight_batch = layer.forward(attn_output_batch,K,K)
            attn_weight_batch_list.append(attn_weight_batch)
        
        return attn_output_batch,attn_weight_batch_list   
    
class Encoder(nn.Module):
    def __init__(self,aa_type_count,pad_index,state_dim,bond_dim,env_dim,hidden_dim,num_heads, dropout, forward_expansion,attention_layer_num):
        super(Encoder, self).__init__()
        self.embedding_layer = nn.Embedding(aa_type_count,hidden_dim,padding_idx=pad_index)
        self.pos_encoder = Summer(PositionalEncoding1D(hidden_dim))
        self.seq_self_encoder = SeqEncoder(hidden_dim,num_heads, dropout, forward_expansion,attention_layer_num)
        self.bond_encoder = BondEncoder(bond_dim,hidden_dim,dropout)
        self.state_encoder = StateEncoder(state_dim,hidden_dim,dropout)
        self.env_encoder = EnvEncoder(env_dim,hidden_dim,dropout)
        
    def masked_mean(self,input:torch.Tensor,mask:torch.Tensor):
        # input: [batch,seq_len,hidden]
        # mask: [batch,seq_len], bool ,true is valid
        mask_float = mask.float()
        valid_counts = mask_float.sum(dim=1, keepdim=True)
        weighted_sum = (input * mask_float.unsqueeze(-1)).sum(dim=1, keepdim=True)
        mean = weighted_sum / valid_counts.clamp(min=1e-6).unsqueeze(-1)
        # mean: [batch,1,hidden]
        return mean

    def forward(self, seq_index_batch:torch.Tensor,
                seq_padding_mask_batch:torch.Tensor,
                bond_index_batch:List[int],
                state_vec_batch:torch.Tensor,
                bond_vec_batch:torch.Tensor,
                env_vec_batch:torch.Tensor)->torch.Tensor:
        # seq_index_batch : [batch_len,max_seq_len]
        # seq_padding_mask_batch : [batch_len,max_seq_len]
        # bond_index_batch : [batch_len,]
        # bond_vec_batch: [batch_len,bond_dim]
        # state_vec_batch: [batch_len,state_dim]
        # env_vec_batch: [batch_len,ENV_DIM]
  
        # seq_embedding_batch : [batch_len,max_seq_len,HIDDEN_DIM]
        seq_embedding_batch:torch.Tensor = self.embedding_layer(seq_index_batch)
        seq_embedding_batch              = self.pos_encoder(seq_embedding_batch)
        # bond_embedding_batch : [batch_len,2,HIDDEN_DIM]
     
        # seq_embedding_batch : [batch_len,max_seq_len,HIDDEN_DIM]
        # seq_attn_weight_list :[attention_layer_num,batch_len,max_seq_len,max_seq_len]
        seq_embedding_batch,seq_attn_weight_list = self.seq_self_encoder.forward(seq_embedding_batch,seq_padding_mask_batch)
        # seq_vec_batch : [batch_len,1,HIDDEN_DIM]
        seq_vec_batch = self.masked_mean(seq_embedding_batch,~seq_padding_mask_batch)
        
        # state_vec_batch: [batch_len,state_dim,HIDDEN_DIM]
        # env_vec_batch: [batch_len,env_dim,HIDDEN_DIM]
        env_vec_batch = self.env_encoder.forward(env_vec_batch)
        state_vec_batch = self.state_encoder.forward(state_vec_batch)
        bond_vec_batch = self.bond_encoder.forward(bond_vec_batch)
        # latent_vec_batch : [batch_len,1+state_dim+bond_dim+env_dim,HIDDEN_DIM]
        
        latent_vec_batch = torch.concat((seq_vec_batch,state_vec_batch,bond_vec_batch,env_vec_batch),dim=1)
        # latent_vec_batch : [batch_len,(1+state_dim+bond_dim+env_dim)*HIDDEN_DIM]
        latent_vec_batch = latent_vec_batch.flatten(start_dim=1)

      
        return latent_vec_batch

class Decoder(nn.Module):
    def __init__(self,state_dim,bond_dim,env_dim,hidden_dim,output_dim, dropout,):
        super(Decoder, self).__init__()
        self.input_dim = (1+state_dim+bond_dim+env_dim)* hidden_dim
        self.ffn_1 = nn.Sequential(
            nn.Linear(self.input_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.ffn_2 = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim//4),
            nn.BatchNorm1d(hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.linear = nn.Linear(hidden_dim//4,output_dim)

    def forward(self, latent_vec_batch:torch.Tensor)->torch.Tensor:
        # latent_vec_batch : [batch_len,HIDDEN_DIM]
        x = self.ffn_1.forward(latent_vec_batch)
        x = self.ffn_2.forward(x)
        out = self.linear.forward(x)
        # out : [batch,output_dim]
        return out
    
class Model(nn.Module):
    def __init__(self,config:dict):
        
        super(Model, self).__init__()
        aa_type_count = len(config['seq']['alphabet'])
        pad_index = str(config['seq']['alphabet']).index(str(config['seq']['pad_char']))
        env_dim = len(config['csv']['env_var_col_name'])
        bond_dim = len(config['csv']['bond_var_col_name'])
        state_dim = len(config['csv']['state_var_col_name'])
        self.state_norm = nn.BatchNorm1d(state_dim)
        self.bond_norm = nn.BatchNorm1d(bond_dim)
        self.env_norm = nn.BatchNorm1d(env_dim)
        self.encoder = Encoder(
                                aa_type_count=aa_type_count,
                                pad_index=pad_index,
                                state_dim=state_dim,
                                bond_dim=bond_dim,
                                env_dim=env_dim,
                                hidden_dim=config['model']['hidden_dim'],
                                num_heads=config['model']['num_heads'],
                                dropout=config['model']['dropout'],
                                forward_expansion=config['model']['forward_expansion'],
                                attention_layer_num=config['model']['attention_layer_num'],)
        self.decoder = Decoder(
                                state_dim=state_dim,
                                env_dim=env_dim,
                                bond_dim=bond_dim,
                                output_dim=2,
                                hidden_dim=config['model']['hidden_dim'],
                                dropout=config['model']['dropout'],
                                )

        self.param_dict: dict = config
        self.param_dict['model']['aa_type_count'] = aa_type_count
        self.param_dict['model']['env_dim'] = env_dim
        self.param_dict['model']['bond_dim'] = bond_dim
        self.param_dict['model']['env_dim'] = state_dim
   
        

    
    def forward(self, seq_index_batch:torch.Tensor,
                seq_padding_mask_batch:torch.Tensor,
                bond_index_batch:List[int],
                state_vec_batch:torch.Tensor,
                bond_vec_batch:torch.Tensor,
                env_vec_batch:torch.Tensor)->torch.Tensor:
        # seq_index_batch : [batch_len,max_seq_len]
        # seq_padding_mask_batch : [batch_len,max_seq_len]
        # bond_index_batch : [batch_len,]
        # bond_vec_batch: [batch_len,bond_dim]
        # state_vec_batch: [batch_len,state_dim]
        # env_vec_batch: [batch_len,ENV_DIM]
        state_vec_batch = self.state_norm.forward(state_vec_batch)
        bond_vec_batch = self.bond_norm.forward(bond_vec_batch)
        env_vec_batch = self.env_norm.forward(env_vec_batch)
        # latent_vec_batch : [batch_len,HIDDEN_DIM]
        # attn_weight_dict : dict
        latent_vec_batch = self.encoder.forward(
                                                seq_index_batch=seq_index_batch,
                                                seq_padding_mask_batch=seq_padding_mask_batch,
                                                bond_index_batch=bond_index_batch,
                                                state_vec_batch=state_vec_batch,
                                                bond_vec_batch=bond_vec_batch,
                                                env_vec_batch=env_vec_batch)
        out = self.decoder.forward(latent_vec_batch)
        # out : [batch,2]
        return out
    def __str__(self) -> str:
        param_dict_yaml = yaml.dump(self.param_dict,sort_keys=False)
        model_arch = super().__str__()
        return param_dict_yaml+'\n'+model_arch


def focal_loss(logits, labels,alpha=0.25, gamma=2, reduction="mean"):
    # negative weight = alpha, postive weight = 1-alpha
    ce_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    pt = torch.exp(-ce_loss)
    focal_weight = (alpha * (1-labels) + (1 - alpha) * labels)*((1 - pt) ** gamma)
    loss = focal_weight *  ce_loss
    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return  loss.mean()
    else:
       return loss

