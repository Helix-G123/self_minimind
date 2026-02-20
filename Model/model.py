from transformers import PretrainedConfig
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F

class RMSNorm(nn.Module):

    def __init__(self, dim:int, eps:float=1e-5):
        super().__init__()
        self.dim=dim
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x*torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)
    
    def forward(self, x):
        return self._norm(x.float())*self.weight.type_as(x)



def precompute_freqs_cis(dim:int, end:int=int(32*1024),rope_base:float=1e6, rope_scaling:Optional[dict]=None):
    #RoPE
    
    freqs=1.0/(rope_base**((torch.arange(0, dim, 2, dtype=torch.float32))/dim))

    if rope_scaling:
        orig_max, factor, beta_fast, beta_slow=(
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4),
            rope_scaling.get("beta_slow", 1.0)
        )
        if end/orig_max>1.0:
        #计算corr_dim
            corr_dim=next((i for i in range(dim//2) if 2*math.pi/freqs[i]>orig_max),dim//2)

        #计算power
            power=torch.arange(0, dim//2, device=freqs.device, dtype=torch.float32)/max(dim//2-1, 1)
        #计算beta
            beta=beta_slow+(beta_fast-beta_slow)*power

        #计算scale
            scale=torch.where(
                torch.arange(0, dim//2, device=freqs.device)<corr_dim,
                (beta*factor-beta+1)/(beta*factor),
                1.0/factor
            )
        #应用scale
            freqs=freqs*scale
    t=torch.arange(end, device=freqs.device)
    freqs=torch.outer(freqs, t)
    freqs_cos=torch.cos(freqs).repeat_interleave(2, dim=-1)
    freqs_sin=torch.sin(freqs).repeat_interleave(2, dim=-1)
    #返回一个cos和sin
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat([-x[...,x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)
    q_embed=(q*cos.unsqueeze(unsqueeze_dim))+rotate_half(q)*sin.unsqueeze(unsqueeze_dim)
    k_embed=(k*cos.unsqueeze(unsqueeze_dim))+rotate_half(k)*sin.unsqueeze(unsqueeze_dim)
    return q_embed, k_embed

def repeat_kv(x:torch.Tensor, n_rep:int) -> torch.Tensor:
    batch_size, slen, num_key_value_heads, head_dim=x.shape
    if n_rep==1:
        return x
    return (
        x.unsqueeze(3).expand(batch_size, slen, num_key_value_heads, n_rep, head_dim).reshape(batch_size, slen, num_key_value_heads*n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args:MiniMindConfig):
        super().__init__()
        self.num_key_value_heads=args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % args.num_key_value_heads==0
        self.n_local_head=args.num_attention_heads
        self.n_local_kv_heads=self.num_key_value_heads
        self.n_rep=self.n_local_head//self.n_local_kv_heads
        self.head_dim=args.hidden_size//args.num_attention_heads
        self.q_proj=nn.Linear(args.hidden_size, self.head_dim * args.num_attention_heads, bias=False)
        self.k_proj=nn.Linear(args.hidden_size, self.head_dim * args.num_attention_heads, bias=False)
        self.v_proj=nn.Linear(args.hidden_size, self.head_dim * args.num_attention_heads, bias=False)
        self.o_proj=nn.Linear(self.head_dim * args.num_attention_heads, args.hidden_size, bias=False)
        self.attn_dropout=nn.Dropout(args.dropout)
        self.resid_dropout=nn.Dropout(args.dropout)
        self.dropout=args.dropout
        self.flash=hasattr(torch.nn.functional, "scaled_dot_product_attention") and args.flash_attn

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                paste_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None
                ):
        batch_size, seq_len=x.shape
        xq, xk, xv=self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq=xq.reshape(batch_size, seq_len, self.n_local_head, self.head_dim)
        xk=xk.reshape(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)
        xv=xv.reshape(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin=position_embeddings
        xq, xk=apply_rotary_pos_emb(xq, xk, cos, sin)

        if paste_key_value:
            xk=torch.cat([paste_key_value[0], xk], dim=1)
            xv=torch.cat([paste_key_value[1], xv], dim=1)
        paste_key_value=[xk, xv] if use_cache else None

        xq, xk, xv=(
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if self.flash and (seq_len>1) and (paste_key_value is None) and (attention_mask is None or torch.all(attention_mask==1)):
            output=F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores=(xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)

            if attention_mask:
                extended_attention_mask=attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask=(1.0-extended_attention_mask)* -1e9
                scores=scores+extended_attention_mask

            scores=F.softmax(scores.float(), dim=-1).type_as(xq)
            scores=self.attn_dropout(scores)
            scores=scores @ xv
        
        output=output.transpose(1,2).reshape(batch_size, seq_len, -1)
        output=self.resid_dropout(self.o_proj(output))
        return output, paste_key_value