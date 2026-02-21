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
from transformers.activations import ACT2FN
import torch.nn.init as init
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

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
class FeedForward(nn.Module):
    def __init__(self, config:MiniMindConfig):
        super.__init__()
        if config.intermediate_size is None:
            intermediate_size = int((config.hidden_size * 8 / 3))
            config.intermediate_size=64*((config.intermediate_size+64 - 1)// 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(self.up_proj(self.act_fn(self.gate_proj(x)) * self.down_proj(x)))
    
class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        scores=logits.softmax(dim=-1)

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            dominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / dominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux=scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len*aux_topk, device=hidden_states.device)).div_(
                                    seq_len * aux_topk / self.n_routed_experts
                                )
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss
    
class MOEFeedForwad(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts>0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])
        
    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)
            for i, expert in self.experts:
                expert_out = expert(x[flat_topk_idx==i])
                if expert_out.shape[0] > 0:
                    y[flat_topk_idx==i] = expert_out.to(y.dtype)
                else:
                    y[flat_topk_idx==i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForwad(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_norm(hidden_states))
        return hidden_states, present_key_value
    
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim = config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, "layers"): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_values) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_values = past_key_values,
                use_cache = use_cache,
                attention_mask = attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForwad)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss