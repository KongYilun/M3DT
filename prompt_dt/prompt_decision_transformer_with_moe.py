# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import torch.nn as nn

import transformers

from .trajectory_gpt2 import GPT2Model

import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_utils import (
    Conv1D,
    PreTrainedModel,
    SequenceSummary,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)


class Router(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super(Router, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2 * hidden_dim)  
        self.fc2 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)  
        self.fc_new = nn.Linear(2 * hidden_dim, 2 * hidden_dim)  
        # self.fc_new_2 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)  
        self.fc3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, num_experts)  
        self.activation = nn.ReLU()  
        # self.ln1=nn.LayerNorm(2 * hidden_dim)
        # self.ln2=nn.LayerNorm(2 * hidden_dim)
        # self.ln3=nn.LayerNorm(2 * hidden_dim)
        # self.ln4=nn.LayerNorm(1 * hidden_dim)
        # self.dropout = nn.Dropout(0.1)
        # print('Initializing Layer Norm')
        # print('Initializing Dropout')
    
    def forward(self, x):
        # h = self.activation(self.ln1(self.fc1(x)))
        # # h = self.dropout(h)
        # h = self.activation(self.ln2(self.fc2(h)))
        # # h = self.dropout(h)
        # h = self.activation(self.ln3(self.fc_new(h)))
        # # h = self.dropout(h)
        # h = self.activation(self.ln4(self.fc3(h)))
        # h = self.dropout(h)
        h = self.activation(self.fc1(x))
        # h = self.dropout(h)
        h = self.activation(self.fc2(h))
        # h = self.dropout(h)
        h = self.activation(self.fc_new(h))
        # h = self.dropout(h)
        # h = self.activation(self.fc_new_2(h))
        h = self.activation(self.fc3(h))
        # h = self.dropout(h)
        return F.softmax(self.fc4(h), dim=-1)#self.fc2(h)
    
class NoisyTopKRouter(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts,top_k):
        super(NoisyTopKRouter, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2 * hidden_dim)  
        self.fc2 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)  
        self.fc_new = nn.Linear(2 * hidden_dim, 2 * hidden_dim)  
        # self.fc_new_2 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)  
        self.fc3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, num_experts)  
        self.activation = nn.ReLU()  
        self.top_k=top_k
        # self.dropout = nn.Dropout(0.1)
        self.noise_linear =nn.Linear(input_dim, num_experts)
        print(f'Top-{top_k} MoE!' )
    
    def forward(self, x):
        h = self.activation(self.fc1(x))
        # h = self.dropout(h)
        h = self.activation(self.fc2(h))
        # h = self.dropout(h)
        h = self.activation(self.fc_new(h))
        # h = self.dropout(h)
        # h = self.activation(self.fc_new_2(h))
        h = self.activation(self.fc3(h))
        # h = self.dropout(h)
        h = self.fc4(h)
        noise_logits = self.noise_linear(x)
        noise=torch.randn_like(h)*F.softplus(noise_logits)
        noisy_logits = h + noise
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices#self.fc2(h)

class SampleLevelRouter(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super(SampleLevelRouter, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, num_experts)  
        self.activation = nn.ReLU()  
    
    def forward(self, x):
        x=x.mean(dim=1)
        h = self.activation(self.fc1(x))
        return F.softmax(self.fc2(h), dim=-1)
    
class Expert(nn.Module):
    def __init__(self, n_state, config):  
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)
    
class TransformerBlockWithExpert(nn.Module):
    def __init__(self, original_block, expert, config):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = original_block.ln_1  # LayerNorm，保持和原始一样
        self.attn = original_block.attn  # Attention模块，保持不变
        self.ln_2 = original_block.ln_2  # LayerNorm，保持不变
        if config.add_cross_attention:
            self.crossattention = original_block.crossattention
            self.ln_cross_attn = original_block.ln_cross_attn
        # 替换掉原有的 MLP
        self.mlp = original_block.mlp
        self.expert = expert#Expert(inner_dim,config)

    def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
    ):
        attn_outputs = self.attn(
            self.ln_1(hidden_states),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + hidden_states

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attn_outputs = self.crossattention(
                self.ln_cross_attn(hidden_states),
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = hidden_states + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        hidden_states=self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        feed_forward_hidden_states_expert = self.expert(hidden_states)

        # residual connection
        hidden_states = hidden_states + feed_forward_hidden_states + feed_forward_hidden_states_expert
        # hidden_states = hidden_states + self.adapter_ln(self.adapter_mlp(hidden_states))

        outputs = [hidden_states] + outputs
        return outputs  # hidden_states, present, (attentions, cross_attentions)

class PromptDTExpert(nn.Module):

    def __init__(
            self,
            pretrained_model,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__()
        self.pretrained_pdt=pretrained_model
        for param in self.pretrained_pdt.parameters():
            param.requires_grad = False
        config = transformers.GPT2Config(vocab_size=1, n_embd=self.pretrained_pdt.hidden_size, **kwargs)
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.experts=[Expert(inner_dim, config) for _ in range(config.n_layer)]
        for i,block in enumerate(self.pretrained_pdt.transformer.h):
            self.pretrained_pdt.transformer.h[i]=TransformerBlockWithExpert(block, self.experts[i], config)
    
    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, prompt=None):
        state_preds, action_preds, return_preds = self.pretrained_pdt.forward(states, actions, rewards, returns_to_go, timesteps, attention_mask, prompt)
        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        action=self.pretrained_pdt.get_action(states, actions, rewards, returns_to_go, timesteps, **kwargs)
        return action
    
    def save_expert(self, expert_id, postfix, folder):
        model_name = '/expert_' + str(expert_id) + postfix
        expert_dict = {f'block_{i}': self.experts[i].state_dict() for i in range(len(self.experts))}
        torch.save(expert_dict,folder+model_name)  # model save
    
    def load_expert(self, expert_path):
        params=torch.load(expert_path)
        for i in range(len(self.experts)):
            self.experts[i].load_state_dict(params[f'block_{i}'])

class MoE(nn.Module):
    def __init__(self, input_dim, num_experts, experts):
        super().__init__()
        self.router=Router(input_dim, 2* input_dim, num_experts)
        self.experts=experts
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False
    def forward(self, x):
        route_weight=self.router(x)
        outputs=torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_output=expert(x)
            outputs += route_weight[:,:,i].unsqueeze(-1)*expert_output
        return outputs
    
class ParalMoE(nn.Module):
    def __init__(self, input_dim, num_experts, experts):
        super().__init__()
        self.router=Router(input_dim, 2* input_dim, num_experts)
        self.experts=experts
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False
    def forward(self, x):
        route_weight=self.router(x)
        outputs=torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_output=expert(x)
            outputs += route_weight[:,:,i].unsqueeze(-1)*expert_output
        return outputs
    
class OriginalTopKMoE(nn.Module):
    def __init__(self, input_dim, num_experts, experts, k):
        super().__init__()
        self.router=NoisyTopKRouter(input_dim, 2 * input_dim, num_experts, k)
        self.experts=experts
        self.top_k=k
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False   
    def forward(self, x):
        topk_values, topk_indices=self.router(x)
        # topk_values, topk_indices = torch.topk(route_weight, self.top_k, dim=-1)
        # outputs=torch.zeros_like(x)
        expert_outputs=torch.stack([expert(x) for expert in self.experts], dim=2)
        # topk_values, topk_indices = torch.topk(route_weight, self.top_k, dim=-1)
        expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, x.size(2))
        selected_expert_outputs = expert_outputs.gather(2, expanded_indices)
        outputs = (topk_values.unsqueeze(-1) * selected_expert_outputs).sum(dim=2)
        # for i, expert in enumerate(self.experts):
        #     expert_output=expert(x)
        #     outputs += route_weight[:,:,i].unsqueeze(-1)*expert_output
        return outputs
    
class TopKMoE(nn.Module):
    def __init__(self, input_dim, num_experts, experts, k):
        super().__init__()
        self.router=NoisyTopKRouter(input_dim, 2 * input_dim, num_experts, k)
        self.experts=experts
        self.top_k=k
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False
    def forward(self, x):
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))
        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                final_output[expert_mask] += weighted_output.squeeze(1)
        return final_output

class SampleLevelMoE(nn.Module):
    def __init__(self, input_dim, num_experts, experts):
        super().__init__()
        self.router=SampleLevelRouter(input_dim, 2 * input_dim, num_experts)
        self.experts=experts
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False
    def forward(self, x):
        route_weight=self.router(x)
        outputs=torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_output=expert(x)
            outputs += route_weight[:,i].unsqueeze(-1).unsqueeze(-1)*expert_output
        return outputs

class TransformerBlockWithMOE(nn.Module):
    def __init__(self, original_block, moe, config):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = original_block.ln_1  # LayerNorm，保持和原始一样
        self.attn = original_block.attn  # Attention模块，保持不变
        self.ln_2 = original_block.ln_2  # LayerNorm，保持不变
        if config.add_cross_attention:
            self.crossattention = original_block.crossattention
            self.ln_cross_attn = original_block.ln_cross_attn
        # 替换掉原有的 MLP
        self.mlp = original_block.mlp
        self.moe = moe#Expert(inner_dim,config)

    def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
    ):
        attn_outputs = self.attn(
            self.ln_1(hidden_states),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + hidden_states

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attn_outputs = self.crossattention(
                self.ln_cross_attn(hidden_states),
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = hidden_states + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        hidden_states=self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        feed_forward_hidden_states_moe = self.moe(hidden_states)

        # residual connection
        hidden_states = hidden_states + feed_forward_hidden_states + feed_forward_hidden_states_moe
        # hidden_states = hidden_states + self.adapter_ln(self.adapter_mlp(hidden_states))

        outputs = [hidden_states] + outputs
        return outputs  # hidden_states, present, (attentions, cross_attentions)

class PromptDTMoE(nn.Module):

    def __init__(
            self,
            pretrained_model,
            experts_params,
            top_k,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__()
        self.expert_num=len(experts_params)
        self.pretrained_pdt=pretrained_model
        for param in self.pretrained_pdt.parameters():
            param.requires_grad = False
        config = transformers.GPT2Config(vocab_size=1, n_embd=self.pretrained_pdt.hidden_size, **kwargs)
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd

        self.experts=nn.ModuleList([nn.ModuleList([Expert(inner_dim, config) for _ in range(self.expert_num)]) for _ in range(config.n_layer)])
        for i in range(self.expert_num):
            for j in range(config.n_layer):
                self.experts[j][i].load_state_dict(experts_params[i][f'block_{j}'])
        for param in self.experts.parameters():
            param.requires_grad = False

        self.moes=nn.ModuleList([MoE(config.n_embd, self.expert_num, self.experts[i]) for i in range(config.n_layer)])
        for i,block in enumerate(self.pretrained_pdt.transformer.h):
            self.pretrained_pdt.transformer.h[i]=TransformerBlockWithMOE(block, self.moes[i], config)
    
    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, prompt=None):
        state_preds, action_preds, return_preds = self.pretrained_pdt.forward(states, actions, rewards, returns_to_go, timesteps, attention_mask, prompt)
        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        action=self.pretrained_pdt.get_action(states, actions, rewards, returns_to_go, timesteps, **kwargs)
        return action
    
    def save_moe(self, postfix, folder):
        model_name = '/moe_' + postfix
        moe_dict = {f'block_{i}': self.moes[i].state_dict() for i in range(len(self.moes))}
        torch.save(moe_dict,folder+model_name)  # model save
        print('model saved to ', folder+model_name)
    
    def load_moe(self, router_path):
        params=torch.load(router_path)
        for i in range(len(self.moes)):
            self.moes[i].load_state_dict(params[f'block_{i}'])
        
