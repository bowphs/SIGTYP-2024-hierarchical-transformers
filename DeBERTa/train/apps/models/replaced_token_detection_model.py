#
# Author: penhe@microsoft.com
# Date: 04/25/2021
#
""" Replaced token detection model for representation learning
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import torch
import torch.nn as nn
from ...deberta import *
from .masked_language_model import MaskedLanguageModel

__all__ = ['LMMaskPredictionHead', 'ReplacedTokenDetectionModel', 'RTDCombinedModel']

class LMMaskPredictionHead(nn.Module):
  """ Replaced token prediction head
  """
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.transform_act_fn = ACT2FN[config.hidden_act] \
      if isinstance(config.hidden_act, str) else config.hidden_act
    self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
    self.classifier = nn.Linear(config.hidden_size, 1)

  def forward(self, hidden_states, input_ids, input_mask, lm_labels=None):
    # b x d
    ctx_states = hidden_states[:,0,:]
    seq_states = self.LayerNorm(ctx_states.unsqueeze(-2) + hidden_states)
    seq_states = self.dense(seq_states)
    seq_states = self.transform_act_fn(seq_states)

    # b x max_len
    logits = self.classifier(seq_states).squeeze(-1)
    mask_loss = torch.tensor(0).to(logits).float()
    mask_labels = None
    if lm_labels is not None:
      mask_logits = logits.view(-1)
      _input_mask = input_mask.view(-1).to(mask_logits)
      input_idx = (_input_mask>0).nonzero().view(-1)
      mask_labels = ((lm_labels>0) & (lm_labels!=input_ids)).view(-1)
      mask_labels = torch.gather(mask_labels.to(mask_logits), 0, input_idx)
      mask_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
      mask_logits = torch.gather(mask_logits, 0, input_idx).float()
      mask_loss = mask_loss_fn(mask_logits, mask_labels)
    return mask_logits, mask_labels, mask_loss


class ReplacedTokenDetectionModel(NNModule):
  """ RTD with DeBERTa
  """
  def __init__(self, config, *wargs, **kwargs):
    super().__init__(config)
    self.deberta = DeBERTa(config)

    self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
    self.position_buckets = getattr(config, 'position_buckets', -1)
    if self.max_relative_positions <1:
      self.max_relative_positions = config.max_position_embeddings
    self.mask_predictions = LMMaskPredictionHead(self.deberta.config)
    self.apply(self.init_weights)

  def forward(self, input_ids, input_mask=None, labels=None, position_ids=None, attention_mask=None):
    device = list(self.parameters())[0].device
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    type_ids = None
    lm_labels = labels.to(device)
    if attention_mask is not None:
      attention_mask = attention_mask.to(device)
    else:
      attention_mask = input_mask

    encoder_output = self.deberta(input_ids, input_mask, type_ids, output_all_encoded_layers=True, position_ids=position_ids)
    encoder_layers = encoder_output['hidden_states']
    (mask_logits, mask_labels, mask_loss) = self.mask_predictions(encoder_layers[-1], input_ids, input_mask, lm_labels)

    return {
      'logits' : mask_logits,
      'labels' : mask_labels,
      'loss' : mask_loss.float(),
    }


class RTDCombinedModel(NNModule):
  def __init__(self, config, tokenizer=None, *wargs, **kwargs):
    super().__init__(config)
    gen_config = config.generator
    disc_config = config.discriminator
    gen_config.token_format = config.token_format
    disc_config.token_format = config.token_format
    self.config = config

    if tokenizer and gen_config.vocab_size != len(tokenizer.vocab):
      gen_config.vocab_size = len(tokenizer.vocab)
    if tokenizer and disc_config.vocab_size != len(tokenizer.vocab):
      disc_config.vocab_size = len(tokenizer.vocab)

    self.generator = MaskedLanguageModel(gen_config)
    self.discriminator = ReplacedTokenDetectionModel(disc_config)

    self.generator._register_load_state_dict_pre_hook(self._pre_load_hook)
    self.discriminator._register_load_state_dict_pre_hook(self._pre_load_hook)

    self.share_embedding = getattr(config, 'embedding_sharing', "none").lower()
    if self.share_embedding == 'gdes': # Gradient-disentangled weight/embedding sharing
      word_bias = torch.zeros_like(self.discriminator.deberta.embeddings.word_embeddings.weight)
      word_bias = torch.nn.Parameter(word_bias)
      position_bias = torch.zeros_like(self.discriminator.deberta.embeddings.position_embeddings.weight)
      position_bias = torch.nn.Parameter(position_bias)
      delattr(self.discriminator.deberta.embeddings.word_embeddings, 'weight')
      self.discriminator.deberta.embeddings.word_embeddings.register_parameter('_weight', word_bias)
      delattr(self.discriminator.deberta.embeddings.position_embeddings, 'weight')
      self.discriminator.deberta.embeddings.position_embeddings.register_parameter('_weight', position_bias)
    self.register_discriminator_fw_hook()

  def _pre_load_hook(self, state_dict, prefix, local_metadata, strict,
      missing_keys, unexpected_keys, error_msgs):
    bert_prefix = prefix + 'bert.'
    deberta_prefix = prefix + 'deberta.'
    for k in list(state_dict.keys()):
      if k.startswith(bert_prefix):
        nk = deberta_prefix + k[len(bert_prefix):]
        value = state_dict[k]
        del state_dict[k]
        state_dict[nk] = value

  def forward(self, **kwargs):
    return self.generator_fw(**kwargs)

  def discriminator_fw(self, **kwargs):
    return self.discriminator(**kwargs)

  def generator_fw(self, **kwargs):
    return self.generator(**kwargs)

  def topk_sampling(self, logits, topk = 1, start=0, temp=1):
    top_p = torch.nn.functional.softmax(logits/temp, dim=-1)
    topk = max(1, topk)
    next_tokens = torch.multinomial(top_p, topk)
    return next_tokens, top_p
  
  def make_electra_data(self, input_data, temp=1, rand=None):
    new_data = input_data.copy()
    if rand is None:
      rand = random
    gen = self.generator_fw(**new_data)
    lm_logits = gen['logits']
    lm_labels = input_data['labels']
    lm_loss = gen['loss']
    mask_index = (lm_labels.view(-1)>0).nonzero().view(-1)
    topk_labels, top_p = self.topk_sampling(lm_logits, topk=1, temp=temp)
    
    top_ids = torch.zeros_like(lm_labels.view(-1))
    top_ids.scatter_(index=mask_index, src=topk_labels.view(-1).int(), dim=-1)
    top_ids = top_ids.view(lm_labels.size())
    new_ids = torch.where(lm_labels>0, top_ids, input_data['input_ids'])
    new_data['input_ids'] = new_ids.detach()
    return new_data, lm_loss, gen

  def register_discriminator_fw_hook(self, *wargs):
    def fw_hook(module, *inputs):
      if self.share_embedding == 'gdes': # Gradient-disentangled weight/embedding sharing
        g_w_ebd = self.generator.deberta.embeddings.word_embeddings
        d_w_ebd = self.discriminator.deberta.embeddings.word_embeddings
        self._set_param(d_w_ebd, 'weight', g_w_ebd.weight.detach() + d_w_ebd._weight)

        g_p_ebd = self.generator.deberta.embeddings.position_embeddings
        d_p_ebd = self.discriminator.deberta.embeddings.position_embeddings
        self._set_param(d_p_ebd, 'weight', g_p_ebd.weight.detach() + d_p_ebd._weight)
      elif self.share_embedding == 'es': # vallina embedding sharing
        g_w_ebd = self.generator.deberta.embeddings.word_embeddings
        d_w_ebd = self.discriminator.deberta.embeddings.word_embeddings
        self._set_param(d_w_ebd, 'weight', g_w_ebd.weight)

        g_p_ebd = self.generator.deberta.embeddings.position_embeddings
        d_p_ebd = self.discriminator.deberta.embeddings.position_embeddings
        self._set_param(d_p_ebd, 'weight', g_p_ebd.weight)
      return None
    self.discriminator.register_forward_pre_hook(fw_hook)

  @staticmethod
  def _set_param(module, param_name, value):
    if hasattr(module, param_name):
      delattr(module, param_name)
    module.register_buffer(param_name, value)