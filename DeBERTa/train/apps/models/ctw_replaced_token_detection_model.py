import torch
import random
from ...deberta import *
from .ctw_masked_language_model import CharToWord_MaskedLanguageModel

__all__ = ['CharToWord_ReplacedTokenDetectionModel', 'CharToWord_RTDCombinedModel']

class CharToWord_ReplacedTokenDetectionModel(NNModule):
  """ RTD with CharToWord_DeBERTa
  """
  def __init__(self, config, *wargs, **kwargs):
    super().__init__(config)
    self.ctw_deberta = CharToWord_DeBERTa(config)
    self.mask_predictions = CharToWord_ReplacedTokenDetectionHead(self.ctw_deberta.config)
    self.apply(self.init_weights)

  def forward(self, input_ids, char_input_mask, word_input_mask, char_position_ids=None, word_position_ids=None, labels=None):
    device = list(self.parameters())[0].device

    char_input_mask = char_input_mask.to(device)
    labels = labels.to(device) if labels is not None else None
    if char_position_ids is not None:
      char_position_ids = char_position_ids.to(device)
    if word_position_ids is not None:
      word_position_ids = word_position_ids.to(device)

    encoder_output = self.ctw_deberta(input_ids, char_input_mask,
      word_input_mask, char_position_ids, word_position_ids,
      output_all_encoded_layers=False)
    logits = self.mask_predictions(encoder_output, self.ctw_deberta.intra_word_encoder.get_rel_embedding())

    #mask_loss = torch.tensor(0).to(logits).float()
    mask_logits = logits.view(-1)
    input_mask = char_input_mask.view(-1).to(mask_logits)
    input_idx = (input_mask > 0).nonzero().view(-1)
    mask_labels = ((labels > 0) & (labels != input_ids.view(labels.shape))).view(-1)
    mask_labels = torch.gather(mask_labels.to(mask_logits), 0, input_idx)
    mask_logits = torch.gather(mask_logits, 0, input_idx).float()
    mask_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    mask_loss = mask_loss_fn(mask_logits, mask_labels)
    return {
      'logits' : mask_logits,
      'labels' : mask_labels,
      'loss' : mask_loss.float(),
    }


class CharToWord_RTDCombinedModel(NNModule):
  def __init__(self, config, tokenizer=None, *wargs, **kwargs):
    super().__init__(config)
    gen_config = config.generator
    disc_config = config.discriminator
    gen_config.token_format = config.token_format
    gen_config.max_word_length = config.max_word_length
    disc_config.token_format = config.token_format
    disc_config.max_word_length = config.max_word_length
    self.config = config

    if tokenizer and gen_config.vocab_size != len(tokenizer.vocab):
      gen_config.vocab_size = len(tokenizer.vocab)
    if tokenizer and disc_config.vocab_size != len(tokenizer.vocab):
      disc_config.vocab_size = len(tokenizer.vocab)

    self.generator = CharToWord_MaskedLanguageModel(gen_config)
    self.discriminator = CharToWord_ReplacedTokenDetectionModel(disc_config)

    self.generator._register_load_state_dict_pre_hook(self._pre_load_hook)
    self.discriminator._register_load_state_dict_pre_hook(self._pre_load_hook)

    self.share_embedding = getattr(config, 'embeddichar_experiments/configs/ctw_tiny.jsonng_sharing', "none").lower()
    if self.share_embedding == 'gdes': # Gradient-disentangled weight/embedding sharing
      char_bias = torch.zeros_like(self.discriminator.deberta.char_embeddings.weight)
      char_bias = torch.nn.Parameter(char_bias)
      position_bias = torch.zeros_like(self.discriminator.deberta.word_position_embeddings.weight)
      position_bias = torch.nn.Parameter(position_bias)
      delattr(self.discriminator.deberta.char_embeddings, 'weight')
      self.discriminator.deberta.char_embeddings.register_parameter('_weight', char_bias)
      delattr(self.discriminator.deberta.word_position_embeddings, 'weight')
      self.discriminator.deberta.word_position_embeddings.register_parameter('_weight', position_bias)
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
    mask_index = (lm_labels.view(-1) > 0).nonzero().view(-1)
    topk_labels, top_p = self.topk_sampling(lm_logits, topk=1, temp=temp)
    
    top_ids = torch.zeros_like(lm_labels.view(-1))
    top_ids.scatter_(index=mask_index, src=topk_labels.view(-1).long(), dim=-1)
    top_ids = top_ids.view(lm_labels.shape)
    old_shape = input_data['input_ids'].shape
    input_ids = input_data['input_ids'].view(top_ids.shape)
    mask = lm_labels > 0
    new_ids = torch.where(mask, top_ids, input_ids)
    new_data['input_ids'] = new_ids.detach().reshape(old_shape)
    return new_data, lm_loss, gen

  def register_discriminator_fw_hook(self, *wargs):
    def fw_hook(module, *inputs):
      if self.share_embedding == 'gdes': # Gradient-disentangled weight/embedding sharing
        g_w_ebd = self.generator.deberta.char_embeddings
        d_w_ebd = self.discriminator.deberta.char_embeddings
        self._set_param(d_w_ebd, 'weight', g_w_ebd.weight.detach() + d_w_ebd._weight)

        g_p_ebd = self.generator.deberta.word_position_embeddings
        d_p_ebd = self.discriminator.deberta.word_position_embeddings
        self._set_param(d_p_ebd, 'weight', g_p_ebd.weight.detach() + d_p_ebd._weight)
      elif self.share_embedding == 'es': # vallina embedding sharing
        g_w_ebd = self.generator.deberta.char_embeddings
        d_w_ebd = self.discriminator.deberta.char_embeddings
        self._set_param(d_w_ebd, 'weight', g_w_ebd.weight)

        g_p_ebd = self.generator.deberta.word_position_embeddings
        d_p_ebd = self.discriminator.deberta.word_position_embeddings
        self._set_param(d_p_ebd, 'weight', g_p_ebd.weight)
      return None
    self.discriminator.register_forward_pre_hook(fw_hook)

  @staticmethod
  def _set_param(module, param_name, value):
    if hasattr(module, param_name):
      delattr(module, param_name)
    module.register_buffer(param_name, value)