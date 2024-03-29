# coding=utf-8
# Copyright 2020 Microsoft and the Hugging Face Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch CharToWord DeBERTa model."""

from collections.abc import Sequence
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import copy

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm

from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout, DebertaV2Layer, ConvLayer, build_relative_position
from transformers.modeling_outputs import BaseModelOutput, ModelOutput, MaskedLMOutput, TokenClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, AutoModelForMaskedLM, AutoTokenizer, DebertaV2Tokenizer, DebertaV2TokenizerFast
from .configuration_ctw_deberta import CharToWordDebertaConfig

logger = logging.get_logger(__name__)


@dataclass
class CharToWordDebertaBaseModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor] = None
    attentions: Tuple[torch.FloatTensor] = None

    word_embeds: torch.FloatTensor = None
    initial_embeds: torch.FloatTensor = None
    initial_word_embeds: torch.FloatTensor = None
    intra_word_mask: torch.LongTensor = None
    char_embeds: torch.LongTensor = None
    input_shape: Tuple[int, int, int, int] = None


# Copied from DebertaV2Encoder and modified to add shared relative embeddings support
class CharToWordDebertaEncoder(nn.Module):
    """Modified BertEncoder with relative position bias support"""

    def __init__(self, config, shared_rel_embeddings=False):
        super().__init__()

        self.layer = nn.ModuleList([DebertaV2Layer(config) for _ in range(config.num_hidden_layers)])
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings

            self.position_buckets = getattr(config, "position_buckets", -1)
            pos_ebd_size = self.max_relative_positions * 2

            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2

            if not shared_rel_embeddings:
                self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)

        self.norm_rel_ebd = [x.strip() for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")]

        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=True)

        self.conv = ConvLayer(config) if getattr(config, "conv_kernel_size", 0) > 0 else None
        self.gradient_checkpointing = False

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        if rel_embeddings is not None and ("layer_norm" in self.norm_rel_ebd):
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(
                q,
                hidden_states.size(-2),
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
                device=hidden_states.device,
            )
        return relative_pos

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=True,
        relative_embeddings=None,
    ):
        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            input_mask = attention_mask.sum(-2) > 0
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        rel_embeddings = relative_embeddings if relative_embeddings is not None else self.get_rel_embedding()
        output_states = next_kv
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)

            if self.gradient_checkpointing and self.training:
                output_states = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    next_kv,
                    attention_mask,
                    query_states,
                    relative_pos,
                    rel_embeddings,
                    output_attentions,
                )
            else:
                output_states = layer_module(
                    next_kv,
                    attention_mask,
                    query_states=query_states,
                    relative_pos=relative_pos,
                    rel_embeddings=rel_embeddings,
                    output_attentions=output_attentions,
                )

            if output_attentions:
                output_states, att_m = output_states

            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)

            if query_states is not None:
                query_states = output_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = output_states

            if output_attentions:
                all_attentions = all_attentions + (att_m,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (output_states,)

        if not return_dict:
            return tuple(v for v in [output_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=output_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


# Copied from transformers.models.deberta.modeling_deberta.DebertaPreTrainedModel with Deberta->CharToWordDeberta
class CharToWordDebertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CharToWordDebertaConfig
    base_model_prefix = "ctw_deberta"
    _keys_to_ignore_on_load_unexpected = ["position_embeddings"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class CharToWordDebertaModel(CharToWordDebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.char_embeddings = torch.nn.Embedding(config.vocab_size, config.intra_word_encoder.hidden_size, padding_idx=0)
        self.char_embedding_layer_norm = LayerNorm(config.intra_word_encoder.hidden_size, config.intra_word_encoder.layer_norm_eps)
        self.char_embedding_dropout = StableDropout(config.intra_word_encoder.hidden_dropout_prob)

        self.intra_word_encoder = CharToWordDebertaEncoder(config.intra_word_encoder)
        self.inter_word_encoder = CharToWordDebertaEncoder(config.inter_word_encoder)

        self.z_steps = 0
        self.config = config
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.char_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.char_embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        char_input_mask: Optional[torch.Tensor] = None,
        word_input_mask: Optional[torch.Tensor] = None,
        char_position_ids: Optional[torch.Tensor] = None, # Not used (yet)
        word_position_ids: Optional[torch.Tensor] = None, # Not used (yet)
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        combined_word_embeddings: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_embeds = self.char_embeddings(input_ids)
        input_embeds = self.char_embedding_layer_norm(input_embeds)
        input_embeds = input_embeds * char_input_mask.unsqueeze(-1).to(input_embeds) # MaskedLayerNorm
        input_embeds = self.char_embedding_dropout(input_embeds)

        batch_size, num_word, num_char, hidden_size = input_embeds.shape

        # reshape to attend to intra-word tokens rather than full sequence
        input_embeds = input_embeds.reshape(batch_size * num_word, num_char, hidden_size)
        intra_word_mask = char_input_mask.reshape(batch_size * num_word, num_char)
        intra_word_output = self.intra_word_encoder(
            input_embeds,
            intra_word_mask,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        initial_embeds = intra_word_output.last_hidden_state

        # extract [WORD_CLS] embeddings, which are always at the beginning of each word
        initial_word_embeds = initial_embeds[:,0,:]

        # reshape and extract contextualized inter-word representation
        word_embeds = initial_word_embeds.reshape(batch_size, num_word, hidden_size)
        inter_word_output = self.inter_word_encoder(
            word_embeds,
            word_input_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        if combined_word_embeddings:
            initial_word_embeds = initial_word_embeds.reshape(batch_size, num_word, hidden_size)
            contextual_word_embeds = inter_word_output.last_hidden_state
            combined_word_embeds = torch.cat([initial_word_embeds, contextual_word_embeds], dim=2)
            last_hidden_state = combined_word_embeds
        else:
            last_hidden_state = inter_word_output.last_hidden_state

        return CharToWordDebertaBaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=inter_word_output.hidden_states if output_hidden_states else None,
            attentions=inter_word_output.attentions if output_attentions else None,
            word_embeds=inter_word_output.last_hidden_state,
            initial_embeds=initial_embeds,
            initial_word_embeds=initial_word_embeds,
            intra_word_mask=intra_word_mask,
            char_embeds=input_embeds,
            input_shape=(batch_size, num_word, num_char, hidden_size),
        )


class CharToWordDebertaForMaskedLM(CharToWordDebertaPreTrainedModel):
    _tied_weights_keys = ["cls.decoder.weight", "cls.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        # NOTE: This property name must match "base_model_prefix" in the base class
        self.ctw_deberta = CharToWordDebertaModel(config)
        self.cls = CharToWordDebertaLMPredictionHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.decoder = new_embeddings

    # Copied from transformers.models.deberta.modeling_deberta.DebertaForMaskedLM.forward with Deberta->DebertaV2
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        char_input_mask: Optional[torch.Tensor] = None,
        word_input_mask: Optional[torch.Tensor] = None,
        char_position_ids: Optional[torch.Tensor] = None, # Not used (yet)
        word_position_ids: Optional[torch.Tensor] = None, # Not used (yet)
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, num_words, max_chars_per_word)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.ctw_deberta(
            input_ids,
            char_input_mask=char_input_mask,
            word_input_mask=word_input_mask,
            char_position_ids=char_position_ids,
            word_position_ids=word_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            combined_word_embeddings=False,
        )

        prediction_scores = self.cls(outputs)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CharToWordDebertaLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        intra_word_encoder_config = copy.copy(config.intra_word_encoder)
        intra_word_encoder_config.num_hidden_layers = 1
        self.intra_word_encoder = CharToWordDebertaEncoder(intra_word_encoder_config, shared_rel_embeddings=True)
        self.residual_word_embedding = getattr(config, 'residual_word_embedding', False)

        if getattr(config, "tie_word_embeddings", True):
            # The output weights are the same as the input embeddings, but there is
            # an output-only bias for each token.
            self.decoder = nn.Linear(config.intra_word_encoder.hidden_size, config.vocab_size, bias=False)
            self.bias = nn.Parameter(torch.zeros(config.vocab_size))
            # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
            self.decoder.bias = self.bias
        else:
            self.decoder = nn.Linear(config.intra_word_encoder.hidden_size, config.vocab_size)

    def forward(self, base_model_output: CharToWordDebertaBaseModelOutput, rel_embeddings: torch.FloatTensor=None):
        batch_size, num_word, num_char, hidden_size = base_model_output.input_shape
        word_embeds = word_embeds.reshape(batch_size * num_word, 1, hidden_size)

        if self.residual_word_embedding:
          # residual connection between initial word embeddings and contextual word embeddings as mentioned in the paper (section A.3)
          word_embeds += base_model_output.initial_word_embeds.unsqueeze(1)

        # concatenate to restore the character-level token sequence
        char_embeds = torch.cat([base_model_output.word_embeds, base_model_output.initial_embeds[:,1:,:]], dim=1)
        intra_word_output = self.intra_word_encoder(
            char_embeds,
            base_model_output.intra_word_mask,
            relative_embeddings=rel_embeddings,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )

        char_logits = self.decoder(intra_word_output.last_hidden_state)
        batch_size, num_word, num_char, hidden_size = base_model_output.input_shape
        char_logits = char_logits.reshape(batch_size, num_word * num_char, -1)
        return char_logits


# Copied from transformers.models.deberta.modeling_deberta.DebertaForTokenClassification with Deberta->CharToWordDeberta
class CharToWordDebertaForTokenClassification(CharToWordDebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.ctw_deberta = CharToWordDebertaModel(config)
        self.dropout = nn.Dropout(config.inter_word_encoder.hidden_dropout_prob)
        self.classifier = nn.Linear(config.inter_word_encoder.hidden_size*2, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        char_input_mask: Optional[torch.Tensor] = None,
        word_input_mask: Optional[torch.Tensor] = None,
        char_position_ids: Optional[torch.Tensor] = None,
        word_position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.ctw_deberta(
            input_ids,
            char_input_mask=char_input_mask,
            word_input_mask=word_input_mask,
            char_position_ids=char_position_ids,
            word_position_ids=word_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            combined_word_embeddings=True,
        )

        combined_word_embeds = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(combined_word_embeds)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


AutoConfig.register("ctw_deberta", CharToWordDebertaConfig)
AutoModel.register(CharToWordDebertaConfig, CharToWordDebertaModel)
AutoModelForTokenClassification.register(CharToWordDebertaConfig, CharToWordDebertaForTokenClassification)
AutoModelForMaskedLM.register(CharToWordDebertaConfig, CharToWordDebertaForMaskedLM)
AutoTokenizer.register(CharToWordDebertaConfig, DebertaV2Tokenizer, DebertaV2TokenizerFast)