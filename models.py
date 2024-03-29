from transformers import RobertaPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaConfig, RobertaModel
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput
from DeBERTa.huggingface import *
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


class MultitaskRobertaForTokenClassification(RobertaPreTrainedModel):
    config_class = RobertaConfig

    def __init__(self, config, labels_dict, id2components):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.labels_dict = labels_dict
        self.classifiers = nn.ModuleDict(
            {
                key: nn.Linear(self.roberta.config.hidden_size, len(labels))
                for key, labels in labels_dict.items()
            }
        )

        self.id2component_tensor = torch.stack(
            [id2components[a] for a in range(len(id2components))]
            + [torch.full((len(labels_dict),), -100)],
            dim=0,
        ).to(device)
        self.num_classes = self.id2component_tensor.shape[0]
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        sequence_output = self.dropout(outputs[0])
        logits_task_dict = {
            task: head(sequence_output) for task, head in self.classifiers.items()
        }

        if labels is not None:
            labels[labels == -100] = self.num_classes - 1
            labels_decomposed = torch.stack(
                [
                    torch.index_select(self.id2component_tensor, 0, label)
                    for label in labels
                ]
            ).to(device)
            loss_fct = nn.CrossEntropyLoss()
            losses = []
            for idx, (task, logits) in enumerate(logits_task_dict.items()):
                pred = logits.view(-1, len(self.labels_dict[task]))
                gold = labels_decomposed[:, :, idx].view(-1)
                losses.append(loss_fct(pred, gold))
            loss = sum(losses) / len(losses)
        output = TokenClassifierOutput(
            loss=loss,
            logits=logits_task_dict,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return output


class MultitaskDebertaForTokenClassification(DebertaV2PreTrainedModel):
    def __init__(self, config, labels_dict, id2components):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.labels_dict = labels_dict
        self.classifiers = nn.ModuleDict(
            {
                key: nn.Linear(self.deberta.config.hidden_size, len(labels))
                for key, labels in labels_dict.items()
            }
        )

        self.id2component_tensor = torch.stack(
            [id2components[a] for a in range(len(id2components))]
            + [torch.full((len(labels_dict),), -100)],
            dim=0,
        ).to(device)
        self.num_classes = self.id2component_tensor.shape[0]
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.deberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        sequence_output = self.dropout(outputs[0])
        logits_task_dict = {
            task: head(sequence_output) for task, head in self.classifiers.items()
        }

        loss = None
        if labels is not None:
            labels[labels == -100] = self.num_classes - 1
            labels_decomposed = torch.stack(
                [
                    torch.index_select(self.id2component_tensor, 0, label)
                    for label in labels
                ]
            ).to(device)
            loss_fct = nn.CrossEntropyLoss()
            losses = []
            for idx, (task, logits) in enumerate(logits_task_dict.items()):
                pred = logits.view(-1, len(self.labels_dict[task]))
                gold = labels_decomposed[:, :, idx].view(-1)
                losses.append(loss_fct(pred, gold))
            loss = sum(losses) / len(losses)
        output = TokenClassifierOutput(
            loss=loss,
            logits=logits_task_dict,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return output


class MultitaskCharToWordDebertaForTokenClassification(
    CharToWordDebertaPreTrainedModel
):
    def __init__(
        self, config, labels_dict, id2components, hidden_layer_index: int = -1
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.ctw_deberta = CharToWordDebertaModel(config)
        self.dropout = nn.Dropout(config.intra_word_encoder.hidden_dropout_prob)
        self.labels_dict = labels_dict
        self.classifiers = nn.ModuleDict(
            {
                key: nn.Linear(
                    self.ctw_deberta.config.intra_word_encoder.hidden_size * 2,
                    len(labels),
                )
                for key, labels in labels_dict.items()
            }
        )

        self.id2component_tensor = torch.stack(
            [id2components[a] for a in range(len(id2components))]
            + [torch.full((len(labels_dict),), -100)],
            dim=0,
        ).to(device)
        self.num_classes = self.id2component_tensor.shape[0]
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        char_input_mask=None,
        word_input_mask=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.ctw_deberta(
            input_ids, char_input_mask=char_input_mask, word_input_mask=word_input_mask
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits_task_dict = {
            task: head(sequence_output) for task, head in self.classifiers.items()
        }

        loss = None
        if labels is not None:
            labels[labels == -100] = self.num_classes - 1
            labels_decomposed = torch.stack(
                [
                    torch.index_select(self.id2component_tensor, 0, label)
                    for label in labels
                ]
            ).to(device)
            loss_fct = nn.CrossEntropyLoss()
            losses = []
            for idx, (task, logits) in enumerate(logits_task_dict.items()):
                pred = logits.view(-1, len(self.labels_dict[task]))
                gold = labels_decomposed[:, :, idx].view(-1)
                losses.append(loss_fct(pred, gold))
            loss = sum(losses) / len(losses)
        output = TokenClassifierOutput(
            loss=loss,
            logits=logits_task_dict,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return output
