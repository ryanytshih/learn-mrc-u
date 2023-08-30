import json
import math
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.checkpoint
from datasets import load_dataset, load_metric
from packaging import version
from pytorch_lightning import LightningDataModule, LightningModule
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset
from transformers import (AdamW, AdapterConfig, AutoModelForQuestionAnswering,
                          AutoTokenizer, BertModel, BertPreTrainedModel,
                          BertTokenizer, DataCollatorWithPadding,
                          EvalPrediction, PreTrainedTokenizerBase,
                          RobertaModel, RobertaPreTrainedModel,
                          XLMRobertaConfig, default_data_collator)
from transformers.adapters.composition import Fuse, Stack
from transformers.file_utils import PaddingStrategy
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from project.utils_qa import (postprocess_qa_predictions,
                              prepare_train_features,
                              prepare_validation_features)

from config import MODEL_NAME

# MODEL_NAME = "bert-base-multilingual-cased"
# MODEL_NAME = "xlm-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25", use_fast=True)
# tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)

overwrite_cache = True

question_column_name = 'question'
context_column_name = 'context'
answer_column_name = 'answers'
# Post-processing:
def post_processing_function(examples, features, predictions, stage="eval", output_dir="output/"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=True,
        n_best_size=20,
        max_answer_length=30,
        null_score_diff_threshold=0.0,
        output_dir=output_dir,
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    if True:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

    references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)

# Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    """
    Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    Args:
        start_or_end_logits(:obj:`tensor`):
            This is the output predictions of the model. We can only enter either start or end logits.
        eval_dataset: Evaluation dataset
        max_len(:obj:`int`):
            The maximum length of the output tensor. ( See the model.eval() part for more details )
    """

    step = 0
    # create a numpy array and fill it with -100.
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
    # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather
    for i, output_logit in enumerate(start_or_end_logits):  # populate columns
        # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
        # And after every iteration we have to change the step

        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]

        if step + batch_size < len(dataset):
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size

    return logits_concat

# metric = load_metric("squad_v2")
metric = load_metric("metrics/squad_chinese/squad_chinese.py")

class BertForMixQuestionAnswering(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        # self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.qa_outputs = nn.ModuleList([
            nn.Linear(config.hidden_size, config.num_labels) for _ in range(2)
        ])
        # self.qa_outputs = nn.ModuleDict({
        #     "English": nn.Linear(config.hidden_size, config.num_labels),
        #     "Chinese": nn.Linear(config.hidden_size, config.num_labels)
        # })

        self.init_weights()
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        target=None
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        
        # print(f"target: {target}")
        # logits = self.qa_outputs[target.item()](sequence_output)
        
        if target == torch.tensor(0):
            logits = self.qa_outputs[0](sequence_output)
        elif target == torch.tensor(1):
            logits = self.qa_outputs[1](sequence_output)
        else:
            print(target)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class XLMRobertaForMixQuestionAnswering(RobertaPreTrainedModel):

    config_class = XLMRobertaConfig

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.ModuleList([
            nn.Linear(config.hidden_size, config.num_labels) for _ in range(2)
        ])

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        target=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        if target == torch.tensor(0):
            logits = self.qa_outputs[0](sequence_output)
        elif target == torch.tensor(1):
            logits = self.qa_outputs[1](sequence_output)
        else:
            print(target)

        # logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@dataclass
class DataCollatorWithQAMix:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    target: int = 0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        batch["target"] = torch.tensor(self.target)
        return batch

class QADataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 16
        self.num_workers = 20
        self.en_data_collator = DataCollatorWithQAMix(tokenizer, target=0)
        self.zh_data_collator = DataCollatorWithQAMix(tokenizer, target=1)
        self.data_collator = DataCollatorWithPadding(tokenizer)
        
        dev_size = 10
        
        # squad
        squad_raw_dataset = load_dataset("squad_v2")
        # , split={"train": f"train[:1%]", "validation": "validation[:1%]"}
        # print(squad_raw_dataset)

        squad_train_examples = squad_raw_dataset["train"]
        squad_eval_examples = squad_raw_dataset["validation"]

        self.squad_eval_examples = squad_eval_examples

        column_names = squad_raw_dataset["train"].column_names
        squad_train_dataset = squad_train_examples.map(
            prepare_train_features,
            batched=True,
            num_proc=self.num_workers,
            remove_columns=column_names,
            load_from_cache_file=not overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
        squad_eval_dataset = squad_eval_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=self.num_workers,
            remove_columns=column_names,
            load_from_cache_file=not overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
        squad_eval_dataset_for_model = squad_eval_dataset.remove_columns(["example_id", "offset_mapping"])
        
        self.squad_train_dataset = squad_train_dataset
        self.squad_eval_dataset = squad_eval_dataset
        self.squad_test_dataset = squad_eval_dataset

        self.squad_eval_dataset_for_model = squad_eval_dataset_for_model
        
        # drcd
        drcd_raw_dataset = load_dataset("json", data_files={
            "train": "/user_data/squad_drcd/data/drcd/train.jsonl",
            "validation": "/user_data/squad_drcd/data/drcd/valid.jsonl",
            "test": "/user_data/squad_drcd/data/drcd/test.jsonl"
        })
        # , split={"train": "train[:10%]", "validation": "validation[:10%]", "test": "test[:10%]"}

        drcd_train_examples = drcd_raw_dataset["train"]
        drcd_eval_examples = drcd_raw_dataset["validation"]
        drcd_test_examples = drcd_raw_dataset["test"]
        
        self.drcd_eval_examples = drcd_eval_examples

        column_names = drcd_raw_dataset["train"].column_names
        drcd_train_dataset = drcd_train_examples.map(
            prepare_train_features,
            batched=True,
            num_proc=self.num_workers,
            remove_columns=column_names,
            load_from_cache_file=not overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
        drcd_eval_dataset = drcd_eval_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=self.num_workers,
            remove_columns=column_names,
            load_from_cache_file=not overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
        drcd_eval_dataset_for_model = drcd_eval_dataset.remove_columns(["example_id", "offset_mapping"])
        
        drcd_test_dataset = drcd_test_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=self.num_workers,
            remove_columns=column_names,
            load_from_cache_file=not overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
        drcd_test_dataset_for_model = drcd_test_dataset.remove_columns(["example_id", "offset_mapping"])
        
        self.drcd_train_dataset = drcd_train_dataset
        self.drcd_val_dataset = drcd_eval_dataset
        self.drcd_test_dataset = drcd_test_dataset
        
        self.drcd_val_dataset_for_model = drcd_eval_dataset_for_model
        self.drcd_test_dataset_for_model = drcd_test_dataset_for_model

        # # squad translated
        # squad_trans_raw_dataset = load_dataset("json", data_files={"train": "/user_data/translate_squad/data/train_translated_64001.jsonl"})

        # squad_trans_train_examples = squad_trans_raw_dataset["train"]

        # column_names = squad_trans_raw_dataset["train"].column_names
        # squad_trans_train_dataset = squad_trans_train_examples.map(
        #     prepare_train_features,
        #     batched=True,
        #     num_proc=20,
        #     remove_columns=column_names,
        #     load_from_cache_file=False,
        #     desc="Running tokenizer on train dataset",
        # )

        # squad dev set 所有google翻譯過的no answer + has answer翻譯過，答案在context找的到且只找到一個
        noans_squadtrans = load_dataset("json", data_files={"validation": "/user_data/translate_squad/data/dev_translated.jsonl"})
        noans_eval_examples = noans_squadtrans["validation"]
        column_names = noans_squadtrans["validation"].column_names
        noans_eval_dataset = noans_eval_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=self.num_workers,
            remove_columns=column_names,
            load_from_cache_file=not overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
        noans_eval_dataset_for_model = noans_eval_dataset.remove_columns(["example_id", "offset_mapping"])

        self.squad_trans_eval_examples = noans_eval_examples
        self.squad_trans_eval_dataset = noans_eval_dataset
        self.squad_trans_eval_dataset_for_model = noans_eval_dataset_for_model
        # batch_size = 16
        # noans_valid_dl = DataLoader(noans_eval_dataset_for_model, collate_fn=data_collator, batch_size=batch_size)

        "/user_data/squad_drcd/data/squad_drcd/unans/drcd_valid_w2v_noans.jsonl"
        noans_w2v = load_dataset("json", data_files={
            "validation": "/user_data/squad_drcd/data/squad_drcd/unans/drcd_valid_add_w2v_noans.jsonl",
            "test": "/user_data/squad_drcd/data/squad_drcd/unans/drcd_test_add_w2v_noans.jsonl"})
        # noans_w2v_eval_examples = noans_w2v["validation"]
        # column_names = noans_w2v["validation"].column_names
        noans_w2v_eval_examples = noans_w2v["test"]
        column_names = noans_w2v["test"].column_names
        noans_w2v_eval_dataset = noans_w2v_eval_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=self.num_workers,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on validation dataset",
        )
        noans_w2v_eval_dataset_for_model = noans_w2v_eval_dataset.remove_columns(["example_id", "offset_mapping"])

        self.squad_w2v_eval_examples = noans_w2v_eval_examples
        self.squad_w2v_eval_dataset = noans_w2v_eval_dataset
        self.squad_w2v_eval_dataset_for_model = noans_w2v_eval_dataset_for_model

    # def prepare_data(self):
    #     # download, split, etc...
    #     # only called on 1 GPU/TPU in distributed
    # def setup(self, stage):
    #     # make assignments here (val/train/test split)
    #     # called on every process in DDP
    def val_examples(self):
        # return self.drcd_eval_examples
        return self.squad_w2v_eval_examples

    def val_dataset(self):
        # return self.drcd_val_dataset
        return self.squad_w2v_eval_dataset

    def train_dataloader(self):
        return {
            "en": DataLoader(
                self.squad_train_dataset,
                collate_fn=self.en_data_collator,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True
            ),
            "zh": DataLoader(
                self.drcd_train_dataset,
                collate_fn=self.zh_data_collator,
                batch_size=self.batch_size,
                shuffle=True
            )
        }

    def val_dataloader(self):
        # return DataLoader(
        #     self.drcd_val_dataset_for_model,
        #     collate_fn=self.zh_data_collator,
        #     batch_size=self.batch_size,
        #     num_workers=self.num_workers,
        #     shuffle=False
        # )
        return DataLoader(
            self.squad_w2v_eval_dataset_for_model,
            collate_fn=self.en_data_collator,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
        # return {
        #     "en": DataLoader(self.squad_val_dataset, collate_fn=self.data_collator, batch_size=self.batch_size, shuffle=False),
        #     "zh": DataLoader(self.drcd_val_dataset, collate_fn=self.data_collator, batch_size=self.batch_size, shuffle=False)
        # }

    def test_dataloader(self):
        # return DataLoader(
        #     self.drcd_test_dataset_for_model,
        #     collate_fn=self.zh_data_collator,
        #     batch_size=self.batch_size,
        #     num_workers=self.num_workers,
        #     shuffle=False
        # )
        return DataLoader(
            self.squad_trans_val_dataset_for_model,
            collate_fn=self.en_data_collator,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
        # return {
        #     "en": DataLoader(self.squad_test_dataset, collate_fn=self.data_collator, batch_size=self.batch_size, shuffle=False),
        #     "zh": DataLoader(self.drcd_test_dataset, collate_fn=self.data_collator, batch_size=self.batch_size, shuffle=False)
        # }

class QAMixModel(LightningModule):
    def __init__(self, hf_path="bert-base-multilingual-uncased", eval_examples=None, eval_dataset=None, epoch=0):
        super().__init__()
        self.model = BertForMixQuestionAnswering.from_pretrained(hf_path)
        self.configure_validation_dataset(eval_examples, eval_dataset)
        self.epoch = epoch
        self.test = 0
        self.log("init", self.test)

    def forward(self, x):
        return self.model(**x)
    
    def training_step(self, batch, batch_idx):
        self.log("into training step", self.test)
        # batch["en"]["target"] = 0
        # self.log("batch en target device", batch["en"]["target"].device())
        # print(batch["en"]["target"].device())
        # batch["zh"]["target"] = 1
        # self.log("batch zh target device", batch["zh"]["target"].device())
        # print(batch["zh"]["target"].device())
        outputs1 = self(batch["en"])
        outputs2 = self(batch["zh"])
        loss_en = outputs1.loss
        loss_zh = outputs2.loss
        loss = (loss_en + loss_zh) / 2.0
        self.log('loss', loss.detach(), on_step=True, on_epoch=True)
        return {'loss': loss, 'log': {'train_loss': loss.detach()}}
    
#     def training_step_end(self, batch_parts):
#         # predictions from each GPU
#         predictions = batch_parts["pred"]
#         # losses from each GPU
#         losses = batch_parts["loss"]
#         loss = (losses[0] + losses[1]) / 2

#         gpu_0_prediction = predictions[0]
#         gpu_1_prediction = predictions[1]

#         # do something with both outputs
#         # return (losses[0] + losses[1]) / 2
#         return {'loss': loss, 'log': {'train_loss': loss.detach()}}
    
    def validation_step(self, batch, batch_idx):
        # batch["target"] = 0
        # print(batch)
        outputs = self(batch)
        # outputs1 = self(batch["en"], target=0)
        # outputs2 = self(batch["zh"], target=0)
        # loss = outputs.loss
        # print(outputs)
        pred = {
            "start_logits": outputs.start_logits.cpu().detach().numpy(),
            "end_logits": outputs.end_logits.cpu().detach().numpy()
        }
        
        # pred = {
        #     "start_logits": outputs.start_logits,
        #     "end_logits": outputs.end_logits
        # }
        
        # self.log('dev_loss',loss, on_step=True, on_epoch=True)
        return pred
    
#     def validation_step_end(self, batch_parts):
#         # predictions from each GPU
#         predictions = batch_parts["pred"]
#         # losses from each GPU
#         losses = batch_parts["loss"]
#         loss = (losses[0] + losses[1]) / 2

#         gpu_0_prediction = predictions[0]
#         gpu_1_prediction = predictions[1]

#         # do something with both outputs
#         # return (losses[0] + losses[1]) / 2
#         return {"loss": outputs.loss, "pred": pred}
    
    def validation_epoch_end(self, validation_step_outputs):
        examples = self.eval_examples
        dataset = self.eval_dataset
        # import pickle
        # with open("validation_step_outputs.pkl", "wb") as f:
        #     pickle.dump(validation_step_outputs, f)
        # print(len(validation_step_outputs))
        # print(validation_step_outputs[0]["start_logits"].size)
        all_start_logits = np.array([output["start_logits"] for output in validation_step_outputs])
        all_end_logits = np.array([output["end_logits"] for output in validation_step_outputs])
        
        max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
        
        # concatenate the numpy array
        start_logits_concat = create_and_fill_np_array(all_start_logits, dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, dataset, max_len)
        
        # delete the list of numpy arrays
        del all_start_logits
        del all_end_logits

        # null_score_diff_threshold 0.9
        prefix = "ep" + str(self.epoch)
        # prefix = "ep" + str(self.current_epoch)
        # output_dir = "model/mbert/squad_drcd_mix/drcd_add_w2v/testset_nsdt_0.0/"
        output_dir = "model/mbert/squad_drcd_mix/drcd_add_w2v/test_pred/"
        # output_dir = "model/mbert/squad_drcd_mix/squad_trans_pred/"

        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction = post_processing_function(examples, dataset, outputs_numpy, stage=prefix, output_dir=output_dir)
        eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
        # with open("tmp.json", "w", encoding="utf8") as f:
        #     json.dump(validation_step_outputs, f, indent=2, ensure_ascii=False)
        with open(os.path.join(output_dir, "eval_metric.jsonl"), "a", encoding="utf8") as f:
            f.write(json.dumps(eval_metric, ensure_ascii=False) + "\n")
        print(eval_metric)
        # del start_logits_concat
        # del end_logits_concat
        # del outputs_numpy
        # del prediction
        
        return eval_metric
    
    def configure_validation_dataset(self, eval_examples=None, eval_dataset=None):
        self.eval_examples = eval_examples
        self.eval_dataset = eval_dataset

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=5e-5)

class XLMRobertaQAMixModel(LightningModule):
    def __init__(self, hf_path="xlm-roberta-base", eval_examples=None, eval_dataset=None, epoch=0):
        super().__init__()
        self.model = XLMRobertaForMixQuestionAnswering.from_pretrained(hf_path)
        self.configure_validation_dataset(eval_examples, eval_dataset)
        self.epoch = epoch
        self.test = 0
        self.log("init", self.test)

    def forward(self, x):
        return self.model(**x)
    
    def training_step(self, batch, batch_idx):
        self.log("into training step", self.test)
        
        outputs1 = self(batch["en"])
        outputs2 = self(batch["zh"])
        loss_en = outputs1.loss
        loss_zh = outputs2.loss
        loss = (loss_en + loss_zh) / 2.0
        self.log('loss', loss.detach(), on_step=True, on_epoch=True)
        return {'loss': loss, 'log': {'train_loss': loss.detach()}}
    
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        
        pred = {
            "start_logits": outputs.start_logits.cpu().detach().numpy(),
            "end_logits": outputs.end_logits.cpu().detach().numpy()
        }
        return pred
    
    
    def validation_epoch_end(self, validation_step_outputs):
        examples = self.eval_examples
        dataset = self.eval_dataset
        # import pickle
        # with open("validation_step_outputs.pkl", "wb") as f:
        #     pickle.dump(validation_step_outputs, f)
        # print(len(validation_step_outputs))
        # print(validation_step_outputs[0]["start_logits"].size)
        all_start_logits = np.array([output["start_logits"] for output in validation_step_outputs])
        all_end_logits = np.array([output["end_logits"] for output in validation_step_outputs])
        
        max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
        
        # concatenate the numpy array
        start_logits_concat = create_and_fill_np_array(all_start_logits, dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, dataset, max_len)
        
        # delete the list of numpy arrays
        del all_start_logits
        del all_end_logits

        # null_score_diff_threshold 0.9
        prefix = "ep" + str(self.epoch)
        # prefix = "ep" + str(self.current_epoch)
        # output_dir = "model/xlm/squad_drcd_mix/trash/"
        # output_dir = "model/xlm/squad_drcd_mix_bak/drcd_add_w2v/dev_pred/"
        output_dir = "model/xlm/squad_drcd_mix/drcd_add_w2v/test_pred/"
        # output_dir = "model/xlm/squad_drcd_mix/squad_trans_pred/"

        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction = post_processing_function(examples, dataset, outputs_numpy, stage=prefix, output_dir=output_dir)
        eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
        # with open("tmp.json", "w", encoding="utf8") as f:
        #     json.dump(validation_step_outputs, f, indent=2, ensure_ascii=False)
        with open(os.path.join(output_dir, "eval_metric.jsonl"), "a", encoding="utf8") as f:
            f.write(json.dumps(eval_metric, ensure_ascii=False) + "\n")
        print(eval_metric)
        # del start_logits_concat
        # del end_logits_concat
        # del outputs_numpy
        # del prediction
        
        return eval_metric
    
    def configure_validation_dataset(self, eval_examples=None, eval_dataset=None):
        self.eval_examples = eval_examples
        self.eval_dataset = eval_dataset

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=5e-5)

class QAAdapterModel(LightningModule):
    def __init__(self, hf_path="bert-base-multilingual-cased", eval_examples=None, eval_dataset=None, epoch=0):
        super().__init__()
        self.epoch = epoch
        self.model = AutoModelForQuestionAnswering.from_pretrained(hf_path)
        self.configure_validation_dataset(eval_examples, eval_dataset)

        # Load the language adapters
        lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
        self.model.load_adapter("en/wiki@ukp", config=lang_adapter_config, model_name=MODEL_NAME)
        self.model.load_adapter("zh/wiki@ukp", config=lang_adapter_config, model_name=MODEL_NAME)

        # Add a new task adapter
        self.model.add_adapter("qa")
    
    def activate(self, lang="en"):
        self.model.active_adapters = Stack(lang, "qa")
        self.model.train_adapter(["qa"])
        # self.model.active_adapters = [lang]

    def train_adapter(self):
        self.model.train_adapter(["qa"])

    def save_adapter(self, save_path, adapter_name="qa"):
        self.model.save_adapter(save_path, adapter_name)

    def forward(self, x):
        return self.model(**x)
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log('loss',loss.detach(), on_step=True, on_epoch=True)
        return {'loss': loss, 'log': {'train_loss': loss.detach()}}
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        # loss = outputs.loss
        # print(outputs)
        pred = {
            "start_logits": outputs.start_logits.cpu().detach().numpy(),
            "end_logits": outputs.end_logits.cpu().detach().numpy()
        }
        # self.log('dev_loss',loss, on_step=True, on_epoch=True)
        return pred
    
    # def validation_step_end(self, batch_parts):
    #     # predictions from each GPU
    #     predictions = batch_parts["pred"]

    #     return {"pred": predictions}
    
    def validation_epoch_end(self, validation_step_outputs):
        examples = self.eval_examples
        dataset = self.eval_dataset
        # print(len(validation_step_outputs))
        # return
        all_start_logits = np.array([output["start_logits"] for output in validation_step_outputs])
        all_end_logits = np.array([output["end_logits"] for output in validation_step_outputs])
        
        max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
        
        # concatenate the numpy array
        start_logits_concat = create_and_fill_np_array(all_start_logits, dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, dataset, max_len)
        
        # delete the list of numpy arrays
        del all_start_logits
        del all_end_logits

        # null_score_diff_threshold 0.9
        prefix = "ep" + str(self.epoch)
        # prefix = "ep" + str(self.current_epoch)
        # output_dir = "model/xlm/drcd/squad_adapter/squad_trans_pred/"
        # output_dir = "model/mbert/drcd/squad_adapter/squad_trans_pred/"
        # output_dir = "model/xlm/drcd/squad_adapter/drcd_add_w2v/dev_pred/"
        # output_dir = "model/xlm/madx/drcd_add_w2v/test_pred"
        # output_dir = "model/xlm/madx/squad_trans_pred"
        output_dir = "model/mbert/drcd/squad_adapter/drcd_add_w2v/test_pred/"
        # output_dir = "model/mbert/madx/drcd_add_w2v/test_pred/"
        # output_dir = "model/mbert/madx/squad_trans_pred/"
        # output_dir = "model/mbert/drcd/squad_adapter/train/"

        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction = post_processing_function(examples, dataset, outputs_numpy, stage=prefix, output_dir=output_dir)
        eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
        # with open("tmp.json", "w", encoding="utf8") as f:
        #     json.dump(validation_step_outputs, f, indent=2, ensure_ascii=False)
        print(eval_metric)
        with open(os.path.join(output_dir, "eval_metric.jsonl"), "a", encoding="utf8") as f:
            f.write(json.dumps(eval_metric, ensure_ascii=False) + "\n")
        
        return eval_metric
    
    def configure_validation_dataset(self, eval_examples=None, eval_dataset=None):
        self.eval_examples = eval_examples
        self.eval_dataset = eval_dataset

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=5e-5)

class QAAdapterFusionModel(LightningModule):
    def __init__(self, hf_path="bert-base-multilingual-cased", eval_examples=None, eval_dataset=None, epoch=0):
        super().__init__()
        self.epoch = epoch
        self.model = AutoModelForQuestionAnswering.from_pretrained(hf_path)
        self.configure_validation_dataset(eval_examples, eval_dataset)

        # Load the language adapters
        lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
        self.model.load_adapter("en/wiki@ukp", config=lang_adapter_config, model_name=MODEL_NAME)
        self.model.load_adapter("zh/wiki@ukp", config=lang_adapter_config, model_name=MODEL_NAME)

        qa_adapter_path = "/user_data/unans_qa/model/adapters/madx/qa"
        qnli_adapter_path = "/user_data/unans_qa/model/adapters/madx/qnli/qnli"
        # qa_adapter_path = "/user_data/unans_qa/model/adapters/xlm/madx/qa"
        # qnli_adapter_path = "/user_data/unans_qa/model/adapters/xlm/madx/qnli/qnli"
        

        config = AdapterConfig.load("pfeiffer")
        self.model.load_adapter(qa_adapter_path, config=config, model_name=MODEL_NAME)
        self.model.load_adapter(qnli_adapter_path, config=config, model_name=MODEL_NAME)
        self.model.add_adapter_fusion(["qa", "qnli"])

    
    def activate(self, lang="en"):
        # self.model.active_adapters = Stack(lang, "qa")
        self.model.train_adapter_fusion(Fuse("qa", "qnli"))
        self.model.active_adapters = Stack(lang, Fuse("qa", "qnli"))
        # self.model.train_adapter(["qa"])
        # self.model.active_adapters = [lang]
        # print(self.model)

    # def train_adapter(self):
    #     self.model.train_adapter(["qa"])
    #     self.model.train_adapter_fusion(Fuse("qa", "qnli"))

    def forward(self, x):
        return self.model(**x)
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log('loss',loss.detach(), on_step=True, on_epoch=True)
        return {'loss': loss, 'log': {'train_loss': loss.detach()}}
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        # loss = outputs.loss
        # print(outputs)
        pred = {
            "start_logits": outputs.start_logits.cpu().detach().numpy(),
            "end_logits": outputs.end_logits.cpu().detach().numpy()
        }
        # self.log('dev_loss',loss, on_step=True, on_epoch=True)
        return pred
    
    # def validation_step_end(self, batch_parts):
    #     # predictions from each GPU
    #     predictions = batch_parts["pred"]

    #     return {"pred": predictions}
    
    def validation_epoch_end(self, validation_step_outputs):
        examples = self.eval_examples
        dataset = self.eval_dataset
        # print(len(validation_step_outputs))
        # return
        all_start_logits = np.array([output["start_logits"] for output in validation_step_outputs])
        all_end_logits = np.array([output["end_logits"] for output in validation_step_outputs])
        
        max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
        
        # concatenate the numpy array
        start_logits_concat = create_and_fill_np_array(all_start_logits, dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, dataset, max_len)
        
        # delete the list of numpy arrays
        del all_start_logits
        del all_end_logits

        # null_score_diff_threshold 0.9
        prefix = "ep" + str(self.epoch)
        # prefix = "ep" + str(self.current_epoch)
        # output_dir = "model/mbert/fusion/train/"
        # output_dir = "model/mbert/fusion/qa_qnli/drcd_add_w2v/test_pred"
        output_dir = "model/mbert/fusion/drcd_qa_qnli_fusion/drcd_add_w2v/test_pred/"
        # output_dir = "model/mbert/fusion/drcd_qa_qnli_fusion/squad_trans_pred/"
        # output_dir = "model/mbert/fusion/qa_qnli/squad_trans_pred"
        # output_dir = "model/xlm/fusion/qa_qnli/squad_trans_pred"
        # output_dir = "model/xlm/fusion/qa_qnli/drcd_add_w2v/dev_pred/"
        # output_dir = "model/xlm/fusion/drcd_qa_qnli_fusion/drcd_add_w2v/dev_pred/"
        # output_dir = "model/xlm/fusion/drcd_qa_qnli_fusion/drcd_add_w2v/test_pred/"
        # output_dir = "model/xlm/fusion/drcd_qa_qnli_fusion/squad_trans_pred/"

        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction = post_processing_function(examples, dataset, outputs_numpy, stage=prefix, output_dir=output_dir)
        eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
        # with open("tmp.json", "w", encoding="utf8") as f:
        #     json.dump(validation_step_outputs, f, indent=2, ensure_ascii=False)
        print(eval_metric)
        with open(os.path.join(output_dir, "eval_metric.jsonl"), "a", encoding="utf8") as f:
            f.write(json.dumps(eval_metric, ensure_ascii=False) + "\n")
        
        return eval_metric
    
    def configure_validation_dataset(self, eval_examples=None, eval_dataset=None):
        self.eval_examples = eval_examples
        self.eval_dataset = eval_dataset

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=5e-5)

class DRCDQAModel(LightningModule):
    def __init__(self, hf_path="bert-base-multilingual-cased", eval_examples=None, eval_dataset=None, epoch=0):
        super().__init__()
        self.epoch = epoch
        self.model = AutoModelForQuestionAnswering.from_pretrained(hf_path)
        self.configure_validation_dataset(eval_examples, eval_dataset)

    def forward(self, x):
        return self.model(**x)
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log('loss',loss.detach(), on_step=True, on_epoch=True)
        return {'loss': loss, 'log': {'train_loss': loss.detach()}}
    
    def training_epoch_end(self, training_step_outputs):
        output_path = f"/user_data/unans_qa/model/xlm/drcd/hf/drcd-epoch-{self.current_epoch:02d}"
        self.save_huggingface_model(output_path)
        
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        # loss = outputs.loss
        # print(outputs)
        pred = {
            "start_logits": outputs.start_logits.cpu().detach().numpy(),
            "end_logits": outputs.end_logits.cpu().detach().numpy()
        }
        # self.log('dev_loss',loss, on_step=True, on_epoch=True)
        return pred
    
    # def validation_step_end(self, batch_parts):
    #     # predictions from each GPU
    #     predictions = batch_parts["pred"]

    #     return {"pred": predictions}
    
    def validation_epoch_end(self, validation_step_outputs):
        examples = self.eval_examples
        dataset = self.eval_dataset
        # print(len(validation_step_outputs))
        # return
        all_start_logits = np.array([output["start_logits"] for output in validation_step_outputs])
        all_end_logits = np.array([output["end_logits"] for output in validation_step_outputs])
        
        max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
        
        # concatenate the numpy array
        start_logits_concat = create_and_fill_np_array(all_start_logits, dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, dataset, max_len)
        
        # delete the list of numpy arrays
        del all_start_logits
        del all_end_logits

        
        # null_score_diff_threshold 0.9
        prefix = "ep" + str(self.epoch)
        # prefix = "ep" + str(self.current_epoch)
        # "model/mbert/squad_drcd/drcd_add_w2v/nsdt_0.0/"
        output_dir = "model/xlm/drcd/drcd_test_pred/"
        # output_dir = "model/squad/pred/"

        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction = post_processing_function(examples, dataset, outputs_numpy, stage=prefix, output_dir=output_dir)
        eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
        # with open("tmp.json", "w", encoding="utf8") as f:
        #     json.dump(validation_step_outputs, f, indent=2, ensure_ascii=False)
        print(eval_metric)
        with open(os.path.join(output_dir, "eval_metric.jsonl"), "a", encoding="utf8") as f:
            f.write(json.dumps(eval_metric, ensure_ascii=False) + "\n")

        return eval_metric
    
    def configure_validation_dataset(self, eval_examples=None, eval_dataset=None):
        self.eval_examples = eval_examples
        self.eval_dataset = eval_dataset
    
    def save_huggingface_model(self, output_dir):
        self.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("Saved model to", output_dir)

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=5e-5)


class QAModel(LightningModule):
    def __init__(self, hf_path="bert-base-multilingual-uncased", eval_examples=None, eval_dataset=None, epoch=0):
        super().__init__()
        self.epoch = epoch
        self.model = AutoModelForQuestionAnswering.from_pretrained(hf_path)
        self.configure_validation_dataset(eval_examples, eval_dataset)

    def forward(self, x):
        return self.model(**x)
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log('loss',loss.detach(), on_step=True, on_epoch=True)
        return {'loss': loss, 'log': {'train_loss': loss.detach()}}
    
    # def training_epoch_end(self, training_step_outputs):
    #     self.save_huggingface_model()

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        # loss = outputs.loss
        # print(outputs)
        pred = {
            "start_logits": outputs.start_logits.cpu().detach().numpy(),
            "end_logits": outputs.end_logits.cpu().detach().numpy()
        }
        # self.log('dev_loss',loss, on_step=True, on_epoch=True)
        return pred
    
    # def validation_step_end(self, batch_parts):
    #     # predictions from each GPU
    #     predictions = batch_parts["pred"]

    #     return {"pred": predictions}
    
    def validation_epoch_end(self, validation_step_outputs):
        examples = self.eval_examples
        dataset = self.eval_dataset
        # print(len(validation_step_outputs))
        # return
        all_start_logits = np.array([output["start_logits"] for output in validation_step_outputs])
        all_end_logits = np.array([output["end_logits"] for output in validation_step_outputs])
        
        max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
        
        # concatenate the numpy array
        start_logits_concat = create_and_fill_np_array(all_start_logits, dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, dataset, max_len)
        
        # delete the list of numpy arrays
        del all_start_logits
        del all_end_logits

        
        # null_score_diff_threshold 0.9
        prefix = "ep" + str(self.epoch)
        # prefix = "ep" + str(self.current_epoch)
        # "model/mbert/squad_drcd/drcd_add_w2v/nsdt_0.0/"
        # output_dir = "model/xlm/squad_drcd/squad_trans_pred/"
        # output_dir = "model/xlm/squad_drcd/drcd_add_w2v/test_pred/"
        output_dir = "model/mbert/squad_drcd/drcd_add_w2v/test_pred/"
        # output_dir = "model/mbert/squad_drcd/squad_trans_pred/"
        # output_dir = "model/squad/pred/"
        # output_dir = "model/xlm/train"

        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction = post_processing_function(examples, dataset, outputs_numpy, stage=prefix, output_dir=output_dir)
        eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
        # with open("tmp.json", "w", encoding="utf8") as f:
        #     json.dump(validation_step_outputs, f, indent=2, ensure_ascii=False)
        print(eval_metric)
        with open(os.path.join(output_dir, "eval_metric.jsonl"), "a", encoding="utf8") as f:
            f.write(json.dumps(eval_metric, ensure_ascii=False) + "\n")

        return eval_metric
    
    def configure_validation_dataset(self, eval_examples=None, eval_dataset=None):
        self.eval_examples = eval_examples
        self.eval_dataset = eval_dataset
    
    def save_huggingface_model(self, output_dir):
        self.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("Saved model to", output_dir)

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=5e-5)


def main():
    model = QAMixModel()

if __name__ == "__main__":
    main()
