import pytorch_lightning as pl
from datasets import load_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from config import MODEL_NAME
from project.lit_qa import QAModel
from project.utils_qa import (prepare_train_features,
                              prepare_validation_features)

# MODEL_NAME = "bert-base-multilingual-cased"
# MODEL_NAME = "xlm-roberta-base"

global tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

overwrite_cache = True

class QADataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 8
        self.num_workers = 20
        self.data_collator = DataCollatorWithPadding(tokenizer)


        squad_drcd_dataset = load_dataset("json", data_files={
            "train": "/user_data/squad_drcd/data/squad_drcd/train.jsonl",
            "validation": "/user_data/squad_drcd/data/squad_drcd/valid.jsonl"
            }
        )
        squad_drcd_train_examples = squad_drcd_dataset["train"]
        squad_drcd_eval_examples = squad_drcd_dataset["validation"]

        self.squad_drcd_eval_examples = squad_drcd_eval_examples

        column_names = squad_drcd_dataset["train"].column_names
        squad_drcd_train_dataset = squad_drcd_train_examples.map(
            prepare_train_features,
            batched=True,
            num_proc=self.num_workers,
            remove_columns=column_names,
            load_from_cache_file=not overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
        squad_drcd_eval_dataset = squad_drcd_eval_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=self.num_workers,
            remove_columns=column_names,
            load_from_cache_file=not overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
        squad_drcd_eval_dataset_for_model = squad_drcd_eval_dataset.remove_columns(["example_id", "offset_mapping"])

        self.squad_drcd_train_dataset = squad_drcd_train_dataset
        self.squad_drcd_eval_dataset = squad_drcd_eval_dataset
        self.squad_drcd_test_dataset = squad_drcd_eval_dataset

        self.squad_drcd_eval_dataset_for_model = squad_drcd_eval_dataset_for_model

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
            load_from_cache_file=not overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
        noans_w2v_eval_dataset_for_model = noans_w2v_eval_dataset.remove_columns(["example_id", "offset_mapping"])

        self.squad_w2v_eval_examples = noans_w2v_eval_examples
        self.squad_w2v_eval_dataset = noans_w2v_eval_dataset
        self.squad_w2v_eval_dataset_for_model = noans_w2v_eval_dataset_for_model

    def val_examples(self):
        return self.squad_w2v_eval_examples

    def val_dataset(self):
        return self.squad_w2v_eval_dataset

    def train_dataloader(self):
        return DataLoader(
            self.squad_drcd_train_dataset,
            collate_fn=self.data_collator,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.squad_w2v_eval_dataset_for_model,
            collate_fn=self.data_collator,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

def print_params():
    model = QAModel.load_from_checkpoint(
        "model/xlm/squad_drcd/squad-drcd-epoch=00.ckpt",
        hf_path=MODEL_NAME,
    )
    from torchinfo import summary
    summary(model)


# f"model/xlm/squad_drcd/squad-drcd-epoch={i:02d}.ckpt"
# f"model/mbert/squad_drcd/squad-drcd-epoch={i:02d}.ckpt"
def evaluate():
    dm = QADataModule()
    trainer = Trainer(
        gpus=1,
        strategy="ddp"
    )
    for i in range(30):
        ckpt_path = f"model/mbert/squad_drcd/squad-drcd-epoch={i:02d}.ckpt"
        model = QAModel.load_from_checkpoint(
            ckpt_path,
            hf_path=MODEL_NAME, 
            eval_examples=dm.val_examples(), 
            eval_dataset=dm.val_dataset(),
            epoch=f"{i:02d}",
        )

        trainer.validate(model, dm.val_dataloader())


def main():
    callbacks = [
        ModelCheckpoint(
            dirpath="model/xlm/squad_drcd",
            filename="squad-drcd-{epoch:02d}",
            save_top_k=-1,
        )
    ]

    trainer = Trainer(
        gpus=-1,
        max_epochs=30,
        callbacks=callbacks,
        strategy="ddp"
    )
    dm = QADataModule()

    model = QAModel(hf_path=MODEL_NAME, eval_examples=dm.val_examples(), eval_dataset=dm.val_dataset())
    trainer.fit(model, datamodule=dm, ckpt_path="/user_data/unans_qa/model/xlm/squad_drcd/squad-drcd-epoch=19.ckpt")
    # trainer.save_checkpoint("finetuned_squad/squad_mix.ckpt")


if __name__ == "__main__":
    # main()

    evaluate()
    # print_params()
