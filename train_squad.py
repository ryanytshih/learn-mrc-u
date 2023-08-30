import pytorch_lightning as pl
from datasets import load_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from project.lit_qa import QAModel
from project.utils_qa import (prepare_train_features,
                              prepare_validation_features)

MODEL_NAME = "bert-base-multilingual-cased"
# MODEL_NAME = "facebook/mbart-large-cc25"
# MODEL_NAME = "xlm-roberta-base"

global tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)


class QADataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 8
        self.num_workers = 20
        self.data_collator = DataCollatorWithPadding(tokenizer)

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
            load_from_cache_file=False,
            desc="Running tokenizer on train dataset",
        )
        squad_eval_dataset = squad_eval_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=self.num_workers,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on validation dataset",
        )
        squad_eval_dataset_for_model = squad_eval_dataset.remove_columns(["example_id", "offset_mapping"])

        self.squad_train_dataset = squad_train_dataset
        self.squad_eval_dataset = squad_eval_dataset
        self.squad_test_dataset = squad_eval_dataset

        self.squad_eval_dataset_for_model = squad_eval_dataset_for_model

    def val_examples(self):
        return self.squad_eval_examples

    def val_dataset(self):
        return self.squad_eval_dataset

    def train_dataloader(self):
        return DataLoader(
            self.squad_train_dataset,
            collate_fn=self.data_collator,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.squad_eval_dataset_for_model,
            collate_fn=self.data_collator,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )


def evaluate():
    dm = QADataModule()
    trainer = Trainer(
        gpus=1,
        strategy="ddp"
    )
    for i in range(1):
        ckpt_path = f"model/squad/squad-epoch={i:02d}.ckpt"
        model = QAModel.load_from_checkpoint(
            ckpt_path,
            hf_path=MODEL_NAME, 
            eval_examples=dm.val_examples(), 
            eval_dataset=dm.val_dataset()
        )

        trainer.validate(model, dm.val_dataloader())


def main():
    callbacks = [
        ModelCheckpoint(
            dirpath="model/xlm/squad",
            filename="squad-{epoch:02d}",
            save_top_k=-1,
        )
    ]

    trainer = Trainer(
        gpus=-1,
        max_epochs=20,
        callbacks=callbacks,
        strategy="ddp"
    )
    dm = QADataModule()

    model = QAModel(hf_path=MODEL_NAME, eval_examples=dm.val_examples(), eval_dataset=dm.val_dataset())
    trainer.fit(model, datamodule=dm)
    # trainer.save_checkpoint("finetuned_squad/squad_mix.ckpt")


if __name__ == "__main__":
    # main()

    evaluate()
