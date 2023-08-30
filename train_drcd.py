import pytorch_lightning as pl
from datasets import load_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from project.lit_qa import DRCDQAModel
from project.utils_qa import (prepare_train_features,
                              prepare_validation_features)

# MODEL_NAME = "bert-base-multilingual-cased"
# MODEL_NAME = "facebook/mbart-large-cc25"
MODEL_NAME = "xlm-roberta-base"

LOG_DIR = "pl_logs"
LOG_NAME = "XLMR-DRCD"

# global tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

overwrite_cache = True

class QADataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 16
        self.num_workers = 20
        self.data_collator = DataCollatorWithPadding(tokenizer)

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
        self.drcd_test_examples = drcd_test_examples

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
        self.drcd_eval_dataset = drcd_eval_dataset
        self.drcd_test_dataset = drcd_test_dataset
        
        self.drcd_eval_dataset_for_model = drcd_eval_dataset_for_model
        self.drcd_test_dataset_for_model = drcd_test_dataset_for_model

    def val_examples(self):
        return self.drcd_test_examples

    def val_dataset(self):
        return self.drcd_test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.drcd_train_dataset,
            collate_fn=self.data_collator,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.drcd_test_dataset_for_model,
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
    # for i in range(1):
    #     ckpt_path = f"model/squad/squad-epoch={i:02d}.ckpt"
    #     model = QAModel.load_from_checkpoint(
    #         ckpt_path,
    #         hf_path=MODEL_NAME, 
    #         eval_examples=dm.val_examples(), 
    #         eval_dataset=dm.val_dataset()
    #     )
    #     trainer.validate(model, dm.val_dataloader())
    for i in range(20):
        hf_path = f"/user_data/unans_qa/model/xlm/drcd/hf/drcd-epoch-{i:02d}"
        model = DRCDQAModel(
            hf_path=hf_path, 
            eval_examples=dm.val_examples(), 
            eval_dataset=dm.val_dataset(),
            epoch=f"{i:02d}",
        )
        trainer.validate(model, dm.val_dataloader())

    


def main():
    logger1 = TensorBoardLogger(save_dir=LOG_DIR, name=LOG_NAME)
    logger2 = WandbLogger(save_dir=LOG_DIR, name=LOG_NAME)

    # callbacks = [
    #     ModelCheckpoint(
    #         dirpath="model/mbert/drcd",
    #         filename="drcd-{epoch:02d}",
    #         save_top_k=-1,
    #     )
    # ]

    trainer = Trainer(
        gpus=-1,
        max_epochs=20,
        # callbacks=callbacks,
        strategy="ddp",
        logger=[logger1, logger2]
    )
    dm = QADataModule()

    model = DRCDQAModel(hf_path=MODEL_NAME, eval_examples=dm.val_examples(), eval_dataset=dm.val_dataset())
    trainer.fit(model, datamodule=dm)
    # trainer.save_checkpoint("finetuned_squad/squad_mix.ckpt")


if __name__ == "__main__":
    # main()

    evaluate()
