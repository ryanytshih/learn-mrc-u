import pytorch_lightning as pl
from datasets import load_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from project.lit_qa import QAAdapterFusionModel
from project.utils_qa import (prepare_train_features,
                              prepare_validation_features)

# MODEL_NAME = "xlm-roberta-base"
# MODEL_NAME = "bert-base-multilingual-cased"
MODEL_NAME = "/user_data/unans_qa/model/mbert/drcd/hf/drcd-epoch-02"
# MODEL_NAME = "/user_data/unans_qa/model/xlm/drcd/hf/drcd-epoch-03"

LOG_DIR = "pl_logs"
LOG_NAME = "XLMR-DRCD-Fusion-QA-QNLI"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

class QADataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 16
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
            load_from_cache_file=True,
            desc="Running tokenizer on train dataset",
        )
        squad_eval_dataset = squad_eval_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=self.num_workers,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on validation dataset",
        )
        squad_eval_dataset_for_model = squad_eval_dataset.remove_columns(["example_id", "offset_mapping"])

        self.squad_train_dataset = squad_train_dataset
        self.squad_eval_dataset = squad_eval_dataset
        self.squad_test_dataset = squad_eval_dataset

        self.squad_eval_dataset_for_model = squad_eval_dataset_for_model


        # squad dev set 所有google翻譯過的no answer + has answer翻譯過，答案在context找的到且只找到一個
        noans_squadtrans = load_dataset("json", data_files={"validation": "/user_data/translate_squad/data/dev_translated.jsonl"})
        noans_eval_examples = noans_squadtrans["validation"]
        column_names = noans_squadtrans["validation"].column_names
        noans_eval_dataset = noans_eval_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=self.num_workers,
            remove_columns=column_names,
            load_from_cache_file=True,
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
            load_from_cache_file=False,
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
            self.squad_train_dataset,
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

# f"model/mbert/madx_no_task/madx-notask-epoch={i:02d}.ckpt"
# f"model/xlm/fusion/qa_qnli/fusion-qnli-epoch={i:02d}.ckpt"
# f"model/xlm/fusion/drcd_qa_qnli_fusion/fusion-qnli-epoch={i:02d}.ckpt"
# f"model/mbert/fusion/qa_qnli/fusion-qnli-epoch={i:02d}.ckpt"
# f"model/mbert/fusion/drcd_qa_qnli_fusion/fusion-qnli-epoch={i:02d}.ckpt"
def evaluate():
    dm = QADataModule()
    trainer = Trainer(
        gpus=1,
        # strategy="ddp"
    )
    for i in range(30):
        ckpt_path = f"model/mbert/fusion/drcd_qa_qnli_fusion/fusion-qnli-epoch={i:02d}.ckpt"
        model = QAAdapterFusionModel.load_from_checkpoint(
            ckpt_path,
            hf_path=MODEL_NAME, 
            eval_examples=dm.val_examples(), 
            eval_dataset=dm.val_dataset(),
            epoch=f"{i:02d}",
        )
        model.activate("zh")

        trainer.validate(model, dm.val_dataloader())

# "model/xlm/fusion/qa_qnli"
# "model/mbert/fusion/drcd_qa_qnli_fusion"
def main():
    
    logger1 = TensorBoardLogger(save_dir=LOG_DIR, name=LOG_NAME)
    logger2 = WandbLogger(save_dir=LOG_DIR, name=LOG_NAME)

    callbacks = [
        ModelCheckpoint(
            dirpath="model/xlm/fusion/drcd_qa_qnli_fusion",
            filename="fusion-qnli-{epoch:02d}",
            save_top_k=-1,
        )
    ]
    trainer = Trainer(
        gpus=-1,
        max_epochs=30,
        callbacks=callbacks,
        strategy="ddp",
        logger=[logger1, logger2]
    )
    dm = QADataModule()

    model = QAAdapterFusionModel(hf_path=MODEL_NAME, eval_examples=dm.val_examples(), eval_dataset=dm.val_dataset())
    model.activate("en")
    from torchinfo import summary
    summary(model)
    trainer.fit(model, datamodule=dm, ckpt_path="/user_data/unans_qa/model/xlm/fusion/drcd_qa_qnli_fusion/fusion-qnli-epoch=14.ckpt")
    # trainer.validate(model, datamodule=dm)
    # trainer.save_checkpoint("finetuned_squad/squad_mix.ckpt")

if __name__ == "__main__":
    # main()
    evaluate()
