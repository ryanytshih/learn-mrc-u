from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from config import MODEL_NAME
from project.lit_qa import QADataModule, QAMixModel, XLMRobertaQAMixModel

# MODEL_NAME = "bert-base-multilingual-cased"
# MODEL_NAME = "xlm-roberta-base"

def print_params():
    model = QAMixModel.load_from_checkpoint(
        "model/mbert/squad_drcd_mix/qamix-epoch=00.ckpt",
        hf_path=MODEL_NAME,
    )
    from torchinfo import summary
    summary(model)

# f"model/mbert/squad_drcd_mix/qamix-epoch={i:02d}.ckpt"
# f"model/xlm/squad_drcd_mix_bak/qamix-epoch={i:02d}.ckpt"
# f"model/xlm/squad_drcd_mix/qamix-epoch={i:02d}.ckpt"
def evaluate():
    dm = QADataModule()
    trainer = Trainer(
        gpus=1,
        # strategy="ddp"
    )
    for i in range(30):
        ckpt_path = f"model/mbert/squad_drcd_mix/qamix-epoch={i:02d}.ckpt"
        # QAMixModel
        # XLMRobertaQAMixModel
        model = QAMixModel.load_from_checkpoint(
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
            dirpath="model/xlm/squad_drcd_mix",
            filename="qamix-{epoch:02d}",
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

    mix_model = XLMRobertaQAMixModel(hf_path=MODEL_NAME, eval_examples=dm.val_examples(), eval_dataset=dm.val_dataset())
    trainer.fit(mix_model, datamodule=dm, ckpt_path="/user_data/unans_qa/model/xlm/squad_drcd_mix_bak/qamix-epoch=09.ckpt")
    # trainer.validate(mix_model, datamodule=dm)
    # trainer.save_checkpoint("finetuned_squad/squad_mix.ckpt")

if __name__ == "__main__":
    # main()
    # print_params()
    evaluate()
    # print_params()
