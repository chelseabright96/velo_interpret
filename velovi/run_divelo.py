import scanpy as sc
import scarches as sca
import argparse
import velovi
import scvi
import torch
import os

import optuna
from optuna.integration import PyTorchLightningPruningCallback


#parser = argparse.ArgumentParser()

#parser.add_argument("adata_file", default="data/Pancreas/pancreas_data_annotations.h5ad")
#parser.add_argument("adata_output", default="data/Pancreas/pancreas_data_output.h5ad")

#args = parser.parse_args()

#adata_file = args.adata_file
#adata_output = args.adata_output

adata_file = "data/Pancreas/pancreas_data_annotations.h5ad"
adata_output = "data/Pancreas/pancreas_data_output.h5ad"


early_stopping_kwargs = {
    "early_stopping_metric": "val_unweighted_loss",
    "threshold": 0,
    "patience": 50,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}


print(adata_file)

adata = sc.read(adata_file)

velovi.VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")

adata_train = adata.copy()

omega_reactome=torch.ones(adata.varm["I"][:,50:].shape[1])
omega_panglao=torch.zeros(adata.varm["I"][:,:50].shape[1])
omega=torch.cat((omega_panglao,omega_reactome))


def objective(trial):
    # PyTorch Lightning will try to restore model parameters from previous trials if checkpoint
    # filenames match. Therefore, the filenames for each trial must be made unique.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "trial_{}".format(trial.number)), monitor="accuracy"
    )

    # The default logger in PyTorch Lightning writes to event files to be consumed by
    # TensorBoard. We create a simple logger instead that holds the log in memory so that the
    # final accuracy can be obtained after optimization. When using the default logger, the
    # final accuracy could be stored in an attribute of the `Trainer` instead.
    logger = DictLogger(trial.number)

    trainer_kwargs = {
        "checkpoint_callback": checkpoint_callback,
        "early_stop_callback": PyTorchLightningPruningCallback(trial, monitor="validation_loss")
        }


    vae = velovi.VELOVI(
            trial,
            adata=adata_train
        )

    vae.train(
        n_epochs=500,
        omega=None,
        use_gpu=True,
        weight_decay=0.,
        early_stopping_kwargs=early_stopping_kwargs,
        use_early_stopping=True,
        **trainer_kwargs
    )

    return logger.metrics[-1]["validation_loss"]


study = optuna.create_study(direction="minimize", pruner=pruner)
study.optimize(objective, n_trials=2, timeout=600)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

