import scanpy as sc
import scarches as sca
import argparse
import velovi
import scvi
from hyperopt import hp
import torch
import os

from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback


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

def train_tune_checkpoint(config, num_epochs=10):

    trainer_kwargs = {
        "max_epochs": num_epochs,
        "logger": TensorBoardLogger(
            save_dir=os.getcwd(), name="", version="."),
        "enable_progress_bar": False,
        "callbacks": [
            TuneReportCheckpointCallback(
                metrics={
                    "validation_loss": "validation_loss"
                },
                filename="checkpoint",
                on="validation_end")
        ]
    }



    vae = velovi.VELOVI(
            adata=adata_train,
            config=config
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

#latent = vae.get_latent_representation()

def tune_pbt(num_samples=5, num_epochs=10):

    config = {"n_layers": 1 + tune.randint(0,5),
            "dropout_rate": tune.choice([0.1, 0.3, 0.5, 0.7]),
            "recon_loss": tune.choice(["zinb", "nb"]),
            "lr": 1e-3, #tune.choice([0.01, 0.005, 0.001, 0.0005, 0.0001]),
            "alpha_GP": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1]),
            "alpha_kl": tune.choice([0.5, 0.1, 0.05, 0.01, 0.005])}

    scheduler = PopulationBasedTraining(
        perturbation_interval=4,
        hyperparam_mutations={
            "lr": tune.loguniform(1e-4, 1e-1)
        })

    reporter = CLIReporter(
        parameter_columns=["n_layers", "dropout_rate", "recon_loss", "lr", "alpha_GP", "alpha_kl"],
        metric_columns=["validation_loss"])
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                train_tune_checkpoint,
                num_epochs=num_epochs),
            resources={
                "cpu": 2,
                "gpu": 1
            }
        ),
        tune_config=tune.TuneConfig(
            metric="validation_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            name="tune_divelo_test1",
            progress_reporter=reporter,
        ),
        param_space=config,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)





#alphas_kl = [0.5, 0.1, 0.05, 0.01, 0.005]

# for i, alpha_kl in enumerate(alphas_kl):
    
#     vae = velovi.VELOVI(
#         adata=adata_train
#     )

#     vae.train(
#         n_epochs=500,
#         alpha=0.7,
#         omega=None,
#         use_gpu=True,
#         alpha_kl=alpha_kl,
#         weight_decay=0.,
#         early_stopping_kwargs=early_stopping_kwargs,
#         use_early_stopping=True,
#         seed=2020
#     )

#     adata.obsm["X_divelo_"+str(i)] = vae.get_latent(only_active=True)

adata.write(adata_output)