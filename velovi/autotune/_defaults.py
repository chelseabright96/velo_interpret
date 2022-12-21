from pytorch_lightning import LightningDataModule, LightningModule, Trainer

from scvi import model
from velovi import VELOVI
from scvi.module.base import BaseModuleClass, JaxBaseModuleClass, PyroBaseModuleClass
from scvi.train import TrainRunner

# colors for rich table columns
COLORS = [
    "dodger_blue1",
    "dark_violet",
    "green",
    "dark_orange",
]

# default rich table column kwargs
COLUMN_KWARGS = {
    "justify": "center",
    "no_wrap": True,
    "overflow": "fold",
}

# maps classes to the type of hyperparameters they use
TUNABLE_TYPES = {
    "model": [
        BaseModuleClass,
        JaxBaseModuleClass,
        PyroBaseModuleClass,
    ],
    "train": [
        LightningDataModule,
        Trainer,
        TrainRunner,
    ],
    "train_plan": [
        LightningModule,
    ],
}

# supported model classes
SUPPORTED = [model.SCVI, VELOVI]

# default hyperparameter search spaces for each model class
DEFAULTS = {
    model.SCVI: {
        "n_hidden": {"fn": "choice", "args": [[64, 128]]},
    },

    VELOVI: {
            "n_layers": {"fn": "choice", "args": [[1,2,3]]},
            "dropout_rate": {"fn": "choice", "args": [[0.1, 0.3, 0.5, 0.7]]},
            "recon_loss": {"fn": "choice", "args": [["zinb", "nb"]]},
            #"lr": 1e-3, #tune.choice([0.01, 0.005, 0.001, 0.0005, 0.0001]),
            #"alpha_GP": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1]),
            #"alpha_kl": tune.choice([0.5, 0.1, 0.05, 0.01, 0.005])
        }
}