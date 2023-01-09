import scanpy as sc
import scarches as sca
import argparse
import velovi
import scvi
from hyperopt import hp



parser = argparse.ArgumentParser()

parser.add_argument("adata_file", default="data/Pancreas/pancreas_data_annotations.h5ad")
parser.add_argument("adata_output", default="data/Pancreas/pancreas_data_output.h5ad")

args = parser.parse_args()

adata_file = args.adata_file
adata_output = args.adata_output

space = {
            "model_tunable_kwargs": {
                "n_hidden": hp.choice("n_hidden", [64, 128, 256]),
                "n_layers": 1 + hp.randint("n_layers", 5),
                "dropout_rate": hp.choice("dropout_rate", [0.1, 0.3, 0.5, 0.7]),
                "gene_likelihood": hp.choice("gene_likelihood", ["zinb", "nb"]),
            },
            "train_func_tunable_kwargs": {
                "lr": hp.choice("lr", [0.01, 0.005, 0.001, 0.0005, 0.0001])
            },

        }

print(adata_file)

adata = sc.read(adata_file)

velovi.VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")

adata_train = adata.copy()

#Do hyperparameter tuning
vae, trials = scvi.inference.autotune.auto_tune_scvi_model("pancreas", adata_train, model_class=velovi.VELOVI, metric_name="elbo_validation",space=space, save_path="trained_models")
latent = vae.get_latent_representation()

# early_stopping_kwargs = {
#     "early_stopping_metric": "val_unweighted_loss",
#     "threshold": 0,
#     "patience": 50,
#     "reduce_lr": True,
#     "lr_patience": 13,
#     "lr_factor": 0.1,
# }



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