import scanpy as sc
import scarches as sca
import argparse
import velovi
import scvi
import torch
import os

from velovi import VELOVI

parser = argparse.ArgumentParser()

parser.add_argument("adata_file", default="data/Pancreas/pancreas_data_annotations.h5ad")
parser.add_argument("adata_output", default="data/Pancreas/pancreas_data_output.h5ad")

args = parser.parse_args()

adata_file = args.adata_file
adata_output = args.adata_output

#adata_file = "data/Pancreas/pancreas_data_annotations.h5ad"
#adata_output = "data/Pancreas/pancreas_data_output.h5ad"

print(adata_file)

adata = sc.read(adata_file)

adata_train = adata.copy()

model_cls = VELOVI
model_cls.setup_anndata(adata_train, spliced_layer="Ms", unspliced_layer="Mu")
tuner = velovi.autotune.ModelTuner(model_cls)
results = tuner.fit(adata=adata_train, metric="validation_loss")













