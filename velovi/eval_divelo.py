import scanpy as sc
import scarches as sca
import argparse
import velovi
import scib

parser = argparse.ArgumentParser()

parser.add_argument("adata_file", default="data/Pancreas/pancreas_data_annotations.h5ad")
parser.add_argument("adata_output", default="data/Pancreas/pancreas_data_output.h5ad")

args = parser.parse_args()

adata_file = args.adata_file
adata_output = args.adata_output

print(adata_file)

n_latents=5

adata = sc.read(adata_file)

for i in range(n_latents):

    sc.pp.neighbors(adata, use_rep="X_divelo_"+str(i))
    scib.me.cluster_optimal_resolution(adata, cluster_key="cluster", label_key="celltype")
    scib.me.nmi(adata, cluster_key="cluster", label_key="celltype")