from pathlib import Path
from typing import Optional, Union
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch

import scvelo as scv
from anndata import AnnData
from sklearn.preprocessing import MinMaxScaler

import numpy as np

# add binary I of size n_vars x number of annotated terms in files
# if I[i,j]=1 then gene i is active in annotation j
def add_annotations(adata, files, min_genes=0, max_genes=None, varm_key='I', uns_key='terms',
                    clean=True, genes_use_upper=True):
    """\
    Add annotations to an AnnData object from files.
    Parameters
    ----------
    adata
        Annotated data matrix.
    files
        Paths to text files with annotations. The function considers rows to be gene sets
        with name of a gene set in the first column followed by names of genes.
    min_genes
        Only include gene sets which have the total number of genes in adata
        greater than this value.
    max_genes
        Only include gene sets which have the total number of genes in adata
        less than this value.
    varm_key
        Store the binary array I of size n_vars x number of annotated terms in files
        in `adata.varm[varm_key]`. if I[i,j]=1 then the gene i is present in the annotation j.
    uns_key
        Store gene sets' names in `adata.uns[uns_key]`.
    clean
        If 'True', removes the word before the first underscore for each term name (like 'REACTOME_')
        and cuts the name to the first thirty symbols.
    genes_use_upper
        if 'True', converts genes' names from files and adata to uppercase for comparison.
    """
    files = [files] if isinstance(files, str) else files
    annot = []

    for file in files:
        with open(file) as f:
            p_f = [l.upper() for l in f] if genes_use_upper else f
            terms = [l.strip('\n').split() for l in p_f]

        if clean:
            terms = [[term[0].split('_', 1)[-1][:30]]+term[1:] for term in terms if term]
        annot+=terms

    var_names = adata.var_names.str.upper() if genes_use_upper else adata.var_names
    I = [[int(gene in term) for term in annot] for gene in var_names]
    I = np.asarray(I, dtype='int32')

    mask = I.sum(0) > min_genes
    if max_genes is not None:
        mask &= I.sum(0) < max_genes
    I = I[:, mask]
    adata.varm[varm_key] = I
    adata.uns[uns_key] = [term[0] for i, term in enumerate(annot) if i not in np.where(~mask)[0]]


def get_permutation_scores(save_path: Union[str, Path] = Path("data/")):
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if not (save_path / "permutation_scores.csv").is_file():
        URL = "https://figshare.com/ndownloader/files/36658185"
        urlretrieve(url=URL, filename=save_path / "permutation_scores.csv")

    return pd.read_csv(save_path / "permutation_scores.csv")


def preprocess_data(
    adata: AnnData,
    spliced_layer: Optional[str] = "Ms",
    unspliced_layer: Optional[str] = "Mu",
    min_max_scale: bool = True,
    filter_on_r2: bool = True,
) -> AnnData:
    """Preprocess data.

    This function removes poorly detected genes and minmax scales the data.

    Parameters
    ----------
    adata
        Annotated data matrix.
    spliced_layer
        Name of the spliced layer.
    unspliced_layer
        Name of the unspliced layer.
    min_max_scale
        Min-max scale spliced and unspliced
    filter_on_r2
        Filter out genes according to linear regression fit

    Returns
    -------
    Preprocessed adata.
    """
    if min_max_scale:
        scaler = MinMaxScaler()
        adata.layers[spliced_layer] = scaler.fit_transform(adata.layers[spliced_layer])

        scaler = MinMaxScaler()
        adata.layers[unspliced_layer] = scaler.fit_transform(
            adata.layers[unspliced_layer]
        )

    if filter_on_r2:
        scv.tl.velocity(adata, mode="deterministic")

        adata = adata[
            :, np.logical_and(adata.var.velocity_r2 > 0, adata.var.velocity_gamma > 0)
        ].copy()
        adata = adata[:, adata.var.velocity_genes].copy()

    return adata

def one_hot_encoder(idx, n_cls):
    assert torch.max(idx).item() < n_cls
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n_cls)
    onehot = onehot.to(idx.device)
    onehot.scatter_(1, idx.long(), 1)
    return onehot
