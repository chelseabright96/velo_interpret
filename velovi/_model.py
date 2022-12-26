import logging
import warnings
from functools import partial
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from anndata import AnnData
from joblib import Parallel, delayed
from scipy.stats import ttest_ind
from typing import Literal
#from scvi._compat import Literal
from scvi._utils import _doc_params
from scvi.data import AnnDataManager
from scvi.data.fields import LayerField
from scvi.dataloaders import AnnDataLoader, DataSplitter
from scvi.model._utils import scrna_raw_counts_properties
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.model.base._utils import _de_core
from scvi.train import TrainingPlan, TrainRunner
from scvi.utils._docstrings import doc_differential_expression, setup_anndata_dsp
from sklearn.metrics.pairwise import cosine_similarity
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from ._trainer import CustomTrainingPlan

from ._constants import REGISTRY_KEYS
from ._module import VELOVAE

logger = logging.getLogger(__name__)


def _softplus_inverse(x: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(x)
    x_inv = torch.where(x > 20, x, x.expm1().log()).numpy()
    return x_inv


class MyEarlyStopping(EarlyStopping):
    def __init__(self, **kwargs):
        super().__init__(**kwargs )
        self.early_stopping_reason = None
        

    def _evaluate_stopping_criteria(self):
        logger.info("Evaluating stopping criteria")
        should_stop, reason  = super()._evaluate_stopping_criteria()

        if not should_stop:
            new_lr = self.lr
            if self.watch_lr is not None and self.watch_lr != new_lr:
                self.watch_lr = new_lr
                self.update_prox_ops()
                logger.info(f"updating prox_ops from early stopping criteria")

        return should_stop, reason


class VELOVI(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    Velocity Variational Inference

    Parameters
    ----------
    adata
        AnnData object that has been registered via :func:`~velovi.VELOVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    gamma_init_data
        Initialize gamma using the data-driven technique.
    linear_decoder
        Use a linear decoder from latent space to time.
    **model_kwargs
        Keyword args for :class:`~velovi.VELOVAE`
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 256,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        gamma_init_data: bool = False,
        linear_decoder: bool = False,
        mask: Optional[Union[np.ndarray, list]] = None,
        mask_key: str = 'I',
        soft_mask: bool = False,
        **model_kwargs,
    ):
        super().__init__(adata)

        if mask is None and mask_key not in self.adata.varm:
            raise ValueError('Please provide mask.')
        
        if mask is None:
            mask = adata.varm[mask_key].T

        self.mask_ = mask if isinstance(mask, list) else mask.tolist()
        mask = torch.tensor(mask).float()
        self.n_latent = len(self.mask_)


        self.soft_mask_ = soft_mask

        spliced = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        unspliced = self.adata_manager.get_from_registry(REGISTRY_KEYS.U_KEY)

        sorted_unspliced = np.argsort(unspliced, axis=0)
        ind = int(adata.n_obs * 0.99)
        us_upper_ind = sorted_unspliced[ind:, :]

        us_upper = []
        ms_upper = []
        for i in range(len(us_upper_ind)):
            row = us_upper_ind[i]
            us_upper += [unspliced[row, np.arange(adata.n_vars)][np.newaxis, :]]
            ms_upper += [spliced[row, np.arange(adata.n_vars)][np.newaxis, :]]
        us_upper = np.median(np.concatenate(us_upper, axis=0), axis=0)
        ms_upper = np.median(np.concatenate(ms_upper, axis=0), axis=0)

        alpha_unconstr = _softplus_inverse(us_upper)
        alpha_unconstr = np.asarray(alpha_unconstr).ravel()

        alpha_1_unconstr = np.zeros(us_upper.shape).ravel()
        lambda_alpha_unconstr = np.zeros(us_upper.shape).ravel()

        if gamma_init_data:
            gamma_unconstr = np.clip(_softplus_inverse(us_upper / ms_upper), None, 10)
        else:
            gamma_unconstr = None

        self.module = VELOVAE(
            n_input=self.summary_stats["n_vars"],
            n_hidden=n_hidden,
            n_latent=self.n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            gamma_unconstr_init=gamma_unconstr,
            alpha_unconstr_init=alpha_unconstr,
            alpha_1_unconstr_init=alpha_1_unconstr,
            lambda_alpha_unconstr_init=lambda_alpha_unconstr,
            switch_spliced=ms_upper,
            switch_unspliced=us_upper,
            linear_decoder=linear_decoder,
            mask=mask,
            soft_mask=self.soft_mask_,
            **model_kwargs,
        )
        self._model_summary_string = (
            "VELOVI Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
        )
        self.init_params_ = self._get_init_params(locals())



    def train(
        self,
        max_epochs: Optional[int] = 500,
        lr: float = 1e-2,
        weight_decay: float = 1e-2,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = 0.1,
        batch_size: int = 256,
        early_stopping: bool = True,
        early_stopping_monitor: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
        ] = "elbo_validation",
        gradient_clip_val: float = 10,
        alpha_GP=0.7,
        alpha_kl=0.1,
        omega: Optional[torch.Tensor] = None,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """
        Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        lr
            Learning rate for optimization
        weight_decay
            Weight decay for optimization
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        gradient_clip_val
            Val for gradient clipping
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        user_plan_kwargs = (
            plan_kwargs.copy() if isinstance(plan_kwargs, dict) else dict()
        )
        plan_kwargs = dict(weight_decay=weight_decay, optimizer="AdamW")
        plan_kwargs.update(user_plan_kwargs)

        
        user_train_kwargs = trainer_kwargs.copy()
        trainer_kwargs = dict(gradient_clip_val=gradient_clip_val)
        trainer_kwargs.update(user_train_kwargs)

        data_splitter = DataSplitter(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )


        training_plan = CustomTrainingPlan(self.module, alpha_GP=alpha_GP, alpha_kl=alpha_kl, omega=omega, **plan_kwargs)
        #training_plan = TrainingPlan(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )

        trainer_kwargs["callbacks"] = (
            [] if "callbacks" not in trainer_kwargs.keys() else trainer_kwargs["callbacks"]
        )
        #trainer_kwargs["callbacks"] += [MyEarlyStopping(monitor=early_stopping_monitor)]

        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )

        # print(next(iter(data_splitter.train_dataloader())))
        # print(next(iter(data_splitter.val_dataloader())))
        return runner()

    

    def get_loss(self, adata):
        # run the model forward on the data

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=None, batch_size=256
        )

        for tensors in scdl:    
            inference_outputs, generative_outputs, loss = self.module.forward(
                tensors=tensors,
                compute_loss=True,
                )
        
        return inference_outputs, generative_outputs, loss


    @torch.no_grad()
    def get_state_assignment(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        hard_assignment: bool = False,
        n_samples: int = 20,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], List[str]]:
        """
        Returns cells by genes by states probabilities.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        hard_assignment
            Return a hard state assignment
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        states = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=True,
                )
                output = generative_outputs["px_pi"]
                output = output[..., gene_mask, :]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes by four
            states.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                states[-1] = np.mean(states[-1], axis=0)

        states = np.concatenate(states, axis=0)
        state_cats = [
            "induction",
            "induction_steady",
            "repression",
            "repression_steady",
        ]
        if hard_assignment and return_mean:
            hard_assign = states.argmax(-1)

            hard_assign = pd.DataFrame(
                data=hard_assign, index=adata.obs_names, columns=adata.var_names
            )
            for i, s in enumerate(state_cats):
                hard_assign = hard_assign.replace(i, s)

            states = hard_assign

        return states, state_cats
    
    def nonzero_terms(self):
        """Return indices of active terms.
           Active terms are the terms which were not deactivated by the group lasso regularization.
        """
        return self.module.decoder.nonzero_terms()

    def latent_enrich(
        self,
        groups,
        comparison='rest',
        n_sample=5000,
        use_directions=False,
        directions_key='directions',
        select_terms=None,
        adata=None,
        exact=True,
        key_added='bf_scores'
    ):
        """Gene set enrichment test for the latent space. Test the hypothesis that latent scores
           for each term in one group (z_1) is bigger than in the other group (z_2).

           Puts results to `adata.uns[key_added]`. Results are a dictionary with
           `p_h0` - probability that z_1 > z_2, `p_h1 = 1-p_h0` and `bf` - bayes factors equal to `log(p_h0/p_h1)`.

           Parameters
           ----------
           groups: String or Dict
                A string with the key in `adata.obs` to look for categories or a dictionary
                with categories as keys and lists of cell names as values.
           comparison: String
                The category name to compare against. If 'rest', then compares each category against all others.
           n_sample: Integer
                Number of random samples to draw for each category.
           use_directions: Boolean
                If 'True', multiplies the latent scores by directions in `adata`.
           directions_key: String
                The key in `adata.uns` for directions.
           select_terms: Array
                If not 'None', then an index of terms to select for the test. Only does the test
                for these terms.
           adata: AnnData
                An AnnData object to use. If 'None', uses `self.adata`.
           exact: Boolean
                Use exact probabilities for comparisons.
           key_added: String
                key of adata.uns where to put the results of the test.
        """
        if adata is None:
            adata = self.adata

        if isinstance(groups, str):
            cats_col = adata.obs[groups]
            cats = cats_col.unique()
        elif isinstance(groups, dict):
            cats = []
            all_cells = []
            for group, cells in groups.items():
                cats.append(group)
                all_cells += cells
            adata = adata[all_cells]
            cats_col = pd.Series(index=adata.obs_names, dtype=str)
            for group, cells in groups.items():
                cats_col[cells] = group
        else:
            raise ValueError("groups should be a string or a dict.")

        if comparison != "rest" and isinstance(comparison, str):
            comparison = [comparison]

        if comparison != "rest" and not set(comparison).issubset(cats):
            raise ValueError("comparison should be 'rest' or among the passed groups")

        scores = {}

        for cat in cats:
            if cat in comparison:
                continue

            cat_mask = cats_col == cat
            if comparison == "rest":
                others_mask = ~cat_mask
            else:
                others_mask = cats_col.isin(comparison)

            choice_1 = np.random.choice(cat_mask.sum(), n_sample)
            choice_2 = np.random.choice(others_mask.sum(), n_sample)

            adata_cat = adata[cat_mask][choice_1]
            adata_others = adata[others_mask][choice_2]

            if use_directions:
                directions = adata.uns[directions_key]
            else:
                directions = None

            z0 = self.get_latent(
                adata_cat,
                mean=False,
                mean_var=exact
            )

            z1 = self.get_latent(
                adata_others,
                mean=False,
                mean_var=exact
            )

            if not exact:
                if directions is not None:
                    z0 *= directions
                    z1 *= directions

                if select_terms is not None:
                    z0 = z0[:, select_terms]
                    z1 = z1[:, select_terms]

                to_reduce = z0 > z1

                zeros_mask = (np.abs(z0).sum(0) == 0) | (np.abs(z1).sum(0) == 0)
            else:
                from scipy.special import erfc

                means0, vars0 = z0
                means1, vars1 = z1

                if directions is not None:
                    means0 *= directions
                    means1 *= directions

                if select_terms is not None:
                    means0 = means0[:, select_terms]
                    means1 = means1[:, select_terms]
                    vars0 = vars0[:, select_terms]
                    vars1 = vars1[:, select_terms]

                to_reduce = (means1 - means0) / np.sqrt(2 * (vars0 + vars1))
                to_reduce = 0.5 * erfc(to_reduce)

                zeros_mask = (np.abs(means0).sum(0) == 0) | (np.abs(means1).sum(0) == 0)

            p_h0 = np.mean(to_reduce, axis=0)
            p_h1 = 1.0 - p_h0
            epsilon = 1e-12
            bf = np.log(p_h0 + epsilon) - np.log(p_h1 + epsilon)

            p_h0[zeros_mask] = 0
            p_h1[zeros_mask] = 0
            bf[zeros_mask] = 0

            scores[cat] = dict(p_h0=p_h0, p_h1=p_h1, bf=bf)

        adata.uns[key_added] = scores

    def term_genes(self, term: Union[str, int], terms: Union[str, list]='terms'):
        """Return the dataframe with genes belonging to the term after training sorted by absolute weights in the decoder.
        """
        if isinstance(terms, str):
            terms = list(self.adata.uns[terms])
        else:
            terms = list(terms)

        # if len(terms) == self.latent_dim_:
        #     if self.n_ext_m_ > 0:
        #         terms += ['constrained_' + str(i) for i in range(self.n_ext_m_)]
        #     if self.n_ext_ > 0:
        #         terms += ['unconstrained_' + str(i) for i in range(self.n_ext_)]

        #lat_mask_dim = self.latent_dim_# + self.n_ext_m_

        # if len(terms) != self.latent_dim_ and len(terms) != lat_mask_dim + self.n_ext_:
        #     raise ValueError('The list of terms should have the same length as the mask.')

        term = terms.index(term) if isinstance(term, str) else term

        #if term < self.latent_dim_:
        weights = self.module.decoder.L0.expr_L.weight[:, term].data.cpu().numpy()
        mask_idx = self.mask_[term]
        # elif term >= lat_mask_dim:
        #     term -= lat_mask_dim
        #     weights = self.model.decoder.L0.ext_L.weight[:, term].data.cpu().numpy()
        #     mask_idx = None
        # else:
        #     term -= self.latent_dim_
        #     weights = self.model.decoder.L0.ext_L_m.weight[:, term].data.cpu().numpy()
        #     mask_idx = self.ext_mask_[term]

        abs_weights = np.abs(weights)
        srt_idx = np.argsort(abs_weights)[::-1][:(abs_weights > 0).sum()]

        result = pd.DataFrame()
        result['genes'] = self.adata.var_names[srt_idx].tolist()
        result['weights'] = weights[srt_idx]
        result['in_mask'] = False

        if mask_idx is not None:
            in_mask = np.isin(srt_idx, np.where(mask_idx)[0])
            result['in_mask'][in_mask] = True

        return result

    @torch.no_grad()
    def get_latent(
    self,
    adata: Optional[AnnData] = None,
    indices: Optional[Sequence[int]] = None,
    gene_list: Optional[Sequence[str]] = None,
    n_samples: int = 1,
    n_samples_overall: Optional[int] = None,
    batch_size: Optional[int] = None,
    return_mean: bool = True,
    return_numpy: Optional[bool] = None,
    only_active: bool = False,
    mean: bool = False,
    mean_var: bool = False):

        """Map `x` in to the latent space. This function will feed data in encoder
            and return z for each sample in data.
            Parameters
            ----------
            x
                Numpy nd-array to be mapped to latent space. `x` has to be in shape [n_obs, input_dim].
                If None, then `self.adata.X` is used.
            c
                `numpy nd-array` of original (unencoded) desired labels for each sample.
            only_active
                Return only the latent variables which correspond to active terms, i.e terms that
                were not deactivated by the group lasso regularization.
            mean
                return mean instead of random sample from the latent space
            mean_var
                return mean and variance instead of random sample from the latent space
                if `mean=False`.
            Returns
            -------
                Returns array containing latent space encoding of 'x'.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )


        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        latent = []
        qz_m_list =[]
        qz_v_list=[]
        for tensors in scdl:
            inference_outputs= self.module.inference(
                spliced=tensors["X"],
                unspliced=tensors["U"],
                n_samples=n_samples,
            )

            if mean_var:
                qz_m=inference_outputs["qz_m"]
                qz_v=inference_outputs["qz_v"]
                qz_m = qz_m.cpu().numpy()
                qz_v = qz_v.cpu().numpy()
                qz_m_list.append(qz_m)
                qz_v_list.append(qz_v)

            else:
                #n_samples x cells x GPs
                z = inference_outputs["z"]
                z = z.cpu().numpy()
                latent.append(z)
                if return_mean and n_samples > 1:
                    latent[-1] = np.mean(latent[-1], axis=0)

            
        if mean_var:
            z_means = np.concatenate(qz_m_list, axis=0)
            z_vars = np.concatenate(qz_v_list, axis=0)
            latent=(z_means,z_vars)

        else:
            latent = np.concatenate(latent, axis=0)

        if only_active and mean_var is False:
            active_idx = self.nonzero_terms()
            latent = latent[:, active_idx]
            return latent
        else:
            return latent


        # if return_numpy is None or return_numpy is False:
        #     return pd.DataFrame(
        #         latent,
        #         columns=adata.var_names,
        #         index=adata.obs_names[indices],
        #     )
        # else:
        #     return latent

    def latent_directions(self, method="sum", get_confidence=False,
                          adata=None, key_added='directions'):
        """Get directions of upregulation for each latent dimension.
           Multipling this by raw latent scores ensures positive latent scores correspond to upregulation.
           Parameters
           ----------
           method: String
                Method of calculation, it should be 'sum' or 'counts'.
           get_confidence: Boolean
                Only for method='counts'. If 'True', also calculate confidence
                of the directions.
           adata: AnnData
                An AnnData object to store dimensions. If 'None', self.adata is used.
           key_added: String
                key of adata.uns where to put the dimensions.
        """
        if adata is None:
            adata = self.adata

        terms_weights = self.module.decoder.L0.expr_L.weight.data
        # if self.n_ext_m_ > 0:
        #     terms_weights = torch.cat([terms_weights, self.module.decoder.L0.ext_L_m.weight.data], dim=1)
        # if self.n_ext_ > 0:
        #     terms_weights = torch.cat([terms_weights, self.module.decoder.L0.ext_L.weight.data], dim=1)

        if method == "sum":
            signs = terms_weights.sum(0).cpu().numpy()
            signs[signs>0] = 1.
            signs[signs<0] = -1.
            confidence = None
        elif method == "counts":
            num_nz = torch.count_nonzero(terms_weights, dim=0)
            upreg_genes = torch.count_nonzero(terms_weights > 0, dim=0)
            signs = upreg_genes / (num_nz+(num_nz==0))
            signs = signs.cpu().numpy()

            confidence = signs.copy()
            confidence = np.abs(confidence-0.5)/0.5
            confidence[num_nz==0] = 0

            signs[signs>0.5] = 1.
            signs[signs<0.5] = -1.

            signs[signs==0.5] = 0
            signs[num_nz==0] = 0
        else:
            raise ValueError("Unrecognized method for getting the latent direction.")

        adata.uns[key_added] = signs
        if get_confidence and confidence is not None:
            adata.uns[key_added + '_confindence'] = confidence

        

    @torch.no_grad()
    def get_latent_time(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        time_statistic: Literal["mean", "max"] = "mean",
        n_samples: int = 1,
        n_samples_overall: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Returns the cells by genes latent time.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        time_statistic
            Whether to compute expected time over states, or maximum a posteriori time over maximal
            probability state.
        n_samples
            Number of posterior samples to use for estimation.
        n_samples_overall
            Number of overall samples to return. Setting this forces n_samples=1.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        times = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                pi = generative_outputs["px_pi"]
                ind_prob = pi[..., 0]
                steady_prob = pi[..., 1]
                rep_prob = pi[..., 2]
                # rep_steady_prob = pi[..., 3]
                switch_time = F.softplus(self.module.switch_time_unconstr)

                ind_time = generative_outputs["px_rho"] * switch_time
                rep_time = switch_time + (
                    generative_outputs["px_tau"] * (self.module.t_max - switch_time)
                )

                if time_statistic == "mean":
                    output = (
                        ind_prob * ind_time
                        + rep_prob * rep_time
                        + steady_prob * switch_time
                        # + rep_steady_prob * self.module.t_max
                    )
                else:
                    t = torch.stack(
                        [
                            ind_time,
                            switch_time.expand(ind_time.shape),
                            rep_time,
                            torch.zeros_like(ind_time),
                        ],
                        dim=2,
                    )
                    max_prob = torch.amax(pi, dim=-1)
                    max_prob = torch.stack([max_prob] * 4, dim=2)
                    max_prob_mask = pi.ge(max_prob)
                    output = (t * max_prob_mask).sum(dim=-1)

                output = output[..., gene_mask]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes by four
            times.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                times[-1] = np.mean(times[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            times = np.concatenate(times, axis=-2)
        else:
            times = np.concatenate(times, axis=0)

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                times,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
        else:
            return times

    @torch.no_grad()
    def get_velocity(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        n_samples: int = 1,
        n_samples_overall: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
        velo_statistic: str = "mean",
        velo_mode: Literal["spliced", "unspliced"] = "spliced",
        clip: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Returns cells by genes velocity estimates.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return velocities for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Number of posterior samples to use for estimation for each cell.
        n_samples_overall
            Number of overall samples to return. Setting this forces n_samples=1.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.
        velo_statistic
            Whether to compute expected velocity over states, or maximum a posteriori velocity over maximal
            probability state.
        velo_mode
            Compute ds/dt or du/dt.
        clip
            Clip to minus spliced value

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
            n_samples = 1
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        velos = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                inference_outputs, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                pi = generative_outputs["px_pi"]
                alpha = inference_outputs["alpha"]
                alpha_1 = inference_outputs["alpha_1"]
                lambda_alpha = inference_outputs["lambda_alpha"]
                beta = inference_outputs["beta"]
                gamma = inference_outputs["gamma"]
                tau = generative_outputs["px_tau"]
                rho = generative_outputs["px_rho"]

                ind_prob = pi[..., 0]
                steady_prob = pi[..., 1]
                rep_prob = pi[..., 2]
                switch_time = F.softplus(self.module.switch_time_unconstr)

                ind_time = switch_time * rho
                u_0, s_0 = self.module._get_induction_unspliced_spliced(
                    alpha, alpha_1, lambda_alpha, beta, gamma, switch_time
                )
                rep_time = (self.module.t_max - switch_time) * tau
                mean_u_rep, mean_s_rep = self.module._get_repression_unspliced_spliced(
                    u_0,
                    s_0,
                    beta,
                    gamma,
                    rep_time,
                )
                if velo_mode == "spliced":
                    velo_rep = beta * mean_u_rep - gamma * mean_s_rep
                else:
                    velo_rep = -beta * mean_u_rep
                mean_u_ind, mean_s_ind = self.module._get_induction_unspliced_spliced(
                    alpha, alpha_1, lambda_alpha, beta, gamma, ind_time
                )
                if velo_mode == "spliced":
                    velo_ind = beta * mean_u_ind - gamma * mean_s_ind
                else:
                    transcription_rate = alpha_1 - (alpha_1 - alpha) * torch.exp(
                        -lambda_alpha * ind_time
                    )
                    velo_ind = transcription_rate - beta * mean_u_ind

                if velo_mode == "spliced":
                    # velo_steady = beta * u_0 - gamma * s_0
                    velo_steady = torch.zeros_like(velo_ind)
                else:
                    # velo_steady = alpha - beta * u_0
                    velo_steady = torch.zeros_like(velo_ind)

                # expectation
                if velo_statistic == "mean":
                    output = (
                        ind_prob * velo_ind
                        + rep_prob * velo_rep
                        + steady_prob * velo_steady
                    )
                # maximum
                else:
                    v = torch.stack(
                        [
                            velo_ind,
                            velo_steady.expand(velo_ind.shape),
                            velo_rep,
                            torch.zeros_like(velo_rep),
                        ],
                        dim=2,
                    )
                    max_prob = torch.amax(pi, dim=-1)
                    max_prob = torch.stack([max_prob] * 4, dim=2)
                    max_prob_mask = pi.ge(max_prob)
                    output = (v * max_prob_mask).sum(dim=-1)

                output = output[..., gene_mask]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes
            velos.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                # mean over samples axis
                velos[-1] = np.mean(velos[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            velos = np.concatenate(velos, axis=-2)
        else:
            velos = np.concatenate(velos, axis=0)

        spliced = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)

        if clip:
            velos = np.clip(velos, -spliced[indices], None)

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                velos,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
        else:
            return velos

    @torch.no_grad()
    def get_velocity_from_latent(
        self,
        latent_representation: np.ndarray,
        return_numpy: Optional[bool] = None,
        velo_statistic: str = "mean",
        velo_mode: Literal["spliced", "unspliced"] = "spliced",
        clip: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame]:
        r"""
        Returns the normalized (decoded) gene expression.

        This is denoted as :math:`\rho_n` in the scVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.
        clip
            Clip to minus spliced value

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = AnnData(latent_representation)
        data_key = "Z"
        manager = AnnDataManager(
            [LayerField(data_key, layer=None, is_count_data=False)]
        )
        manager.register_fields(adata)
        scdl = AnnDataLoader(manager)

        gamma, beta, alpha, alpha_1, lambda_alpha = self.module._get_rates()

        velos = []
        for tensors in scdl:
            z = tensors[data_key]
            generative_outputs = self.module.generative(
                z=z,
                gamma=gamma,
                beta=beta,
                alpha=alpha,
                alpha_1=alpha_1,
                lambda_alpha=lambda_alpha,
            )
            pi = generative_outputs["px_pi"]
            tau = generative_outputs["px_tau"]
            rho = generative_outputs["px_rho"]

            ind_prob = pi[..., 0]
            steady_prob = pi[..., 1]
            rep_prob = pi[..., 2]
            switch_time = F.softplus(self.module.switch_time_unconstr)

            ind_time = switch_time * rho
            u_0, s_0 = self.module._get_induction_unspliced_spliced(
                alpha, alpha_1, lambda_alpha, beta, gamma, switch_time
            )
            rep_time = (self.module.t_max - switch_time) * tau
            mean_u_rep, mean_s_rep = self.module._get_repression_unspliced_spliced(
                u_0,
                s_0,
                beta,
                gamma,
                rep_time,
            )
            if velo_mode == "spliced":
                velo_rep = beta * mean_u_rep - gamma * mean_s_rep
            else:
                velo_rep = -beta * mean_u_rep
            mean_u_ind, mean_s_ind = self.module._get_induction_unspliced_spliced(
                alpha, alpha_1, lambda_alpha, beta, gamma, ind_time
            )
            if velo_mode == "spliced":
                velo_ind = beta * mean_u_ind - gamma * mean_s_ind
            else:
                transcription_rate = alpha_1 - (alpha_1 - alpha) * torch.exp(
                    -lambda_alpha * ind_time
                )
                velo_ind = transcription_rate - beta * mean_u_ind

            if velo_mode == "spliced":
                # velo_steady = beta * u_0 - gamma * s_0
                velo_steady = torch.zeros_like(velo_ind)
            else:
                # velo_steady = alpha - beta * u_0
                velo_steady = torch.zeros_like(velo_ind)

            # expectation
            if velo_statistic == "mean":
                output = (
                    ind_prob * velo_ind
                    + rep_prob * velo_rep
                    + steady_prob * velo_steady
                )
            # maximum
            else:
                v = torch.stack(
                    [
                        velo_ind,
                        velo_steady.expand(velo_ind.shape),
                        velo_rep,
                        torch.zeros_like(velo_rep),
                    ],
                    dim=2,
                )
                max_prob = torch.amax(pi, dim=-1)
                max_prob = torch.stack([max_prob] * 4, dim=2)
                max_prob_mask = pi.ge(max_prob)
                output = (v * max_prob_mask).sum(dim=-1)

            # samples by cells by genes
            velos.append(output.cpu().numpy())

        velos = np.concatenate(velos, axis=0)

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                velos,
                columns=self.adata.var_names,
            )
        else:
            return velos

    @torch.no_grad()
    def get_expression_fit(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
        restrict_to_latent_dim: Optional[int] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        r"""
        Returns the fitted spliced and unspliced abundance (s(t) and u(t)).

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        fits_s = []
        fits_u = []
        for tensors in scdl:
            minibatch_samples_s = []
            minibatch_samples_u = []
            for _ in range(n_samples):
                inference_outputs, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                    generative_kwargs=dict(latent_dim=restrict_to_latent_dim),
                )

                gamma = inference_outputs["gamma"]
                beta = inference_outputs["beta"]
                alpha = inference_outputs["alpha"]
                alpha_1 = inference_outputs["alpha_1"]
                lambda_alpha = inference_outputs["lambda_alpha"]
                px_pi = generative_outputs["px_pi"]
                scale = generative_outputs["scale"]
                px_rho = generative_outputs["px_rho"]
                px_tau = generative_outputs["px_tau"]

                (mixture_dist_s, mixture_dist_u, _,) = self.module.get_px(
                    px_pi,
                    px_rho,
                    px_tau,
                    scale,
                    gamma,
                    beta,
                    alpha,
                    alpha_1,
                    lambda_alpha,
                )
                fit_s = mixture_dist_s.mean
                fit_u = mixture_dist_u.mean

                fit_s = fit_s[..., gene_mask]
                fit_s = fit_s.cpu().numpy()
                fit_u = fit_u[..., gene_mask]
                fit_u = fit_u.cpu().numpy()

                minibatch_samples_s.append(fit_s)
                minibatch_samples_u.append(fit_u)

            # samples by cells by genes
            fits_s.append(np.stack(minibatch_samples_s, axis=0))
            if return_mean:
                # mean over samples axis
                fits_s[-1] = np.mean(fits_s[-1], axis=0)
            # samples by cells by genes
            fits_u.append(np.stack(minibatch_samples_u, axis=0))
            if return_mean:
                # mean over samples axis
                fits_u[-1] = np.mean(fits_u[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            fits_s = np.concatenate(fits_s, axis=-2)
            fits_u = np.concatenate(fits_u, axis=-2)
        else:
            fits_s = np.concatenate(fits_s, axis=0)
            fits_u = np.concatenate(fits_u, axis=0)

        if return_numpy is None or return_numpy is False:
            df_s = pd.DataFrame(
                fits_s,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
            df_u = pd.DataFrame(
                fits_u,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
            return df_s, df_u
        else:
            return fits_s, fits_u

    @torch.no_grad()
    def get_gene_likelihood(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        r"""
        Returns the likelihood per gene. Higher is better.

        This is denoted as :math:`\rho_n` in the scVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of relevant
            magnitude. If set to `"latent"`, use the latent libary size.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        rls = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                inference_outputs, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                spliced = tensors[REGISTRY_KEYS.X_KEY]
                unspliced = tensors[REGISTRY_KEYS.U_KEY]

                gamma = inference_outputs["gamma"]
                beta = inference_outputs["beta"]
                alpha = inference_outputs["alpha"]
                alpha_1 = inference_outputs["alpha_1"]
                lambda_alpha = inference_outputs["lambda_alpha"]
                px_pi = generative_outputs["px_pi"]
                scale = generative_outputs["scale"]
                px_rho = generative_outputs["px_rho"]
                px_tau = generative_outputs["px_tau"]
                dec_mean = generative_outputs["gene_recon"]
                dec_lat = generative_outputs["dec_latent"]

                (mixture_dist_s, mixture_dist_u, _,) = self.module.get_px(
                    px_pi,
                    px_rho,
                    px_tau,
                    scale,
                    gamma,
                    beta,
                    alpha,
                    alpha_1,
                    lambda_alpha,
                )
                reconst_loss_s = -mixture_dist_s.log_prob(spliced)
                reconst_loss_u = -mixture_dist_u.log_prob(unspliced)
                output = -(reconst_loss_s + reconst_loss_u)
                output = output[..., gene_mask]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes by four
            rls.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                rls[-1] = np.mean(rls[-1], axis=0)

        rls = np.concatenate(rls, axis=0)
        return rls.shape, dec_mean.shape, dec_lat.shape

    @torch.no_grad()
    def get_rates(self, mean: bool = True):

        gamma, beta, alpha, alpha_1, lambda_alpha = self.module._get_rates()

        return {
            "beta": beta.cpu().numpy(),
            "gamma": gamma.cpu().numpy(),
            "alpha": alpha.cpu().numpy(),
            "alpha_1": alpha_1.cpu().numpy(),
            "lambda_alpha": lambda_alpha.cpu().numpy(),
        }

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        spliced_layer: str,
        unspliced_layer: str,
        **kwargs,
    ) -> Optional[AnnData]:
        """
        %(summary)s.
        Parameters
        ----------
        %(param_adata)s
        spliced_layer
            Layer in adata with spliced normalized expression
        unspliced_layer
            Layer in adata with unspliced normalized expression

        Returns
        -------
        %(returns)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, spliced_layer, is_count_data=False),
            LayerField(REGISTRY_KEYS.U_KEY, unspliced_layer, is_count_data=False),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.no_grad()
    @_doc_params(
        doc_differential_expression=doc_differential_expression,
    )
    def differential_velocity(
        self,
        adata: Optional[AnnData] = None,
        groupby: Optional[str] = None,
        group1: Optional[Iterable[str]] = None,
        group2: Optional[str] = None,
        idx1: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
        idx2: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
        mode: Literal["vanilla", "change"] = "vanilla",
        delta: float = 0.25,
        batch_size: Optional[int] = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Optional[Iterable[str]] = None,
        batchid2: Optional[Iterable[str]] = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        r"""
        A unified method for differential velocity analysis.

        Implements `"vanilla"` DE [Lopez18]_ and `"change"` mode DE [Boyeau19]_.

        Parameters
        ----------
        {doc_differential_expression}
        **kwargs
            Keyword args for :meth:`scvi.model.base.DifferentialComputation.get_bayes_factors`

        Returns
        -------
        Differential expression DataFrame.
        """
        adata = self._validate_anndata(adata)

        def model_fn(adata, **kwargs):
            if "transform_batch" in kwargs.keys():
                kwargs.pop("transform_batch")
            return partial(
                self.get_velocity,
                batch_size=batch_size,
                n_samples=1,
                return_numpy=True,
                clip=False,
            )(adata, **kwargs)

        col_names = adata.var_names

        result = _de_core(
            self.get_anndata_manager(adata, required=True),
            model_fn,
            groupby,
            group1,
            group2,
            idx1,
            idx2,
            all_stats,
            scrna_raw_counts_properties,
            col_names,
            mode,
            batchid1,
            batchid2,
            delta,
            batch_correction,
            fdr_target,
            silent,
            **kwargs,
        )

        return result

    @torch.no_grad()
    def differential_transition(
        self,
        groupby: str,
        group1: str,
        group2: str,
        adata: Optional[AnnData] = None,
        batch_size: Optional[int] = None,
        n_samples: Optional[int] = 5000,
    ) -> pd.DataFrame:
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)

        if not isinstance(group1, str):
            raise ValueError("Group 1 must be a string")

        cell_idx1 = (adata.obs[groupby] == group1).to_numpy().ravel()
        if group2 is None:
            cell_idx2 = ~cell_idx1
        else:
            cell_idx2 = (adata.obs[groupby] == group2).to_numpy().ravel()

        indices1 = np.random.choice(
            np.asarray(np.where(cell_idx1)[0].ravel()), n_samples
        )
        indices2 = np.random.choice(
            np.asarray(np.where(cell_idx2)[0].ravel()), n_samples
        )

        velo1 = self.get_velocity(
            adata,
            return_numpy=True,
            indices=indices1,
            n_samples=1,
            batch_size=batch_size,
        )
        velo1 = velo1 - velo1.mean(1)[:, np.newaxis]
        velo2 = self.get_velocity(
            adata,
            return_numpy=True,
            indices=indices2,
            n_samples=1,
            batch_size=batch_size,
        )
        velo2 = velo2 - velo2.mean(1)[:, np.newaxis]

        spliced = adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        delta12 = spliced[indices2] - spliced[indices1]
        delta12 = delta12 - delta12.mean(1)[:, np.newaxis]

        delta21 = spliced[indices1] - spliced[indices2]
        delta21 = delta21 - delta21.mean(1)[:, np.newaxis]

        # TODO: Make more efficient
        correlation12 = np.diagonal(cosine_similarity(velo1, delta12))
        correlation21 = np.diagonal(cosine_similarity(velo2, delta21))

        return correlation12, correlation21

    def get_loadings(self) -> pd.DataFrame:
        """
        Extract per-gene weights in the linear decoder.

        Shape is genes by `n_latent`.
        """
        cols = ["Z_{}".format(i) for i in range(self.n_latent)]
        var_names = self.adata.var_names
        loadings = pd.DataFrame(
            self.module.get_loadings(), index=var_names, columns=cols
        )

        return loadings

    def get_variance_explained(
        self,
        adata: Optional[AnnData] = None,
        labels_key: Optional[str] = None,
        n_samples: int = 10,
    ) -> pd.DataFrame:

        if self.module.decoder.linear_decoder is False:
            raise ValueError("Model not trained with linear decoder")
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata)
        n_latent = self.module.n_latent

        if labels_key is not None:
            groups = np.unique(adata.obs[labels_key])
        else:
            groups = [None]

        spliced = adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        unspliced = adata_manager.get_from_registry(REGISTRY_KEYS.U_KEY)

        centered_s = spliced - spliced.mean(0)
        centered_u = unspliced - unspliced.mean(0)

        def r_squared(true_s, pred_s, true_u, pred_u, centered_s, centered_u):
            rss_s = np.sum((true_s - pred_s) ** 2)
            tss_s = np.sum(centered_s**2)
            rss_u = np.sum((true_u - pred_u) ** 2)
            tss_u = np.sum(centered_u**2)

            return (1 - (rss_s + rss_u) / (tss_s + tss_u)) * 100

        df_out = pd.DataFrame(
            data=np.zeros((n_latent, len(groups))),
            index=[f"Z_{i}" for i in range(n_latent)],
            columns=groups if groups[0] is not None else ["0"],
        )

        for i in range(n_latent):
            fitted_s, fitted_u = self.get_expression_fit(
                adata, restrict_to_latent_dim=i, n_samples=n_samples, return_numpy=True
            )
            for j, g in enumerate(groups):
                if g is None:
                    subset = slice(None)
                else:
                    subset = adata.obs[labels_key] == g
                r_2 = r_squared(
                    spliced[subset],
                    fitted_s[subset],
                    unspliced[subset],
                    fitted_u[subset],
                    centered_s[subset],
                    centered_u[subset],
                )
                df_out.iloc[i, j] = r_2

        return df_out

    def get_directional_uncertainty(
        self,
        adata: Optional[AnnData] = None,
        n_samples: int = 50,
        gene_list: Iterable[str] = None,
        n_jobs: int = -1,
    ):

        adata = self._validate_anndata(adata)

        logger.info("Sampling from model...")
        velocities_all = self.get_velocity(
            n_samples=n_samples, return_mean=False, gene_list=gene_list
        )  # (n_samples, n_cells, n_genes)

        df, cosine_sims = _compute_directional_statistics_tensor(
            tensor=velocities_all, n_jobs=n_jobs, n_cells=adata.n_obs
        )
        df.index = adata.obs_names

        return df, cosine_sims

    def get_permutation_scores(
        self, labels_key: str, adata: Optional[AnnData] = None
    ) -> Tuple[pd.DataFrame, AnnData]:
        """
        Compute permutation scores.

        Parameters
        ----------
        labels_key
            Key in adata.obs encoding cell types
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.

        Returns
        -------
        Tuple of DataFrame and AnnData. DataFrame is genes by cell types with score per cell type.
        AnnData is the permutated version of the original AnnData.
        """
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata)
        if labels_key not in adata.obs:
            raise ValueError(f"{labels_key} not found in adata.obs")

        # shuffle spliced then unspliced
        bdata = self._shuffle_layer_celltype(
            adata_manager, labels_key, REGISTRY_KEYS.X_KEY
        )
        bdata_manager = self.get_anndata_manager(bdata)
        bdata = self._shuffle_layer_celltype(
            bdata_manager, labels_key, REGISTRY_KEYS.U_KEY
        )
        bdata_manager = self.get_anndata_manager(bdata)

        ms_ = adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        mu_ = adata_manager.get_from_registry(REGISTRY_KEYS.U_KEY)

        ms_p = bdata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        mu_p = bdata_manager.get_from_registry(REGISTRY_KEYS.U_KEY)

        spliced_, unspliced_ = self.get_expression_fit(adata, n_samples=10)
        root_squared_error = np.abs(spliced_ - ms_)
        root_squared_error += np.abs(unspliced_ - mu_)

        spliced_p, unspliced_p = self.get_expression_fit(bdata, n_samples=10)
        root_squared_error_p = np.abs(spliced_p - ms_p)
        root_squared_error_p += np.abs(unspliced_p - mu_p)

        celltypes = np.unique(adata.obs[labels_key])

        dynamical_df = pd.DataFrame(
            index=adata.var_names,
            columns=celltypes,
            data=np.zeros((adata.shape[1], len(celltypes))),
        )
        N = 200
        for ct in celltypes:
            for g in adata.var_names.tolist():
                x = root_squared_error_p[g][adata.obs[labels_key] == ct]
                y = root_squared_error[g][adata.obs[labels_key] == ct]
                ratio = ttest_ind(x[:N], y[:N])[0]
                dynamical_df.loc[g, ct] = ratio

        return dynamical_df, bdata

    def _shuffle_layer_celltype(
        self, adata_manager: AnnDataManager, labels_key: str, registry_key: str
    ) -> AnnData:
        """Shuffle cells within cell types for each gene."""
        from scvi.data._constants import _SCVI_UUID_KEY

        bdata = adata_manager.adata.copy()
        labels = bdata.obs[labels_key]
        del bdata.uns[_SCVI_UUID_KEY]
        self._validate_anndata(bdata)
        bdata_manager = self.get_anndata_manager(bdata)

        # get registry info to later set data back in bdata
        # in a way that doesn't require actual knowledge of location
        unspliced = bdata_manager.get_from_registry(registry_key)
        u_registry = bdata_manager.data_registry[registry_key]
        attr_name = u_registry.attr_name
        attr_key = u_registry.attr_key

        for lab in np.unique(labels):
            mask = np.asarray(labels == lab)
            unspliced_ct = unspliced[mask].copy()
            unspliced_ct = np.apply_along_axis(
                np.random.permutation, axis=0, arr=unspliced_ct
            )
            unspliced[mask] = unspliced_ct
        # e.g., if using adata.X
        if attr_key is None:
            setattr(bdata, attr_name, unspliced)
        # e.g., if using a layer
        elif attr_key is not None:
            attribute = getattr(bdata, attr_name)
            attribute[attr_key] = unspliced
            setattr(bdata, attr_name, attribute)

        return bdata


def _compute_directional_statistics_tensor(
    tensor: np.ndarray, n_jobs: int, n_cells: int
) -> pd.DataFrame:
    df = pd.DataFrame(index=np.arange(n_cells))
    df["directional_variance"] = np.nan
    df["directional_difference"] = np.nan
    df["directional_cosine_sim_variance"] = np.nan
    df["directional_cosine_sim_difference"] = np.nan
    df["directional_cosine_sim_mean"] = np.nan
    logger.info("Computing the uncertainties...")
    results = Parallel(n_jobs=n_jobs, verbose=3)(
        delayed(_directional_statistics_per_cell)(tensor[:, cell_index, :])
        for cell_index in range(n_cells)
    )
    # cells by samples
    cosine_sims = np.stack([results[i][0] for i in range(n_cells)])
    df.loc[:, "directional_cosine_sim_variance"] = [
        results[i][1] for i in range(n_cells)
    ]
    df.loc[:, "directional_cosine_sim_difference"] = [
        results[i][2] for i in range(n_cells)
    ]
    df.loc[:, "directional_variance"] = [results[i][3] for i in range(n_cells)]
    df.loc[:, "directional_difference"] = [results[i][4] for i in range(n_cells)]
    df.loc[:, "directional_cosine_sim_mean"] = [results[i][5] for i in range(n_cells)]

    return df, cosine_sims


def _directional_statistics_per_cell(
    tensor: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Internal function for parallelization.

    Parameters
    ----------
    tensor
        Shape of samples by genes for a given cell.
    """
    n_samples = tensor.shape[0]
    # over samples axis
    mean_velocity_of_cell = tensor.mean(0)
    cosine_sims = [
        _cosine_sim(tensor[i, :], mean_velocity_of_cell) for i in range(n_samples)
    ]
    angle_samples = [np.arccos(el) for el in cosine_sims]
    return (
        cosine_sims,
        np.var(cosine_sims),
        np.percentile(cosine_sims, 95) - np.percentile(cosine_sims, 5),
        np.var(angle_samples),
        np.percentile(angle_samples, 95) - np.percentile(angle_samples, 5),
        np.mean(cosine_sims),
    )


def _centered_unit_vector(vector: np.ndarray) -> np.ndarray:
    """Returns the centered unit vector of the vector."""
    vector = vector - np.mean(vector)
    return vector / np.linalg.norm(vector)


def _cosine_sim(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Returns cosine similarity of the vectors."""
    v1_u = _centered_unit_vector(v1)
    v2_u = _centered_unit_vector(v2)
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)