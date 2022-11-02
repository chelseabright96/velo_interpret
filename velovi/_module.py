# -*- coding: utf-8 -*-
"""Main module."""
from typing import Callable, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scvi._compat import Literal
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import Encoder, FCLayers
from torch import nn as nn
from torch.distributions import Categorical, Dirichlet, MixtureSameFamily, Normal
from torch.distributions import kl_divergence as kl
from scvi.distributions import NegativeBinomial

import logging
import warnings
from functools import partial
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from velovi import REGISTRY_KEYS
from ._utils import one_hot_encoder

logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True


class MaskedLinear(nn.Linear):
    def __init__(self, n_in,  n_out, mask, bias=True):
        # mask should have the same dimensions as the transposed linear weight
        # n_input x n_output_nodes
        if n_in != mask.shape[0] or n_out != mask.shape[1]:
            raise ValueError('Incorrect shape of the mask.')

        super().__init__(n_in, n_out, bias)

        self.register_buffer('mask', mask.t())

        # zero out the weights for group lasso
        # gradient descent won't change these zero weights
        self.weight.data*=self.mask

    def forward(self, input):
        return nn.functional.linear(input, self.weight*self.mask, self.bias)

class MaskedCondLayers(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cond: int,
        bias: bool,
        n_ext: int = 0,
        n_ext_m: int = 0,
        mask: Optional[torch.Tensor] = None,
        ext_mask: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.n_cond = n_cond
        self.n_ext = n_ext
        self.n_ext_m = n_ext_m

        self.expr_L = nn.Linear(n_in, n_out, bias=bias)

        # if mask is None:
        #     self.expr_L = nn.Linear(n_in, n_out, bias=bias)
        # else:
        #     self.expr_L = MaskedLinear(n_in, n_out, mask, bias=bias)

        # if self.n_cond != 0:
        #     self.cond_L = nn.Linear(self.n_cond, n_out, bias=False)

        # if self.n_ext != 0:
        #     self.ext_L = nn.Linear(self.n_ext, n_out, bias=False)

        # if self.n_ext_m != 0:
        #     if ext_mask is not None:
        #         self.ext_L_m = MaskedLinear(self.n_ext_m, n_out, ext_mask, bias=False)
        #     else:
        #         self.ext_L_m = nn.Linear(self.n_ext_m, n_out, bias=False)

    def forward(self, x: torch.Tensor):
        # if self.n_cond == 0:
        #     expr, cond = x, None
        # else:
        #     expr, cond = torch.split(x, [x.shape[1] - self.n_cond, self.n_cond], dim=1)

        # if self.n_ext == 0:
        #     ext = None
        # else:
        #     expr, ext = torch.split(expr, [expr.shape[1] - self.n_ext, self.n_ext], dim=1)

        # if self.n_ext_m == 0:
        #     ext_m = None
        # else:
        #     expr, ext_m = torch.split(expr, [expr.shape[1] - self.n_ext_m, self.n_ext_m], dim=1)

        expr=x

        out = self.expr_L(expr)
        # if ext is not None:
        #     out = out + self.ext_L(ext)
        # if ext_m is not None:
        #     out = out + self.ext_L_m(ext_m)
        # if cond is not None:
        #     out = out + self.cond_L(cond)
        return out


class DecoderVELOVI(nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions ``n_output``dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    linear_decoder
        Whether to use linear decoder for time
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_ext: int = 0,
        n_ext_m: int = 0,
        n_cond: int = 0,
        last_layer: str =None,
        ext_mask: torch.Tensor = None,
        mask: torch.Tensor = None,
        recon_loss: str = 'nb',
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        linear_decoder: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.n_ouput = n_output
        self.linear_decoder = linear_decoder

        ### GP decoder ###

        if recon_loss == "mse":
            if last_layer == "softmax":
                raise ValueError("Can't specify softmax last layer with mse loss.")
            last_layer = "identity" if last_layer is None else last_layer
        elif recon_loss == "nb":
            last_layer = "softmax" if last_layer is None else last_layer
        else:
            raise ValueError("Unrecognized loss.")

        #print("GP Decoder Architecture:")
        #print("\tMasked linear layer in, ext_m, ext, cond, out: ", in_dim, n_ext_m, n_ext, n_cond, out_dim)
        # if mask is not None:
        #     print('\twith hard mask.')
        # else:
        #     print('\twith soft mask.')

        self.n_ext = n_ext
        self.n_ext_m = n_ext_m

        self.n_cond = 0
        if n_cond is not None:
            self.n_cond = n_cond

        self.L0 = MaskedCondLayers(n_input, n_output, n_cond, bias=False, n_ext=n_ext, n_ext_m=n_ext_m,
                                   mask=mask, ext_mask=ext_mask)

        if last_layer == "softmax":
            self.mean_decoder = nn.Softmax(dim=-1)
        elif last_layer == "softplus":
            self.mean_decoder = nn.Softplus()
        elif last_layer == "exp":
            self.mean_decoder = torch.exp
        elif last_layer == "relu":
            self.mean_decoder = nn.ReLU()
        elif last_layer == "identity":
            self.mean_decoder = lambda a: a
        else:
            raise ValueError("Unrecognized last layer.")

        print("Last Decoder layer:", last_layer)

        self.rho_first_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden if not linear_decoder else n_output,
            n_cat_list=n_cat_list,
            n_layers=n_layers if not linear_decoder else 1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm if not linear_decoder else False,
            use_activation=not linear_decoder,
            bias=not linear_decoder,
            **kwargs,
        )

        self.pi_first_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **kwargs,
        )

        self.px_pi_decoder = nn.Linear(n_hidden, 4 * n_output)
        

        # rho for induction
        self.px_rho_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Sigmoid())

        # tau for repression
        self.px_tau_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Sigmoid())

        self.linear_scaling_tau = nn.Parameter(torch.zeros(n_output))
        self.linear_scaling_tau_intercept = nn.Parameter(torch.zeros(n_output))

    def forward(self, z: torch.Tensor, latent_dim: int = None):
        """
        The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        Parameters
        ----------
        z :
            tensor with shape ``(n_input,)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression

        """

        z_in = z
        if latent_dim is not None:
            mask = torch.zeros_like(z)
            mask[..., latent_dim] = 1
            z_in = z * mask
        # The decoder returns values for the parameters of the ZINB distribution
        rho_first = self.rho_first_decoder(z_in)

        dec_latent = self.L0(z)
        recon_x = self.mean_decoder(dec_latent)

        if not self.linear_decoder:
            px_rho = self.px_rho_decoder(rho_first)
            px_tau = self.px_tau_decoder(rho_first)
        else:
            px_rho = nn.Sigmoid()(rho_first)
            px_tau = 1 - nn.Sigmoid()(
                rho_first * self.linear_scaling_tau.exp()
                + self.linear_scaling_tau_intercept
            )

        # cells by genes by 4
        pi_first = self.pi_first_decoder(z)
        px_pi = nn.Softplus()(
            torch.reshape(self.px_pi_decoder(pi_first), (z.shape[0], self.n_ouput, 4))
        )

        return px_pi, px_rho, px_tau, recon_x, dec_latent

    def nonzero_terms(self):
        v = self.L0.expr_L.weight.data
        nz = (v.norm(p=1, dim=0)>0).cpu().numpy()
        nz = np.append(nz, np.full(self.n_ext_m, True))
        nz = np.append(nz, np.full(self.n_ext, True))
        return nz

    def n_inactive_terms(self):
        n = (~self.nonzero_terms()).sum()
        return int(n)

# VAE model
class VELOVAE(BaseModuleClass):
    """
    Variational auto-encoder model.

    This is an implementation of the scVI model descibed in [Lopez18]_

    Parameters
    ----------
    n_input
        Number of input genes
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    dropout_rate
        Dropout rate for neural networks
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    use_layer_norm
        Whether to use layer norm in layers
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When `None`, defaults to `torch.exp`.
    """

    def __init__(
        self,
        n_input: int,
        true_time_switch: Optional[np.ndarray] = None,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        log_variational: bool = False,
        latent_distribution: str = "normal",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_observed_lib_size: bool = True,
        var_activation: Optional[Callable] = torch.nn.Softplus(),
        model_steady_states: bool = True,
        gamma_unconstr_init: Optional[np.ndarray] = None,
        alpha_unconstr_init: Optional[np.ndarray] = None,
        alpha_1_unconstr_init: Optional[np.ndarray] = None,
        lambda_alpha_unconstr_init: Optional[np.ndarray] = None,
        switch_spliced: Optional[np.ndarray] = None,
        switch_unspliced: Optional[np.ndarray] = None,
        t_max: float = 20,
        penalty_scale: float = 0.2,
        dirichlet_concentration: float = 0.25,
        linear_decoder: bool = False,
        time_dep_transcription_rate: bool = False,
        #Parameters for masked linear decoder
        mask: torch.Tensor = None,
        recon_loss: str = 'nb',
        conditions: list = [],
        use_l_encoder: bool = False,
        dr_rate: float = 0.05,
        use_bn: bool = False,
        use_ln: bool = True,
        decoder_last_layer: Optional[str] = None,
        soft_mask: bool = False,
        n_ext: int = 0,
        n_ext_m: int = 0,
        use_hsic: bool = False,
        hsic_one_vs_all: bool = False,
        ext_mask: Optional[torch.Tensor] = None,
        soft_ext_mask: bool = False
    ):
        super().__init__()
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.latent_distribution = latent_distribution
        self.use_observed_lib_size = use_observed_lib_size
        self.n_input = n_input
        self.model_steady_states = model_steady_states
        self.t_max = t_max
        self.penalty_scale = penalty_scale
        self.dirichlet_concentration = dirichlet_concentration
        self.time_dep_transcription_rate = time_dep_transcription_rate

        if switch_spliced is not None:
            self.register_buffer("switch_spliced", torch.from_numpy(switch_spliced))
        else:
            self.switch_spliced = None
        if switch_unspliced is not None:
            self.register_buffer("switch_unspliced", torch.from_numpy(switch_unspliced))
        else:
            self.switch_unspliced = None

        n_genes = n_input * 2

        # switching time
        self.switch_time_unconstr = torch.nn.Parameter(7 + 0.5 * torch.randn(n_input))
        if true_time_switch is not None:
            self.register_buffer("true_time_switch", torch.from_numpy(true_time_switch))
        else:
            self.true_time_switch = None

        # degradation
        if gamma_unconstr_init is None:
            self.gamma_mean_unconstr = torch.nn.Parameter(-1 * torch.ones(n_input))
        else:
            self.gamma_mean_unconstr = torch.nn.Parameter(
                torch.from_numpy(gamma_unconstr_init)
            )

        # splicing
        # first samples around 1
        self.beta_mean_unconstr = torch.nn.Parameter(0.5 * torch.ones(n_input))

        # transcription
        if alpha_unconstr_init is None:
            self.alpha_unconstr = torch.nn.Parameter(0 * torch.ones(n_input))
        else:
            self.alpha_unconstr = torch.nn.Parameter(
                torch.from_numpy(alpha_unconstr_init)
            )

        # TODO: Add `require_grad`
        if alpha_1_unconstr_init is None:
            self.alpha_1_unconstr = torch.nn.Parameter(0 * torch.ones(n_input))
        else:
            self.alpha_1_unconstr = torch.nn.Parameter(
                torch.from_numpy(alpha_1_unconstr_init)
            )
        self.alpha_1_unconstr.requires_grad = time_dep_transcription_rate

        if lambda_alpha_unconstr_init is None:
            self.lambda_alpha_unconstr = torch.nn.Parameter(0 * torch.ones(n_input))
        else:
            self.lambda_alpha_unconstr = torch.nn.Parameter(
                torch.from_numpy(lambda_alpha_unconstr_init)
            )
        self.lambda_alpha_unconstr.requires_grad = time_dep_transcription_rate

        # likelihood dispersion
        # for now, with normal dist, this is just the variance
        self.scale_unconstr = torch.nn.Parameter(-1 * torch.ones(n_genes, 4))

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"
        self.use_batch_norm_decoder = use_batch_norm_decoder

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_genes
        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            activation_fn=torch.nn.ReLU,
        )

         ### Attributes for masked linear decoder
        self.n_conditions = len(conditions)
        self.conditions = conditions
        self.n_conditions=0
        self.recon_loss = recon_loss
        self.freeze = False
        self.use_bn = use_bn
        self.use_ln = use_ln

        self.use_mmd = False

        self.n_ext_encoder = n_ext + n_ext_m
        self.n_ext_decoder = n_ext
        self.n_ext_m_decoder = n_ext_m

        self.use_hsic = use_hsic and self.n_ext_decoder > 0
        self.hsic_one_vs_all = hsic_one_vs_all

        self.soft_mask = soft_mask and mask is not None
        self.soft_ext_mask = soft_ext_mask and ext_mask is not None

        if decoder_last_layer is None:
            if recon_loss == 'nb':
                self.decoder_last_layer = 'softmax'
            else:
                self.decoder_last_layer = 'identity'
        else:
            self.decoder_last_layer = decoder_last_layer

        self.use_l_encoder = use_l_encoder

        self.dr_rate = dr_rate
        if self.dr_rate > 0:
            self.use_dr = True
        else:
            self.use_dr = False

        if recon_loss == "nb":
            if self.n_conditions != 0:
                self.theta = torch.nn.Parameter(torch.randn(self.n_input, self.n_conditions))
            else:
                self.theta = torch.nn.Parameter(torch.randn(1, self.n_input))
        else:
            self.theta = None

        if self.soft_mask:
            self.n_inact_genes = (1-mask).sum().item()
            soft_shape = mask.shape
            if soft_shape[0] != n_latent or soft_shape[1] != n_input:
                raise ValueError('Incorrect shape of the soft mask.')
            self.mask = mask.t()
            mask = None
        else:
            self.mask = None

        if self.soft_ext_mask:
            self.n_inact_ext_genes = (1-ext_mask).sum().item()
            ext_shape = ext_mask.shape
            if ext_shape[0] != self.n_ext_m_decoder:
                raise ValueError('Dim 0 of ext_mask should be the same as n_ext_m_decoder.')
            if ext_shape[1] != self.n_input:
                raise ValueError('Dim 1 of ext_mask should be the same as n_input.')
            self.ext_mask = ext_mask.t()
            ext_mask = None
        else:
            self.ext_mask = None
            
        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent
        self.decoder = DecoderVELOVI(
            n_input_decoder,
            n_input,
            n_ext = 0,
            n_ext_m= 0,
            n_cond= 0,
            last_layer=None,
            ext_mask = None,
            mask = None,
            recon_loss = 'nb',
            n_cat_list= None,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            activation_fn=torch.nn.ReLU,
            linear_decoder=linear_decoder,
            )

       


    def _get_inference_input(self, tensors):
        spliced = tensors[REGISTRY_KEYS.X_KEY]
        unspliced = tensors[REGISTRY_KEYS.U_KEY]

        input_dict = dict(
            spliced=spliced,
            unspliced=unspliced,
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        gamma = inference_outputs["gamma"]
        beta = inference_outputs["beta"]
        alpha = inference_outputs["alpha"]
        alpha_1 = inference_outputs["alpha_1"]
        lambda_alpha = inference_outputs["lambda_alpha"]

        input_dict = {
            "z": z,
            "gamma": gamma,
            "beta": beta,
            "alpha": alpha,
            "alpha_1": alpha_1,
            "lambda_alpha": lambda_alpha,
        }
        return input_dict

    @auto_move_data
    def inference(
        self,
        spliced,
        unspliced,
        n_samples=1,
    ):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        spliced_ = spliced
        unspliced_ = unspliced
        if self.log_variational:
            spliced_ = torch.log(0.01 + spliced)
            unspliced_ = torch.log(0.01 + unspliced)

        encoder_input = torch.cat((spliced_, unspliced_), dim=-1)

        qz_m, qz_v, z = self.z_encoder(encoder_input)

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)

        gamma, beta, alpha, alpha_1, lambda_alpha = self._get_rates()

        outputs = dict(
            z=z,
            qz_m=qz_m,
            qz_v=qz_v,
            gamma=gamma,
            beta=beta,
            alpha=alpha,
            alpha_1=alpha_1,
            lambda_alpha=lambda_alpha,
        )
        return outputs

    def _get_rates(self):
        # globals
        # degradation
        gamma = torch.clamp(F.softplus(self.gamma_mean_unconstr), 0, 50)
        # splicing
        beta = torch.clamp(F.softplus(self.beta_mean_unconstr), 0, 50)
        # transcription
        alpha = torch.clamp(F.softplus(self.alpha_unconstr), 0, 50)
        if self.time_dep_transcription_rate:
            alpha_1 = torch.clamp(F.softplus(self.alpha_1_unconstr), 0, 50)
            lambda_alpha = torch.clamp(F.softplus(self.lambda_alpha_unconstr), 0, 50)
        else:
            alpha_1 = self.alpha_1_unconstr
            lambda_alpha = self.lambda_alpha_unconstr

        return gamma, beta, alpha, alpha_1, lambda_alpha

    @auto_move_data
    def generative(self, z, gamma, beta, alpha, alpha_1, lambda_alpha, latent_dim=None):
        """Runs the generative model."""
        decoder_input = z
        px_pi_alpha, px_rho, px_tau, dec_mean, dec_latent = self.decoder(decoder_input, latent_dim=latent_dim)

        px_pi = Dirichlet(px_pi_alpha).rsample()

        #dec_mean, dec_latent = self.GP_linear_decoder(decoder_input, batch=None)

        scale_unconstr = self.scale_unconstr
        scale = F.softplus(scale_unconstr)

        mixture_dist_s, mixture_dist_u, end_penalty = self.get_px(
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

        return dict(
            px_pi=px_pi,
            px_rho=px_rho,
            px_tau=px_tau,
            scale=scale,
            px_pi_alpha=px_pi_alpha,
            mixture_dist_u=mixture_dist_u,
            mixture_dist_s=mixture_dist_s,
            end_penalty=end_penalty,
            gene_recon = dec_mean,
            dec_latent = dec_latent
        )

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        cond_batch=None,
        kl_weight: float = 1.0,
        n_obs: float = 1.0,
    ):
        spliced = tensors[REGISTRY_KEYS.X_KEY]
        unspliced = tensors[REGISTRY_KEYS.U_KEY]

        #gene reconstruction loss
        ground_truth_counts = spliced + unspliced
        

        if cond_batch is not None:
            dispersion = F.linear(one_hot_encoder(cond_batch, self.n_conditions), self.theta) #batch is the
        else:
            dispersion = self.theta   
        dispersion = torch.exp(dispersion)

        dec_mean = generative_outputs["gene_recon"]
        negbin = NegativeBinomial(mu=dec_mean, theta=dispersion)
        
        gene_recon_loss = -negbin.log_prob(ground_truth_counts).sum(dim=-1)

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]

        px_pi = generative_outputs["px_pi"]
        px_pi_alpha = generative_outputs["px_pi_alpha"]

        end_penalty = generative_outputs["end_penalty"]
        mixture_dist_s = generative_outputs["mixture_dist_s"]
        mixture_dist_u = generative_outputs["mixture_dist_u"]

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)

        reconst_loss_s = -mixture_dist_s.log_prob(spliced)
        reconst_loss_u = -mixture_dist_u.log_prob(unspliced)
        reconst_loss = reconst_loss_u.sum(dim=-1) + reconst_loss_s.sum(dim=-1) 

        kl_pi = kl(
            Dirichlet(px_pi_alpha),
            Dirichlet(self.dirichlet_concentration * torch.ones_like(px_pi)),
        ).sum(dim=-1)

        # local loss
        kl_local = kl_divergence_z + kl_pi
        weighted_kl_local = kl_weight * (kl_divergence_z) + kl_pi

        local_loss = torch.mean(reconst_loss + gene_recon_loss + weighted_kl_local)
        print(f"local_loss: {local_loss}")

        # combine local and global
        global_loss = 0
        loss = (
            local_loss
            + self.penalty_scale * (1 - kl_weight) * end_penalty
            + (1 / n_obs) * kl_weight * (global_loss)
        )
        print(f"loss: {loss}")
        loss_recorder = LossRecorder(
            loss, reconst_loss, kl_local, torch.tensor(global_loss)
        )

        return loss_recorder


    @auto_move_data
    def get_px(
        self,
        px_pi,
        px_rho,
        px_tau,
        scale,
        gamma,
        beta,
        alpha,
        alpha_1,
        lambda_alpha,
    ) -> torch.Tensor:

        t_s = torch.clamp(F.softplus(self.switch_time_unconstr), 0, self.t_max)

        n_cells = px_pi.shape[0]

        # component dist
        comp_dist = Categorical(probs=px_pi)

        # induction
        mean_u_ind, mean_s_ind = self._get_induction_unspliced_spliced(
            alpha, alpha_1, lambda_alpha, beta, gamma, t_s * px_rho
        )

        if self.time_dep_transcription_rate:
            mean_u_ind_steady = (alpha_1 / beta).expand(n_cells, self.n_input)
            mean_s_ind_steady = (alpha_1 / gamma).expand(n_cells, self.n_input)
        else:
            mean_u_ind_steady = (alpha / beta).expand(n_cells, self.n_input)
            mean_s_ind_steady = (alpha / gamma).expand(n_cells, self.n_input)
        scale_u = scale[: self.n_input, :].expand(n_cells, self.n_input, 4).sqrt()

        # repression
        u_0, s_0 = self._get_induction_unspliced_spliced(
            alpha, alpha_1, lambda_alpha, beta, gamma, t_s
        )

        tau = px_tau
        mean_u_rep, mean_s_rep = self._get_repression_unspliced_spliced(
            u_0,
            s_0,
            beta,
            gamma,
            (self.t_max - t_s) * tau,
        )
        mean_u_rep_steady = torch.zeros_like(mean_u_ind)
        mean_s_rep_steady = torch.zeros_like(mean_u_ind)
        scale_s = scale[self.n_input :, :].expand(n_cells, self.n_input, 4).sqrt()

        end_penalty = ((u_0 - self.switch_unspliced).pow(2)).sum() + (
            (s_0 - self.switch_spliced).pow(2)
        ).sum()

        # unspliced
        mean_u = torch.stack(
            (
                mean_u_ind,
                mean_u_ind_steady,
                mean_u_rep,
                mean_u_rep_steady,
            ),
            dim=2,
        )
        scale_u = torch.stack(
            (
                scale_u[..., 0],
                scale_u[..., 0],
                scale_u[..., 0],
                0.1 * scale_u[..., 0],
            ),
            dim=2,
        )
        dist_u = Normal(mean_u, scale_u)
        mixture_dist_u = MixtureSameFamily(comp_dist, dist_u)

        # spliced
        mean_s = torch.stack(
            (mean_s_ind, mean_s_ind_steady, mean_s_rep, mean_s_rep_steady),
            dim=2,
        )
        scale_s = torch.stack(
            (
                scale_s[..., 0],
                scale_s[..., 0],
                scale_s[..., 0],
                0.1 * scale_s[..., 0],
            ),
            dim=2,
        )
        dist_s = Normal(mean_s, scale_s)
        mixture_dist_s = MixtureSameFamily(comp_dist, dist_s)

        return mixture_dist_s, mixture_dist_u, end_penalty

    def _get_induction_unspliced_spliced(
        self, alpha, alpha_1, lambda_alpha, beta, gamma, t, eps=1e-6
    ):
        if self.time_dep_transcription_rate:
            unspliced = alpha_1 / beta * (1 - torch.exp(-beta * t)) - (
                alpha_1 - alpha
            ) / (beta - lambda_alpha) * (
                torch.exp(-lambda_alpha * t) - torch.exp(-beta * t)
            )

            spliced = (
                alpha_1 / gamma * (1 - torch.exp(-gamma * t))
                + alpha_1
                / (gamma - beta + eps)
                * (torch.exp(-gamma * t) - torch.exp(-beta * t))
                - beta
                * (alpha_1 - alpha)
                / (beta - lambda_alpha + eps)
                / (gamma - lambda_alpha + eps)
                * (torch.exp(-lambda_alpha * t) - torch.exp(-gamma * t))
                + beta
                * (alpha_1 - alpha)
                / (beta - lambda_alpha + eps)
                / (gamma - beta + eps)
                * (torch.exp(-beta * t) - torch.exp(-gamma * t))
            )
        else:
            unspliced = (alpha / beta) * (1 - torch.exp(-beta * t))
            spliced = (alpha / gamma) * (1 - torch.exp(-gamma * t)) + (
                alpha / ((gamma - beta) + eps)
            ) * (torch.exp(-gamma * t) - torch.exp(-beta * t))

        return unspliced, spliced

    def _get_repression_unspliced_spliced(self, u_0, s_0, beta, gamma, t, eps=1e-6):
        unspliced = torch.exp(-beta * t) * u_0
        spliced = s_0 * torch.exp(-gamma * t) - (
            beta * u_0 / ((gamma - beta) + eps)
        ) * (torch.exp(-gamma * t) - torch.exp(-beta * t))
        return unspliced, spliced

    def sample(
        self,
    ) -> np.ndarray:
        """Not implemented."""
        raise NotImplementedError

    @torch.no_grad()
    def get_loadings(self) -> np.ndarray:
        """Extract per-gene weights (for each Z, shape is genes by dim(Z)) in the linear decoder."""
        # This is BW, where B is diag(b) batch norm, W is weight matrix
        if self.decoder.linear_decoder is False:
            raise ValueError("Model not trained with linear decoder")
        w = self.decoder.rho_first_decoder.fc_layers[0][0].weight
        if self.use_batch_norm_decoder:
            bn = self.decoder.rho_first_decoder.fc_layers[0][1]
            sigma = torch.sqrt(bn.running_var + bn.eps)
            gamma = bn.weight
            b = gamma / sigma
            b_identity = torch.diag(b)
            loadings = torch.matmul(b_identity, w)
        else:
            loadings = w
        loadings = loadings.detach().cpu().numpy()

        return loadings

