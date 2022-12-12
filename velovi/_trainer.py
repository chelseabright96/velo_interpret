from scvi.train import TrainingPlan
import torch
import pytorch_lightning as pl
from scvi._compat import Literal
from typing import Union
import logging
logger = logging.getLogger(__name__)
class ProxGroupLasso:
    def __init__(self, alpha_GP, omega=None, inplace=True):
    # omega - vector of coefficients with size
    # equal to the number of groups
        if omega is None:
            self._group_coeff = alpha_GP
        else:
            self._group_coeff = (omega*alpha_GP).view(-1)

        # to check for update
        self._alpha = alpha_GP

        self._inplace = inplace

    def __call__(self, W):
        if not self._inplace:
            W = W.clone()

        norm_vect = W.norm(p=2, dim=0)
        norm_g_gr_vect = norm_vect>self._group_coeff

        scaled_norm_vector = norm_vect/self._group_coeff
        scaled_norm_vector+=(~(scaled_norm_vector>0)).float()

        W-=W/scaled_norm_vector
        W*=norm_g_gr_vect.float()
        #print(f"W:{W}")

        return W


class ProxL1:
    def __init__(self, alpha_GP, I=None, inplace=True):
        self._I = ~I.bool() if I is not None else None
        self._alpha=alpha_GP
        self._inplace=inplace

    def __call__(self, W):
        if not self._inplace:
            W = W.clone()

        W_geq_alpha = W>=self._alpha
        W_leq_neg_alpha = W<=-self._alpha
        W_cond_joint = ~W_geq_alpha&~W_leq_neg_alpha

        if self._I is not None:
            W_geq_alpha &= self._I
            W_leq_neg_alpha &= self._I
            W_cond_joint &= self._I

        W -= W_geq_alpha.float()*self._alpha
        W += W_leq_neg_alpha.float()*self._alpha
        W -= W_cond_joint.float()*W
        return W

class CustomTrainingPlan(TrainingPlan):
    def __init__(self, 
            model,
            alpha_GP,
            alpha_kl,
            lr=1e-2,
            weight_decay=1e-6,
            n_steps_kl_warmup: Union[int, None] = None,
            n_epochs_kl_warmup: Union[int, None] = 400,
            reduce_lr_on_plateau: bool = False,
            lr_factor: float = 0.6,
            lr_patience: int = 30,
            lr_threshold: float = 0.0,
            lr_scheduler_metric: Literal[
                "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
            ] = "elbo_validation",
            lr_min: float = 0,
            omega=None,
            alpha_l1=None,
            alpha_l1_epoch_anneal=100,
            alpha_l1_anneal_each=5,
            gamma_ext=None,
            gamma_epoch_anneal=None,
            gamma_anneal_each=5,
            beta=1.,
            print_stats=True,
            **loss_kwargs,):
        super().__init__(module=model,
            lr=lr,
            weight_decay=weight_decay,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_threshold=lr_threshold,
            lr_scheduler_metric=lr_scheduler_metric,
            lr_min=lr_min,
            **loss_kwargs)

        self.model=model
        self.lr = lr
        self.print_stats = print_stats

        self.alpha_GP = alpha_GP
        self.omega = omega
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.omega is not None:
            self.omega = self.omega.to(device)

        # self.gamma_ext = gamma_ext
        # self.gamma_epoch_anneal = gamma_epoch_anneal
        # self.gamma_anneal_each = gamma_anneal_each

        self.alpha_l1 = alpha_l1
        self.alpha_l1_epoch_anneal = alpha_l1_epoch_anneal
        self.alpha_l1_anneal_each = alpha_l1_anneal_each

        # if self.model.use_hsic:
        #     self.beta = beta
        # else:
        #     self.beta = None

        self.watch_lr = None

        self.use_prox_ops = self.check_prox_ops()
        self.prox_ops = {}

        self.corr_coeffs = self.init_anneal()


    def check_prox_ops(self):
        use_prox_ops = {}

        use_main = self.model.decoder.L0.expr_L.weight.requires_grad

        use_prox_ops['main_group_lasso'] = use_main and self.alpha_GP is not None

        use_mask = use_main and self.model.mask is not None
        use_prox_ops['main_soft_mask'] = use_mask and self.alpha_l1 is not None

        # use_ext_m = self.model.n_ext_m_decoder > 0 and self.alpha_l1 is not None
        # use_ext_m = use_ext_m and self.model.decoder.L0.ext_L_m.weight.requires_grad
        # use_prox_ops['ext_soft_mask'] = use_ext_m and self.model.ext_mask is not None

        return use_prox_ops

    def init_anneal(self):
        corr_coeffs = {}

        use_soft_mask = self.use_prox_ops['main_soft_mask'] #or self.use_prox_ops['ext_soft_mask']
        if use_soft_mask and self.alpha_l1_epoch_anneal is not None:
            corr_coeffs['alpha_l1'] = 1. / self.alpha_l1_epoch_anneal
        else:
            corr_coeffs['alpha_l1'] = 1.

        # if self.use_prox_ops['ext_unannot_l1'] and self.gamma_epoch_anneal is not None:
        #     corr_coeffs['gamma_ext'] = 1. / self.gamma_epoch_anneal
        # else:
        #     corr_coeffs['gamma_ext'] = 1.

        return corr_coeffs

    def anneal(self):
        any_change = False

        # if self.corr_coeffs['gamma_ext'] < 1.:
        #     any_change = True
        #     time_to_anneal = self.epoch > 0 and self.epoch % self.gamma_anneal_each == 0
        #     if time_to_anneal:
        #         self.corr_coeffs['gamma_ext'] = min(self.epoch / self.gamma_epoch_anneal, 1.)
        #         if self.print_stats:
        #             print('New gamma_ext anneal coefficient:', self.corr_coeffs['gamma_ext'])

        if self.corr_coeffs['alpha_l1'] < 1.:
            any_change = True
            time_to_anneal = self.epoch > 0 and self.epoch % self.self.alpha_l1_anneal_each == 0
            if time_to_anneal:
                self.corr_coeffs['alpha_l1'] = min(self.epoch / self.alpha_l1_epoch_anneal, 1.)
                if self.print_stats:
                    print('New alpha_l1 anneal coefficient:', self.corr_coeffs['alpha_l1'])

        return any_change

    def init_prox_ops(self):
        if any(self.use_prox_ops.values()) and self.watch_lr is None:
            self.watch_lr = self.lr

        if 'main_group_lasso' not in self.prox_ops and self.use_prox_ops['main_group_lasso']:
            print('Init the group lasso proximal operator for the main terms.')
            alpha_corr = self.alpha_GP * self.watch_lr
            self.prox_ops['main_group_lasso'] = ProxGroupLasso(alpha_corr, self.omega)

        if 'main_soft_mask' not in self.prox_ops and self.use_prox_ops['main_soft_mask']:
            print('Init the soft mask proximal operator for the main terms.')
            main_mask = self.model.mask.to(self.device)
            alpha_l1_corr = self.alpha_l1 * self.watch_lr * self.corr_coeffs['alpha_l1']
            self.prox_ops['main_soft_mask'] = ProxL1(alpha_l1_corr, main_mask)

        # if 'ext_unannot_l1' not in self.prox_ops and self.use_prox_ops['ext_unannot_l1']:
        #     print('Init the L1 proximal operator for the unannotated extension.')
        #     gamma_ext_corr = self.gamma_ext * self.watch_lr * self.corr_coeffs['gamma_ext']
        #     self.prox_ops['ext_unannot_l1'] = ProxL1(gamma_ext_corr)

        # if 'ext_soft_mask' not in self.prox_ops and self.use_prox_ops['ext_soft_mask']:
        #     print('Init the soft mask proximal operator for the annotated extension.')
        #     ext_mask = self.model.ext_mask.to(self.device)
        #     alpha_l1_corr = self.alpha_l1 * self.watch_lr * self.corr_coeffs['alpha_l1']
        #     self.prox_ops['ext_soft_mask'] = ProxL1(alpha_l1_corr, ext_mask)

    def update_prox_ops(self):
        if 'main_group_lasso' in self.prox_ops:
            alpha_corr = self.alpha_GP * self.watch_lr
            if self.prox_ops['main_group_lasso']._alpha != alpha_corr:
                self.prox_ops['main_group_lasso'] = ProxGroupLasso(alpha_corr, self.omega)

        # if 'ext_unannot_l1' in self.prox_ops:
        #     gamma_ext_corr = self.gamma_ext * self.watch_lr * self.corr_coeffs['gamma_ext']
        #     if self.prox_ops['ext_unannot_l1']._alpha != gamma_ext_corr:
        #         self.prox_ops['ext_unannot_l1']._alpha = gamma_ext_corr

        for mask_key in ('main_soft_mask'):#, 'ext_soft_mask'):
            if mask_key in self.prox_ops:
                alpha_l1_corr = self.alpha_l1 * self.watch_lr * self.corr_coeffs['alpha_l1']
                if self.prox_ops[mask_key]._alpha != alpha_l1_corr:
                    self.prox_ops[mask_key]._alpha = alpha_l1_corr

    def apply_prox_ops(self):
        if 'main_soft_mask' in self.prox_ops:
            self.prox_ops['main_soft_mask'](self.model.decoder.L0.expr_L.weight.data)
        if 'main_group_lasso' in self.prox_ops:
            self.prox_ops['main_group_lasso'](self.model.decoder.L0.expr_L.weight.data)
        # if 'ext_unannot_l1' in self.prox_ops:
        #     self.prox_ops['ext_unannot_l1'](self.model.decoder.L0.ext_L.weight.data)
        # if 'ext_soft_mask' in self.prox_ops:
        #     self.prox_ops['ext_soft_mask'](self.model.decoder.L0.ext_L_m.weight.data)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        self.init_prox_ops()
        """Training step for the model."""
        if "alpha_kl" in self.loss_kwargs:
            alpha_kl = self.alpha_kl
            self.loss_kwargs.update({"alpha_kl": alpha_kl})
            self.log("alpha_kl", alpha_kl, on_step=True, on_epoch=False)
        _, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        #self.log("train_loss", scvi_loss.loss, on_epoch=True)
        #self.log("no. deactivated terms", n_deact_terms, on_epoch=True)
        #self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
        #super().training_step(batch, batch_idx, optimizer_idx=0)
        self.apply_prox_ops()
        return scvi_loss.loss
        
    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        # loss kwargs here contains `n_obs` equal to n_training_obs
        # so when relevant, the actual loss value is rescaled to number
        # of training examples
        _, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        n_deact_terms = self.model.decoder.n_inactive_terms()
        #self.log("no. deactivated terms", n_deact_terms, on_epoch=True)
        #self.log("validation_loss", scvi_loss.loss, on_epoch=True)
        self.log_dict({'no. deactivated terms': n_deact_terms, 'validation_loss': scvi_loss.loss}, prog_bar=True)
        self.compute_and_log_metrics(scvi_loss, self.val_metrics, "validation")
        if self.use_prox_ops['main_group_lasso']:
            n_deact_terms = self.model.decoder.n_inactive_terms()
            msg = f'Number of deactivated terms: {n_deact_terms}'
            msg = '\n' + msg
            logger.info(msg)
            # print(msg)
            # print('-------------------')
        if self.use_prox_ops['main_soft_mask']:
            main_mask = self.prox_ops['main_soft_mask']._I
            share_deact_genes = (self.model.decoder.L0.expr_L.weight.data.abs()==0) & main_mask
            share_deact_genes = share_deact_genes.float().sum().cpu().numpy() / self.model.n_inact_genes
            # print('Share of deactivated inactive genes: %.4f' % share_deact_genes)
            # print('-------------------')
            logger.info('Share of deactivated inactive genes: %.4f' % share_deact_genes)
        any_change = self.anneal()
        logger.info(f"any_change: {any_change}")
        if any_change:
            self.update_prox_ops()
            logger.info(f"updating prox_ops")

    # def validation_step(self, batch, batch_idx):
    #     #super().validation_step(batch, batch_idx)
    #     #if self.print_stats:
    #     if self.use_prox_ops['main_group_lasso']:
    #         n_deact_terms = self.model.decoder.n_inactive_terms()
    #         msg = f'Number of deactivated terms: {n_deact_terms}'
    #         msg = '\n' + msg
    #         print(msg)
    #         print('-------------------')
    #     if self.use_prox_ops['main_soft_mask']:
    #         main_mask = self.prox_ops['main_soft_mask']._I
    #         share_deact_genes = (self.model.decoder.L0.expr_L.weight.data.abs()==0) & main_mask
    #         share_deact_genes = share_deact_genes.float().sum().cpu().numpy() / self.model.n_inact_genes
    #         print('Share of deactivated inactive genes: %.4f' % share_deact_genes)
    #         print('-------------------')
    #     any_change = self.anneal()
    #     print(any_change)
    #     if any_change:
    #         self.update_prox_ops()
    
    # def validation_epoch_end(self, batch, outs):
    #     # outs is a list of whatever you returned in `validation_step`
    #     loss = torch.stack(outs).mean()
    #     #self.log("validation_loss", scvi_loss.loss, on_epoch=True)
    #     #self.compute_and_log_metrics(scvi_loss, self.val_metrics, "validation")
    #     self.log("val_loss", loss)
    #     if self.print_stats:
    #         if self.use_prox_ops['main_group_lasso']:
    #             n_deact_terms = self.model.decoder.n_inactive_terms()
    #             msg = f'Number of deactivated terms: {n_deact_terms}'
    #             msg = '\n' + msg
    #             print(msg)
    #             print('-------------------')
    #         if self.use_prox_ops['main_soft_mask']:
    #             main_mask = self.prox_ops['main_soft_mask']._I
    #             share_deact_genes = (self.model.decoder.L0.expr_L.weight.data.abs()==0) & main_mask
    #             share_deact_genes = share_deact_genes.float().sum().cpu().numpy() / self.model.n_inact_genes
    #             print('Share of deactivated inactive genes: %.4f' % share_deact_genes)
    #             print('-------------------')
    #     any_change = self.anneal()
    #     print(any_change)
    #     if any_change:
    #         self.update_prox_ops()
        # print(f"loss: {scvi_loss.loss}")
        # print(f"sum recon loss: {scvi_loss.reconstruction_loss_sum}")
        # print(f"n_obs_minibatch: {scvi_loss.n_obs_minibatch}")
        # print(f"kl local sum: {scvi_loss.kl_local_sum}")
        # print(f"kl global sum: {scvi_loss.kl_global_sum}")
        # return {'val_loss': scvi_loss.loss, 'log': log}

