from typing import Optional
import math
import enum
import torch as th
import torch.nn.functional as F
import numpy as np
from .diffusion_utils import discretized_gaussian_log_likelihood, normal_kl
from .gaussian_diffusion import ModelMeanType, ModelVarType, LossType, get_named_beta_schedule, mean_flat, _extract_into_tensor
    

class BaseDiffusion:
    """
    Base class for forward diffusion defined by:
        x_t = a(t) * x0 + sigma(t) * eps,   eps ~ N(0, I)

    Subclasses MUST implement:
        - a_t(t): alpha_bar^{1/2} or any a(t) in [0, 1]
        - sigma_t(t): noise scale >= 0

    """
    def __init__(
        self,
        model_mean_type: ModelMeanType,
        model_var_type: ModelVarType,
        loss_type: LossType,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type

        assert model_mean_type in [
            ModelMeanType.PREVIOUS_X,
            ModelMeanType.START_X,
            ModelMeanType.EPSILON,
        ]
        assert model_var_type in [
            ModelVarType.FIXED_SMALL,
            ModelVarType.FIXED_LARGE,
            ModelVarType.LEARNED,
            ModelVarType.LEARNED_RANGE,
        ]
        assert loss_type in [
            LossType.MSE,
            LossType.RESCALED_MSE,
            LossType.KL,
            LossType.RESCALED_KL,
        ]
        # assert self.eta >= 0 and self.eta <= 1, "eta must be in [0, 1]"
    
    @property
    def num_timesteps(self):
        return int(self.a.shape[0])
    
    def a_t(self, t, shape):
        """
        Compute a(t) given t.
        :param t: a Tensor of shape [batch_size] with values in [0, 1]
        :return: a Tensor of shape [batch_size] with values in [0, 1]
        """
        return _extract_into_tensor(self.a, t, shape)
    
    def a_tm1(self, t, shape):
        """
        Compute a(t-1) given t.
        :param t: a Tensor of shape [batch_size] with values in [0, 1]
        :return: a Tensor of shape [batch_size] with values in [0, 1]
        """
        return _extract_into_tensor(self.a_prev, t, shape)

    def sigma_t(self, t, shape):
        """
        Compute sigma(t) given t.
        :param t: a Tensor of shape [batch_size] with values in [0, 1]
        :return: a Tensor of shape [batch_size] with values >= 0
        """
        return _extract_into_tensor(self.sigma, t, shape)
    
    def sigma_tm1(self, t, shape):
        """
        Compute sigma(t-1) given t.
        :param t: a Tensor of shape [batch_size] with values in [0, 1]
        :return: a Tensor of shape [batch_size] with values >= 0
        """
        return _extract_into_tensor(self.sigma_prev, t, shape)

    @property
    def a(self):
        """
        Get a(t) for the full diffusion process.
        :return: a Tensor of shape [num_timesteps] with values in [0, 1]
        """
        raise NotImplementedError()
    
    @property
    def a_prev(self):
        return np.append(1.0, self.a[:-1])
    
    @property
    def sigma(self):
        """
        Get sigma(t) for the full diffusion process.
        :return: a Tensor of shape [num_timesteps] with values >= 0
        """
        raise NotImplementedError()

    @property
    def sigma_prev(self):
        return np.append(0.0, self.sigma[:-1])
    
    @property
    def q_posterior_mean_coef1(self):
        """
        ((a_tm1**2 * sigma_t**2) - (a_t**2 * sigma_tm1**2)) / (a_tm1 * sigma_t**2)
        """
        return (self.a_prev**2 * self.sigma**2 - self.a**2 * self.sigma_prev**2) / (self.a_prev * self.sigma**2)
    
    @property
    def q_posterior_mean_coef2(self):
        """
        (a_t * sigma_tm1**2)/(a_tm1 * sigma_t**2)
        """
        return (self.a * self.sigma_prev**2) / (self.a_prev * self.sigma**2)
    
    @property
    def q_posterior_variance(self):
        """
        sigma_tm1**2 * (1 - (a_t**2 * sigma_tm1**2)/(a_tm1**2 * sigma_t**2)) without eta
        """
        return self.sigma_prev**2 * (1 - (self.a**2 * self.sigma_prev**2) / (self.a_prev**2 * self.sigma**2))
    
    @property
    def q_posterior_log_variance_clipped(self):
        return np.log(np.append(self.q_posterior_variance[1], self.q_posterior_variance[1:]).clip(min=1e-20))
    
    def __str__(self):
        return (
            f"Diffusion(model_mean_type={self.model_mean_type}, "
            f"model_var_type={self.model_var_type}, loss_type={self.loss_type})"
        )
    
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: a 1-D Tensor of timesteps.
        :param noise: if specified, the noise to add. If not specified,
                      random noise will be used.
        :return: a noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        return (
            self.a_t(t, x_start.shape) * x_start
            + self.sigma_t(t, x_start.shape) * noise
        )
    
    def q_mean_variance(self, x_start, t):
        """
        Get the mean and variance of q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: a 1-D Tensor of timesteps.
        :return: a tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = self.a_t(t, x_start.shape) * x_start
        variance = self.sigma_t(t, x_start.shape) ** 2
        log_variance = th.log(variance.clamp(min=1e-20))
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of q(x_{t-1} | x_t, x_0), only for DDPM.
        :param x_start: the initial data batch.
        :param x_t: the diffused data batch.
        :param t: a 1-D Tensor of timesteps.
        :return: a tuple (mean, variance, log_variance), all of x_start's shape.
        """
        coeff1 = _extract_into_tensor(self.q_posterior_mean_coef1, t, x_t.shape)
        coeff2 = _extract_into_tensor(self.q_posterior_mean_coef2, t, x_t.shape)
        mean = coeff1 * x_start + coeff2 * x_t

        variance = _extract_into_tensor(self.q_posterior_variance, t, x_t.shape)
        log_variance = _extract_into_tensor(self.q_posterior_log_variance_clipped, t, x_t.shape)
        return mean, variance, log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """
        Compute x_0 from the predicted noise.
        :param x_t: the noisy data.
        :param t: the diffusion step.
        :param eps: the predicted noise.
        :return: a prediction of x_0.
        """
        return (
            x_t - self.sigma_t(t, x_t.shape) * eps
        ) / self.a_t(t, x_t.shape)
    
    def _predict_eps_from_xstart(self, x_t, t, x0):
        """
        Compute the predicted noise from x_t and x_0.
        :param x_t: the noisy data.
        :param t: the diffusion step.
        :param x0: the predicted x_0.
        :return: a prediction of the noise.
        """
        return (
            x_t - self.a_t(t, x_t.shape) * x0
        ) / self.sigma_t(t, x_t.shape)
    
    def p_mean_variance(self, model, x_t, t, clip_denoised: bool=True, denoised_fn=None, model_kwargs: Optional[dict]=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        x_0.
        :param model: the model, which takes x_t and t and returns either
                      the predicted x_0 or the predicted noise.
        :param x_t: the noisy data.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the x_0 prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the 
                            denoised output.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
                             pass to the model.
        :return: a dict with the following keys:
                 - "mean": the predicted mean of p(x_{t-1} | x_t)
                 - "variance": the predicted variance of p(x_{t-1} | x_t)
                 - "log_variance": the predicted log variance of p(x_{t-1} | x_t)
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, F, C = x_t.shape[:3]
        assert t.shape == (B,)
        model_output = model(x_t, t, **model_kwargs)
        assert model_output.dtype == x_t.dtype

        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, F, C * 2, *x_t.shape[3:])
            model_output, model_var_values = th.split(model_output, C, dim=2)
            min_log = _extract_into_tensor(self.q_posterior_log_variance_clipped, t, x_t.shape)
            max_log = _extract_into_tensor(np.log(self.betas), t, x_t.shape)

            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                ModelVarType.FIXED_SMALL: (
                    self.q_posterior_variance, 
                    self.q_posterior_log_variance_clipped
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x_t.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x_t.shape)
        
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
            epsilon = self._predict_eps_from_xstart(x_t, t, pred_xstart)
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t, t, model_output))
            epsilon = model_output
        else:
            raise NotImplementedError(self.model_mean_type)
        
        model_mean, _, _ = self.q_posterior_mean_variance(pred_xstart, x_t, t)
        
        assert model_mean.shape == model_log_variance.shape == model_variance.shape == x_t.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "epsilon": epsilon,
            "extra": extra,
        }

    @th.no_grad()
    def p_step_eta(
        self,
        model,
        x_t: th.Tensor,
        t: th.Tensor, # LongTensor [B], current t
        *,
        clip_denoised: bool = True,
        denoised_fn = None,
        model_kwargs: Optional[dict] = None,
        eta: float = 0.0,
        use_model_variance: bool = True,
    ) -> th.Tensor:
        """
        ONE reverse step t -> t-1 that unifies DDIM (eta=0) and DDPM (eta=1).

        - Mean uses the DDIM-formula expressed with (a_t, sigma_t).
        - Noise scale sigma is:
            sigma_ddim = eta * sqrt( (1 - barα_{t-1})/(1 - barα_t) ) * sqrt(1 - α_t).
        If `use_model_variance=True`,
        we instead set sigma = eta * exp(0.5 * model_log_variance) to leverage learned σ.
        """
        if model_kwargs is None:
            model_kwargs = {}

        out = self.p_mean_variance(
            model,
            x_t,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        x0_hat = out["pred_xstart"]
        eps = self._predict_eps_from_xstart(x_t, t, x0_hat)

        # DDIM formula
        omega = eta * _extract_into_tensor(self.q_posterior_variance, t, x_t.shape).sqrt()
        if use_model_variance:
            # DDPM cases:
            # if model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            #   log_variance = model_output
            # elif ModelVarType.FIXED_LARGE:
            #   log_variance = np.append(self.posterior_variance[1], self.betas[1:])
            # elif ModelVarType.FIXED_SMALL:
            #   log_variance = default
            omega = eta * th.exp(0.5 * out["log_variance"])
        
        a_t = self.a_t(t, x_t.shape)
        a_tm1 = self.a_tm1(t, x_t.shape)
        sigma_t = self.sigma_t(t, x_t.shape)
        sigma_tm1 = self.sigma_tm1(t, x_t.shape)
        mean = a_tm1 / a_t * x_t + (th.sqrt(th.clamp(sigma_tm1**2 - omega**2, min=0.0)) - a_tm1 / a_t * sigma_t) * eps

        noise = th.randn_like(x_t)
        nonzero = (t != 0).float().view(-1, *([1] * (x_t.ndim - 1)))
        sample = mean + nonzero * omega * noise
        return {'sample': sample, 'pred_xstart': x0_hat, 'epsilon': out['epsilon']}

    def sample_loop_eta(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        use_model_variance=True,
    ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            use_model_variance=use_model_variance
        ):
            final = sample
        return final["sample"]

    def sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        use_model_variance: bool = True,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_step_eta(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                    use_model_variance=use_model_variance
                )
                yield out
                img = out["sample"]
        
    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
       return self.sample_loop_eta(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=1.0,
            use_model_variance=True
       )

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
       return self.sample_loop_eta(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=0.0,
            use_model_variance=False
       )

    def _vb_terms_bpd(
            self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}
    
    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, t, **model_kwargs)
            # try:
            #     model_output = model(x_t, t, **model_kwargs).sample # for tav unet
            # except:
            #     model_output = model(x_t, t, **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, F, C = x_t.shape[:3]
                assert model_output.shape == (B, F, C * 2, *x_t.shape[3:])
                model_output, model_var_values = th.split(model_output, C, dim=2)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=2)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def sample_init_noise(self, shapes, device, dtype=th.float32):
        """
        Sample the initial noise for the diffusion process.
        :param batch_size: the number of samples to produce.
        :param n_frames: the number of frames in each sample.
        :param in_channel: the number of channels in each sample.
        :param latent_size: the height and width of each sample.
        :param device: the device to create the samples on.
        :return: a batch of samples, of shape
                 [batch_size, n_frames, in_channel, latent_size, latent_size].
        """
        return th.randn(*shapes, device=device, dtype=dtype)
    

class GaussianDiffusionV2(BaseDiffusion):
    def __init__(
        self,
        betas,
        model_mean_type: ModelMeanType,
        model_var_type: ModelVarType,
        loss_type: LossType,
    ):
        super().__init__(model_mean_type, model_var_type, loss_type)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bar = np.cumprod(self.alphas, axis=0)
    
    @property
    def a(self):
        return np.sqrt(self.alpha_bar)

    @property
    def sigma(self):
        return np.sqrt(1.0 - self.alpha_bar)
    

if __name__ == "__main__":
    # sanity check
    betas = get_named_beta_schedule("linear", 1000)
    diffusion = GaussianDiffusionV2(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    print(diffusion)

    B, F, C, H, W = 1, 16, 4, 16, 16
    import tqdm
    for i in tqdm.tqdm(range(1)):
        x0 = th.randn(B, F, C, H, W)
        # t = th.randint(0, diffusion.num_timesteps, (B,))
        t = th.zeros(B, dtype=th.long)
        noise = th.randn_like(x0).clamp(-1,1)
        # noise = None
        x_t = diffusion.q_sample(x0, t, noise=noise)
        diff = (x0 - x_t).abs().max()
        print('max_diff', diff)
        # assert th.allclose(x_t, x0)
        
        # pretend the net is perfect EPS predictor:
        eps_true = diffusion._predict_eps_from_xstart(x_t, t, x0)
        # print('eps_true', eps_true[0,:,0,0,0])
        # print('noise', noise[0,:,0,0,0])
        if not th.allclose(eps_true, noise, atol=1e-4):
            # get where it fails
            diff = (eps_true - noise).abs()
            max_diff = diff.max()
            print('max_diff', max_diff)
            raise ValueError("eps_true not close to noise")

        # continue
        out = diffusion.p_mean_variance(lambda x, tt, **_: eps_true, x_t, t)

        if not th.allclose(out["pred_xstart"], x0.clamp(-1,1), atol=1e-4):
            diff = (out["pred_xstart"] - x0.clamp(-1,1)).abs()
            max_diff = diff.max()
            print('max_diff', max_diff)
            raise ValueError("pred_xstart not close to x0")
