import torch
from torch.optim.optimizer import Optimizer


class AdaHessian(Optimizer):
    def __init__(self, params, lr=0.15, betas=(0.9, 0.999), eps=1e-4, weight_decay=0, hessian_power=1, auto_hess=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError("Invalid Hessian power value: {}".format(hessian_power))

        self.auto_hess = auto_hess

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, hessian_power=hessian_power)
        super(AdaHessian, self).__init__(params, defaults)

        for p in self.get_params():
            p.hess = 0.0

    def get_params(self):
        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)

    @torch.no_grad()
    def set_hess(self):
        params = [p for p in self.get_params() if p.grad is not None]
        grads = [p.grad for p in params]

        z = [torch.randint_like(p, high=2) * 2 - 1 for p in params]  # Rademacher distribution {-1.0, 1.0}

        h_zs = torch.autograd.grad(grads, params, grad_outputs=z, only_inputs=True, retain_graph=False)
        for h_z, z_i, p in zip(h_zs, z, params):
            p.hess += h_z * z_i

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if self.auto_hess:
            self.set_hess()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.hess is None:
                    continue

                # Perform stepweight decay
                p.mul_(1 - group['lr'] * group['weight_decay'])

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # Exponential moving average of gradient values
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)  # Exponential moving average of Hessian diagonal square values

                exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, p.grad)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(1 - beta2, p.hess, p.hess)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                k = group['hessian_power']
                denom = (exp_hessian_diag_sq / bias_correction2).pow_(k / 2).add_(group['eps'])

                # make update
                step_size = group['lr'] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

                p.hess = 0

        return loss
