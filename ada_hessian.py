import torch
import torch.distributed as dist


class AdaHessian(torch.optim.optimizer.Optimizer):
    def __init__(self, params, lr=0.1, betas=(0.9, 0.999), eps=1e-4, weight_decay=0.0, hessian_power=1.0, auto_hessian=True, update_each=1, distributed=False):
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

        self.update_each = update_each
        self.auto_hessian = auto_hessian
        self.distributed = distributed

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, hessian_power=hessian_power)
        super(AdaHessian, self).__init__(params, defaults)

        for p in self.get_params():
            p.hess = 0.0
            self.state[p]["hessian step"] = 0

    def get_params(self):
        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)
    
    def zero_hessian(self):
        for p in self.get_params():
            if not isinstance(p.hess, float) and self.state[p]["hessian step"] % self.update_each == 0:
                p.hess.zero_()

    @torch.no_grad()
    def set_hessian(self):
        params = []
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            if self.state[p]["hessian step"] % self.update_each == 0:  # compute the trace only each `update_each` step
                params.append(p)
            self.state[p]["hessian step"] += 1

        if len(params) == 0:
            return

        grads = [p.grad for p in params]

        zs = [torch.randint_like(p, high=2) * 2 - 1 for p in params]  # Rademacher distribution {-1.0, 1.0}
        if self.distributed:
            dist.all_reduce_multigpu(zs, op=dist.ReduceOp.PRODUCT)  # make sure the result is still from Rademacher distribution

        h_zs = torch.autograd.grad(grads, params, grad_outputs=z, only_inputs=True, retain_graph=False)
        for h_z, z, p in zip(h_zs, zs, params):
            p.hess += h_z * z  # enable accumulating hessians

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if self.auto_hessian:
            self.zero_hessian()
            self.set_hessian()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.hess is None:
                    continue

                # Perform correct stepweight decay as in AdamW
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

        return loss
