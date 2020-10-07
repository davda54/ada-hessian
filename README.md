# AdaHessian  🚀

Unofficial implementation of the [AdaHessian optimizer](https://arxiv.org/abs/2006.00719). Created as a drop-in replacement for any PyTorch optimizer – you only need to set `create_graph=True` in the `backward()` call and everything else should work 🥳

Our version supports multiple `param_groups`, distributed training, delayed Hessian updates and more precise approximation of the Hessian trace.

## Usage

```python
from ada_hessian import AdaHessian
...
model = YourModel()
optimizer = AdaHessian(model.parameters())
...
for input, output in data:
  optimizer.zero_grad()
  loss = loss_function(output, model(input))
  loss.backward(create_graph=True)  # this is the important line! 🧐
  optimizer.step()
...
```

<br>

## Documentation

#### `AdaHessian.__init__`

| **Argument**    | **Description** |
| :-------------- | :-------------- |
| `params` (iterable) | iterable of parameters to optimize or dicts defining parameter groups |
| `lr` (float, optional) | learning rate *(default: 0.1)* |
| `betas`((float, float), optional) | coefficients used for computing running averages of gradient and the squared hessian trace *(default: (0.9, 0.999))* |
| `eps` (float, optional)           | term added to the denominator to improve numerical stability *(default: 1e-8)* |
| `weight_decay` (float, optional)   | weight decay (L2 penalty) *(default: 0.0)* |
| `hessian_power` (float, optional)  | exponent of the hessian trace *(default: 1.0)* |
| `update_each` (int, optional)   | compute the hessian trace approximation only after *this* number of steps (to save time) *(default: 1)* |
| `n_samples` (int, optional) | how many times to sample `z` for the approximation of the hessian trace *(default: 1)* |
| `average_conv_kernel` (bool, optional) | average out the hessian traces of convolutional kernels as in the original paper *(default: false)* |

<br>

#### `AdaHessian.step`

Performs a single optimization step.

| **Argument**    | **Description** |
| :-------------- | :-------------- |
| `closure` (callable, optional)        | a closure that reevaluates the model and returns the loss *(default: None)* |
