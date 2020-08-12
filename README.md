# AdaHessian  🚀

Unofficial implementation of the [AdaHessian optimizer](https://arxiv.org/abs/2006.00719). Created as a drop-in replacement for any PyTorch optimizer – you only need to set `create_graph=True` in the `backward()` call and everything else should work 🥳

## Usage

#### Simple example

```python
from ada_hessian import AdaHessian
...
model = YourModel()
optimizer = AdaHessian(model.parameters())
...
for input, output in data:
  loss = loss_function(output, model(input))
  loss.backward(create_graph=True)  # this is the important line! 🧐
  optimizer.step()
...
```

#### Advanced usage

Our code also allows you to have more control over computing the hessian traces. As an example, you can use that for gradient accumulation when you need bigger effective batch size:
```python
from ada_hessian import AdaHessian
...
model = YourModel()
optimizer = AdaHessian(model.parameters(), auto_hess=False)
...
for i, (input, output) in enumerate(data):
  loss = loss_function(output, model(input)) / accumulation_steps
  loss.backward(create_graph=True)
  optimizer.set_hessian()  # accumulate the hessian trace for each parameter
  if (i + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_hessian()  # zero out the hessian trace for each parameter
...
```

## Documentation

#### `AdaHessian.__init__`

| **Argument**    | **Description** |
| :-------------- | :-------------- |
| `params` (iterable) | iterable of parameters to optimize or dicts defining parameter groups |
| `lr` (float, optional) | learning rate *(default: 0.1)* |
| `betas`((float, float), optional) | coefficients used for computing running averages of gradient and the squared hessian trace *(default: (0.9, 0.999))* |
| `eps` (float, optional)           | term added to the denominator to improve numerical stability *(default: 1e-4)* |
| `weight_decay` (float, optional)   | weight decay (L2 penalty) *(default: 0.0)* |
| `hessian_power` (float, optional)  | exponent of the hessian trace *(default: 1.0)* |
| `auto_hessian` (bool, optional)  | automatically call `set_hessian()` and `zero_hessian()` within each step *(default: True)* |
| `update_each` (int, optional)   | compute the hessian trace approximation only after *this* number of steps (to save time) *(default: 1)* |
| `distributed` (bool, optional)   | use a distributed version which shares the hessian traces across multiple GPUs *(default: False)* |

#### `AdaHessian.step`

Performs a single optimization step.

| **Argument**    | **Description** |
| :-------------- | :-------------- |
| `closure` (callable, optional)        | a closure that reevaluates the model and returns the loss *(default: None)* |

#### `AdaHessian.set_hessian`

Computes the Hutchinson approximation of the hessian trace and accumulates it for each trainable parameter. It is called automatically when `auto_hessian == True`.


#### `AdaHessian.zero_hessian`

Zeros out the accumalated hessian traces. It is called automatically when `auto_hessian == True`.
