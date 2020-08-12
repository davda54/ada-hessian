# AdaHessian

Unofficial implementation of the AdaHessian optimizer. Created as a drop-in replacement for any PyTorch optimizer – you only need to set `create_graph=True` in the `backward()` call and everything else should work.

## Usage

```
from ada_hessian import AdaHessian
...
model = YourModel()
optimizer = AdaHessian(model.parameters())
...
for input, output in data:
  loss = loss_function(output, model(input))
  loss.backward(create_graph=True)  # this is the important line!
  optimizer.step()
...
```
