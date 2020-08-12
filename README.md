# AdaHessian  🚀

Unofficial implementation of the AdaHessian optimizer. Created as a drop-in replacement for any PyTorch optimizer – you only need to set `create_graph=True` in the `backward()` call and everything else should work 🥳

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
  loss.backward(create_graph=True)  # this is the important line!
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
  optimizer.set_hess()  # this line accumulates the hessian trace for each parameter
  if (i + 1) % accumulation_steps == 0:
    optimizer.step()
...
```
