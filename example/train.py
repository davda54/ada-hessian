import argparse
import torch

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from torch.optim import SGD
import sys; sys.path.append("..")
from ada_hessian import AdaHessian


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--average_conv_kernel", dest="average_conv_kernel", action="store_true", default=False)
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--optimizer", default="ada_hessian", type=str, help="Type of optimizer, supported values are {'ada_hessian', SGD'}.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--update_each", default=1, type=int, help="Delayed hessian update.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = Cifar(args.batch_size, args.threads)
    log = Log(log_each=10)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

    if args.optimizer == "ada_hessian":
        optimizer = AdaHessian(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            update_each=args.update_each,
            average_conv_kernel=args.average_conv_kernel
        )
    else:
        optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for i, (inputs, labels) in enumerate(dataset.train):
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = smooth_crossentropy(outputs, labels.to(device))
            loss.mean().backward(create_graph=args.optimizer == "ada_hessian")
            optimizer.step()

            with torch.no_grad():
                correct = (torch.argmax(outputs.data, 1).cpu() == labels).float()
                log(model, loss.cpu(), correct, scheduler.lr())
                scheduler(epoch)

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for inputs, labels in dataset.test:
                outputs = model(inputs.to(device))
                loss = smooth_crossentropy(outputs, labels.to(device))
                correct = torch.argmax(outputs.data, 1).cpu() == labels
                log(model, loss.cpu(), correct)

    log.flush()
