import sys

import fedml
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

import fedml
from fedml import FedMLRunner
from fedml.arguments import Arguments
from fedml.model.cv.resnet import resnet20
from fedml.model.cv.resnet_cifar import resnet18_cifar, resnet34_cifar, resnet50_cifar
from fedml.model.cv.resnet_gn import resnet18
from fedml.model.cv.resnet_torch import resnet18 as resnet18_torch
from fedml.model.cv.cnn import Cifar10FLNet
# from FedML_copy.python.fedml import FedMLRunner

sys.path.append("../../../../")
print(sys.path)
from python.fedml import FedMLRunner

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    #create resnet34 model from torch
    
    
    
    if args.model == "resnet18":
        logging.info("ResNet18_GN")
        model = resnet18(group_norm=args.group_norm_channels, num_classes=output_dim)
    elif args.model == "resnet20":
        model = resnet20(class_num=output_dim)
    elif args.model == "resnet18_torch":
        model = resnet18_torch(num_classes=output_dim, in_channels=in_channels)
    elif args.model == "resnet18_cifar":
        logging.info("ResNet18_GN")
        model = resnet18_cifar(group_norm=args.group_norm_channels, num_classes=output_dim)
    elif args.model == "resnet34_cifar":
        model = resnet34_cifar(num_classes=output_dim)
    elif args.model == "resnet50_cifar":
        model = resnet50_cifar(num_classes=output_dim)
    else:
        model = fedml.model.create(args, output_dim)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
