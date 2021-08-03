import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
from uu.utils import memory 


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = BasicBlock(3, 64)
        self.maxp = nn.MaxPool2d((2,2), (2,2))
        self.block2 = BasicBlock(64, 128)

def main():
    torch.set_default_dtype(torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)

    H = 256
    W = 256 
    B = 1
    input_1 = torch.rand(B,3,H,W, requires_grad = True).cuda()

    opList = []
    for name, layer in net.named_modules():
        print(name, layer)
        if isinstance(layer, nn.Conv2d):
            opList.append(layer)

        if isinstance(layer, nn.MaxPool2d):
            opList.append(layer)
    # how to detect skip link here??
    

        

   

   
    print("done")

if __name__=="__main__":
    main()







    

