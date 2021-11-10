from thop import profile
from torchvision.models import resnet50
import torch


model = resnet50()
input = torch.randn(1, 3, 224, 224) # (batch_size, num_channel, Height, Width)
flops, params = profile(model, inputs=(input, ))
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')
