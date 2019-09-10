# -*-ccoding: utf-8 -*-

#from torch.autograd import Variable
import torch

#from vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite

#net =create_mobilenetv3_ssd_lite

device = torch.device("cuda")
print('+++++++++++++++++++++++++++++++')
model = torch.load('./models/mb3-ssd-Epoch-30-Loss-inf.pth')
print('_____________________________________')
model.to(device)
batch_size=1  # 随便一个数
x = torch.randn(batch_size,3,300,300,device='cuda')
print('===================================')
torch.onnx.export(model,x,"sqz.onnx",verbose=False)
