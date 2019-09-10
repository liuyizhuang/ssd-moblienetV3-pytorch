import torch
import torch.nn as nn
import torch.nn.functional as F

class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
    def forward(self, x):
        height,weight =10,10
        return F.avg_pool2d(x, kernel_size=[height,weight])  # this is ok

        #return F.avg_pool2d(x, x.size()[2:])
                # RuntimeError: ONNX symbolic expected a constant value in the trace

        #return F.adaptive_avg_pool2d(x, [75,75])  # this is ok

net = mynet()
x = torch.randn(1, 3, 300, 300)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
x = x.to(device)

out = net(x)
print('out.size ', out.size()) #(1, 3, 1, 1)

torch.onnx.export(net, x, "test.onnx", verbose=True)
