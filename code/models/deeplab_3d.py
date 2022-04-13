import sys
# sys.path.append('./sync_batchnorm')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Resnet import build_resnet
from Aspp import build_aspp
from Decoder import build_decoder
from sync_batchnorm.batchnorm import SynchronizedBatchNorm3d


class Deeplab(nn.Module):
    def __init__(self, output_stride = 16, num_classes = 2,
                 sync_bn=True, freeze_bn=False):
        super(Deeplab, self).__init__()
        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm3d
        else:
            BatchNorm = nn.BatchNorm3d
        self.num_classes = num_classes
        self.out_stride = 16
        self.resnet = build_resnet(output_stride, BatchNorm)
        self.aspp = build_aspp('resnet',output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, 'resnet', BatchNorm)
        self.freeze_bn = freeze_bn

    def forward(self, input):
        x,low_level_feat = self.resnet(input)
        x = self.aspp(x)
        x = self.decoder(x,low_level_feat)
        # x = nn.Softmax(dim=1)(x)
        x = F.interpolate(x,size=input.size()[2:],mode='trilinear',align_corners=True)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm3d):
                m.eval()
            elif isinstance(m, nn.BatchNorm3d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv3d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv3d) or isinstance(m[1], SynchronizedBatchNorm3d) \
                            or isinstance(m[1], nn.BatchNorm3d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv3d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv3d) or isinstance(m[1], SynchronizedBatchNorm3d) \
                            or isinstance(m[1], nn.BatchNorm3d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

if __name__ == "__main__":
    model = Deeplab(num_classes=2,output_stride=16,sync_bn=False,freeze_bn=False)
    model.eval()
    input = torch.rand(1, 1, 128, 128, 128)
    output = model(input)
    print(output.size())
    output = output.detach().numpy()
    print(np.min(output[0,1,:,:,:]))



