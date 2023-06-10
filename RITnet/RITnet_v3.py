# Modify from: https://github.com/RSKothari/EllSeg

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def getSizes(chz, growth, blks=4):
    # This function does not calculate the size requirements for head and tail

    # Encoder sizes
    sizes = {'enc': {'inter':[], 'ip':[], 'op': []},
             'dec': {'skip':[], 'ip': [], 'op': []}}
    sizes['enc']['inter'] = np.array([chz*(i+1) for i in range(0, blks)])
    sizes['enc']['op'] = np.array([np.int32(growth*chz*(i+1)) for i in range(0, blks)])
    sizes['enc']['ip'] = np.array([chz] + [np.int32(growth*chz*(i+1)) for i in range(0, blks-1)])

    # Decoder sizes
    sizes['dec']['skip'] = sizes['enc']['ip'][::-1] + sizes['enc']['inter'][::-1]
    sizes['dec']['ip'] = sizes['enc']['op'][::-1] #+ sizes['dec']['skip']
    sizes['dec']['op'] = np.append(sizes['enc']['op'][::-1][1:], chz)
    return sizes

class convBlock(nn.Module):
    def __init__(self, in_c, inter_c, out_c, actfunc):
        super(convBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, inter_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(inter_c, out_c, kernel_size=3, padding=1)
        self.actfunc = actfunc
        self.bn = torch.nn.BatchNorm2d(num_features=out_c)
    def forward(self, x):
        x = self.actfunc(self.conv1(x))
        x = self.actfunc(self.conv2(x)) # Remove x if not working properly
        x = self.bn(x)
        return x

class Transition_down(nn.Module):
    def __init__(self, in_c, out_c, down_size, norm, actfunc):
        super(Transition_down, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.max_pool = nn.AvgPool2d(kernel_size=down_size) if down_size else False
        self.norm = norm(num_features=in_c)
        self.actfunc = actfunc
    def forward(self, x):
        x = self.actfunc(self.norm(x))
        x = self.conv(x)
        x = self.max_pool(x) if self.max_pool else x
        return x

class DenseNet2D_down_block(nn.Module):
    def __init__(self, in_c, inter_c, op_c, down_size, norm, actfunc):
        super(DenseNet2D_down_block, self).__init__()
        self.conv1 = nn.Conv2d(in_c, inter_c, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(in_c+inter_c, inter_c, kernel_size=1, padding=0)
        self.conv22 = nn.Conv2d(inter_c, inter_c, kernel_size=3, padding=1)
        self.conv31 = nn.Conv2d(in_c+2*inter_c, inter_c, kernel_size=1, padding=0)
        self.conv32 = nn.Conv2d(inter_c, inter_c, kernel_size=3, padding=1)
        self.actfunc = actfunc
        self.bn = norm(num_features=in_c)
        self.TD = Transition_down(inter_c+in_c, op_c, down_size, norm, actfunc)

    def forward(self, x):
        x1 = self.actfunc(self.conv1(self.bn(x)))
        x21 = torch.cat([x, x1], dim=1)
        x22 = self.actfunc(self.conv22(self.conv21(x21)))
        x31 = torch.cat([x21, x22], dim=1)
        out = self.actfunc(self.conv32(self.conv31(x31)))
        out = torch.cat([out, x], dim=1)
        return out, self.TD(out)

class DenseNet2D_up_block(nn.Module):
    def __init__(self, skip_c, in_c, out_c, up_stride, actfunc):
        super(DenseNet2D_up_block, self).__init__()
        self.conv11 = nn.Conv2d(skip_c+in_c, out_c, kernel_size=1, padding=0)
        self.conv12 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(skip_c+in_c+out_c, out_c, kernel_size=1,padding=0)
        self.conv22 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.actfunc = actfunc
        self.up_stride = up_stride

    def forward(self, prev_feature_map, x):
        x = F.interpolate(x,
                          mode='bilinear',
                          align_corners=False,
                          scale_factor=self.up_stride)
        x = torch.cat([x, prev_feature_map], dim=1)
        x1 = self.actfunc(self.conv12(self.conv11(x)))
        x21 = torch.cat([x, x1],dim=1)
        out = self.actfunc(self.conv22(self.conv21(x21)))
        return out

class DenseNet_encoder(nn.Module):
    def __init__(self, in_c=1, chz=32, actfunc=F.leaky_relu, growth=1.5, norm=nn.BatchNorm2d):
        super(DenseNet_encoder, self).__init__()
        sizes = getSizes(chz, growth)
        interSize = sizes['enc']['inter']
        opSize = sizes['enc']['op']
        ipSize = sizes['enc']['ip']

        self.head = convBlock(in_c=1,
                                inter_c=chz,
                                out_c=chz,
                                actfunc=actfunc)
        self.down_block1 = DenseNet2D_down_block(in_c=ipSize[0],
                                                 inter_c=interSize[0],
                                                 op_c=opSize[0],
                                                 down_size=2,
                                                 norm=norm,
                                                 actfunc=actfunc)
        self.down_block2 = DenseNet2D_down_block(in_c=ipSize[1],
                                                 inter_c=interSize[1],
                                                 op_c=opSize[1],
                                                 down_size=2,
                                                 norm=norm,
                                                 actfunc=actfunc)
        self.down_block3 = DenseNet2D_down_block(in_c=ipSize[2],
                                                 inter_c=interSize[2],
                                                 op_c=opSize[2],
                                                 down_size=2,
                                                 norm=norm,
                                                 actfunc=actfunc)
        self.down_block4 = DenseNet2D_down_block(in_c=ipSize[3],
                                                 inter_c=interSize[3],
                                                 op_c=opSize[3],
                                                 down_size=2,
                                                 norm=norm,
                                                 actfunc=actfunc)
        self.bottleneck = DenseNet2D_down_block(in_c=opSize[3],
                                                 inter_c=interSize[3],
                                                 op_c=opSize[3],
                                                 down_size=0,
                                                 norm=norm,
                                                 actfunc=actfunc)
    def forward(self, x):
        x = self.head(x) # chz
        skip_1, x = self.down_block1(x) # chz
        skip_2, x = self.down_block2(x) # 2 chz
        skip_3, x = self.down_block3(x) # 4 chz
        skip_4, x = self.down_block4(x) # 8 chz
        _, x = self.bottleneck(x)
        return skip_4, skip_3, skip_2, skip_1, x

class DenseNet_decoder(nn.Module):
    def __init__(self, chz, out_c, growth, actfunc=F.leaky_relu, norm=nn.BatchNorm2d):
        super(DenseNet_decoder, self).__init__()
        sizes = getSizes(chz, growth)
        skipSize = sizes['dec']['skip']
        opSize = sizes['dec']['op']
        ipSize = sizes['dec']['ip']

        self.up_block4 = DenseNet2D_up_block(skipSize[0], ipSize[0], opSize[0], 2, actfunc)
        self.up_block3 = DenseNet2D_up_block(skipSize[1], ipSize[1], opSize[1], 2, actfunc)
        self.up_block2 = DenseNet2D_up_block(skipSize[2], ipSize[2], opSize[2], 2, actfunc)
        self.up_block1 = DenseNet2D_up_block(skipSize[3], ipSize[3], opSize[3], 2, actfunc)

        self.final = convBlock(chz, chz, out_c, actfunc)

    def forward(self, skip4, skip3, skip2, skip1, x):
         x = self.up_block4(skip4, x)
         x = self.up_block3(skip3, x)
         x = self.up_block2(skip2, x)
         x = self.up_block1(skip1, x)
         o = self.final(x)
         return o

class DenseNet2D(nn.Module):
    def __init__(self,
                 chz=32,
                 growth=1.2,
                 actfunc=F.leaky_relu,
                 norm=nn.InstanceNorm2d,
                 selfCorr=False,
                 disentangle=False):
        super(DenseNet2D, self).__init__()

        self.sizes = getSizes(chz, growth)

        self.enc = DenseNet_encoder(in_c=1, chz=chz, actfunc=actfunc, growth=growth, norm=norm)
        self.dec = DenseNet_decoder(chz=chz, out_c=3, actfunc=actfunc, growth=growth, norm=norm)

        self.clf = nn.Sequential(
            torch.nn.Conv2d(153, 256, 3),
            torch.nn.BatchNorm2d(256),
            torch.nn.GELU(),
            torch.nn.Conv2d(256, 256, 3),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 2),
        )

        self._initialize_weights()

    def forward(self, x):
        x4, x3, x2, x1, x = self.enc(x)
        # return x
        out = self.dec(x4, x3, x2, x1, x)
        out[:, 0] += out[:, 1]
        out[:, 1] = -torch.inf
        # resize back to original size (2x)
        out = F.interpolate(out, scale_factor=2, mode='nearest')
        eye_cond = self.clf(x)
        return out, eye_cond

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    model = DenseNet2D()
    ckpt = torch.load('./all.git_ok')
    msg = model.load_state_dict(ckpt['state_dict'], strict=False)
    assert set(msg.unexpected_keys) == {'elReg.c1.weight', 'elReg.c1.bias', 'elReg.c2.weight', 'elReg.c2.bias', 'elReg.c3.weight', 'elReg.l1.weight', 'elReg.l1.bias', 'elReg.l2.weight', 'elReg.l2.bias'}

    model.cuda()
    img = torch.randn(1, 1, 256, 256, device='cuda')
    out = model(img)
    print(out.shape)
    import ipdb; ipdb.set_trace()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(count_parameters(model))