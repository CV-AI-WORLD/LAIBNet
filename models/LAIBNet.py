# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from thop import profile
'''
Conv block: conv->LeakyRelu
'''
class Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        return  self.conv(x)
'''
Conv_sigmod block: conv->Sigmoid
'''
class Conv_sigmod(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_sigmod, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return  self.conv(x)
'''
Down block:conv->LeakyRelu
'''
class Down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel,  kernel_size=3,stride=2,padding=1),
            nn.LeakyReLU(inplace=True),
            )
    def forward(self, x):
        if x.shape[2]%2 != 0 or x.shape[3]%2 != 0:
            x = nn.functional.pad(x,(0,x.shape[3]%2,0,x.shape[2]%2))
        down_sampled = self.down_conv(x)
        return down_sampled
'''
FCIC block
'''
class FCIC(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FCIC, self).__init__()
        self.up_conv_FFT = nn.Sequential(
            nn.Conv2d(2*in_channel, 2*in_channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm='backward')
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.up_conv_FFT(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm='backward')
        return self.conv(y)
'''
FCICU block
'''
class FCICU(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FCICU, self).__init__()
        self.conv_fft = FCIC(in_channel, out_channel)
        self.up_sampling = nn.ConvTranspose2d(out_channel,out_channel,kernel_size=4,stride=2,padding=1)#nearest
    def forward(self, x):
        convfft = self.conv_fft(x)
        return self.up_sampling(convfft)
'''
LAIBNet
'''
class LAIBNet(nn.Module):
    def __init__(self,channel = 16):
        super(LAIBNet, self).__init__()
        #encode branch 1
        self.down1 = Down(3, channel)
        self.down2 = Down(channel, channel)
        self.down3 = Down(channel, channel)
        #middle
        self.mid = Conv(channel,channel)
        #decode
        self.up3_fft = FCICU(channel, channel)
        self.up2_fft = FCICU(channel, channel)
        self.up1_fft = FCICU(channel, channel)    
        #encode branch 2
        self.d2_fft = FCIC(channel,channel)
        self.d1_fft = FCIC(channel,channel)
        self.d0_fft = FCIC(3,channel)
        self.conv_sigmod = Conv_sigmod(channel,3)
    
    def forward(self, x):
        #encode branch 1
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        #middle
        mid_d3 = self.mid(d3) 
        #decode
        u2fft = self.up3_fft(mid_d3)
        d2fft = self.d2_fft(d2)
        d2_cat_u2_fft = u2fft*d2fft
        #encode branch 2
        u1fft = self.up2_fft(d2_cat_u2_fft)
        d1fft =  self.d1_fft(d1)
        d1_cat_u1_fft = u1fft*d1fft
        #multiply
        u0fft = self.up1_fft(d1_cat_u1_fft)
        xfft = self.d0_fft(x)
        x_cat_u0_fft =xfft*u0fft
        #Conv_sigmod
        r = self.conv_sigmod(x_cat_u0_fft)
        return r

def build_LAIBNet(channel=16):
    return LAIBNet(channel)

if __name__ == '__main__':
    net = build_LAIBNet(16)
    input_tensor = torch.randn(1,3,600,960)
    out_tensor = net(input_tensor)
    flops, params = profile(net, inputs=(input_tensor,))
    print(f"FLOPs: {flops/1e9:.4f} G")
    print(f"{params/1e6:.4f} M")