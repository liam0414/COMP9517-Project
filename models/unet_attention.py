import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


# Use Atrous (Dilated) Convolutions
# Add Attention Mechanisms
# Implement Feature Pyramid Network (FPN) structure
# Use Deep Supervision
# Employ Residual Connections
# Added a combined_loss function that handles both the main output and the deep
# supervision outputs during training. During validation, it uses only the main output.
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ImprovedUNet(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedUNet, self).__init__()
        
        def conv_block(in_channels, out_channels, dilation=1):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3,
                    padding=dilation, dilation=dilation
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=3,
                    padding=dilation, dilation=dilation
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def upconv_block(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = conv_block(512, 1024, dilation=2)
        
        self.upconv4 = upconv_block(1024, 512)
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.dec4 = conv_block(1024, 512)
        
        self.upconv3 = upconv_block(512, 256)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.dec3 = conv_block(512, 256)
        
        self.upconv2 = upconv_block(256, 128)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.dec2 = conv_block(256, 128)
        
        self.upconv1 = upconv_block(128, 64)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec1 = conv_block(128, 64)
        
        self.conv_last = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Deep supervision
        self.deep_sup4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.deep_sup3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.deep_sup2 = nn.Conv2d(128, num_classes, kernel_size=1)
        
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        bottleneck = self.bottleneck(self.pool(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = self.att4(g=dec4, x=enc4)
        dec4 = self.dec4(torch.cat([dec4, enc4], dim=1))
        
        dec3 = self.upconv3(dec4)
        dec3 = self.att3(g=dec3, x=enc3)
        dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))
        
        dec2 = self.upconv2(dec3)
        dec2 = self.att2(g=dec2, x=enc2)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
        
        dec1 = self.upconv1(dec2)
        dec1 = self.att1(g=dec1, x=enc1)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))
        
        out = self.conv_last(dec1)
        
        # Deep supervision outputs
        deep_sup4 = F.interpolate(self.deep_sup4(dec4), scale_factor=8, mode='bilinear', align_corners=True)
        deep_sup3 = F.interpolate(self.deep_sup3(dec3), scale_factor=4, mode='bilinear', align_corners=True)
        deep_sup2 = F.interpolate(self.deep_sup2(dec2), scale_factor=2, mode='bilinear', align_corners=True)
        
        if self.training:
            return out, deep_sup4, deep_sup3, deep_sup2
        else:
            return out