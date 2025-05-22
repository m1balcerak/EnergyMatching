# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################
# Existing Classes (unchanged)
################################################################################

class MixedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha, padding=0):
        super(MixedConv2d, self).__init__()
        self.alpha = alpha
        # Standard convolution layer
        self.standard_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        # Non-negative convolution layer for ICNN component
        self.non_negative_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
    
    def forward(self, x):
        standard_out = self.standard_conv(x)
        # Ensure non-negative weights for ICNN component using GELU
        non_neg_weights = F.gelu(self.non_negative_conv.weight)
        non_negative_out = F.conv2d(
            x, non_neg_weights, self.non_negative_conv.bias,
            stride=self.non_negative_conv.stride,
            padding=self.non_negative_conv.padding,
            dilation=self.non_negative_conv.dilation,
            groups=self.non_negative_conv.groups
        )
        # Combine both outputs scaled by alpha
        return standard_out + self.alpha * non_negative_out

class MixedLinear(nn.Module):
    def __init__(self, in_features, out_features, alpha):
        super(MixedLinear, self).__init__()
        self.alpha = alpha
        # Standard linear layer
        self.standard_linear = nn.Linear(in_features, out_features)
        # Non-negative linear layer for ICNN component
        self.non_negative_linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        standard_out = self.standard_linear(x)
        # Ensure non-negative weights for ICNN component using GELU
        non_neg_weights = F.gelu(self.non_negative_linear.weight)
        non_negative_out = F.linear(
            x, non_neg_weights, self.non_negative_linear.bias
        )
        return standard_out + self.alpha * non_negative_out

class HybridCNNICNN_DeeperThinner(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=32, alpha=0.5):
        super(HybridCNNICNN_DeeperThinner, self).__init__()
        self.alpha = alpha
        self.gelu = nn.GELU()
        
        # Convolutional Layers
        self.conv1 = MixedConv2d(in_channels, hidden_dim, kernel_size=3, padding=1, alpha=self.alpha)
        self.conv2 = MixedConv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, alpha=self.alpha)
        self.conv3 = MixedConv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, alpha=self.alpha)
        self.conv4 = MixedConv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, alpha=self.alpha)
        
        # Pooling Layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layers
        self.fc1 = MixedLinear(hidden_dim, hidden_dim // 2, alpha=self.alpha)
        self.fc2 = MixedLinear(hidden_dim // 2, hidden_dim // 4, alpha=self.alpha)
        self.fc3 = MixedLinear(hidden_dim // 4, hidden_dim // 8, alpha=self.alpha)
        self.fc4 = MixedLinear(hidden_dim // 8, 1, alpha=self.alpha)
    
    def forward(self, x):
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        x = self.gelu(self.conv3(x))
        x = self.gelu(self.conv4(x))
        
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        x = self.gelu(self.fc3(x))
        V = self.fc4(x)
        return V

class PureCNN(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=32):
        super(PureCNN, self).__init__()
        self.gelu = nn.GELU()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Pooling Layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, hidden_dim // 8)
        self.fc4 = nn.Linear(hidden_dim // 8, 1)
    
    def forward(self, x):
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        x = self.gelu(self.conv3(x))
        x = self.gelu(self.conv4(x))
        
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        x = self.gelu(self.fc3(x))
        V = self.fc4(x)
        return V

class DeeperPureCNN(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=24):
        super(DeeperPureCNN, self).__init__()
        self.gelu = nn.GELU()
        
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1) for _ in range(4)
        ])
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, hidden_dim // 8)
        self.fc4 = nn.Linear(hidden_dim // 8, 1)
    
    def forward(self, x):
        x = self.gelu(self.conv1(x))
        for conv in self.conv_layers:
            x = self.gelu(conv(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        x = self.gelu(self.fc3(x))
        V = self.fc4(x)
        return V

################################################################################
# New ResNet-32 Class (using GELU instead of ReLU)
################################################################################

class BasicBlock(nn.Module):
    """
    A standard basic residual block used in ResNet for CIFAR. 
    """
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    
    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))  # Use GELU here
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.gelu(out)  # GELU again
        return out

class ModifiedResNet32(nn.Module):
    """
    A ResNet-32 adapted for CIFAR input (3 × 32 × 32).
    At the end, we add 2 fully connected layers so it outputs a single scalar potential.
    """
    def __init__(self, in_channels=3):
        super(ModifiedResNet32, self).__init__()
        self.in_planes = 16

        # Initial conv + batchnorm
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # For ResNet-32, we have (5, 5, 5) blocks in layers (typical CIFAR variant).
        self.layer1 = self._make_layer(BasicBlock, 16, 5, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 5, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 5, stride=2)
        
        # 2 FC layers to get down to scalar output
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 1)  # single scalar potential
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))  # GELU instead of ReLU
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # Global average pooling
        out = F.avg_pool2d(out, out.size()[3])  
        out = out.view(out.size(0), -1)         # flatten to (B, 64)
        
        # 2 FC layers
        out = F.gelu(self.fc1(out))             # GELU
        out = self.fc2(out)                     # shape: (B, 1)
        return out
    
################################################################################
# ModifiedResNet50 adapted to CIFAR (32×32) with final scalar output
################################################################################

class Bottleneck(nn.Module):
    """
    Bottleneck block used in ResNet-50, ResNet-101, and ResNet-152.
    For ResNet-50, the blocks are arranged in layers of sizes [3,4,6,3].
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        # 1×1 conv
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 3×3 conv
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # 1×1 conv
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.downsample = downsample

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))    # GELU instead of ReLU
        out = F.gelu(self.bn2(self.conv2(out)))  # GELU
        out = self.bn3(self.conv3(out))          # no activation yet
        
        # if we need to match shape (downsampling)
        if self.downsample is not None:
            x = self.downsample(x)
        
        out += x
        out = F.gelu(out)  # final GELU after residual addition
        return out

class ModifiedResNet50(nn.Module):
    """
    A ResNet-50 adapted for 3×32×32 input (CIFAR).
    We replace ReLU with GELU, and add 2 fully-connected layers that reduce 
    the final 2048-dim features down to a single scalar potential.
    """
    def __init__(self, in_channels=3, num_blocks=[3,4,6,3]):
        """
        Args:
            in_channels (int): Number of input channels (3 for RGB).
            num_blocks (list): Number of Bottleneck blocks in each layer.
                               By default [3, 4, 6, 3] for ResNet-50.
        """
        super(ModifiedResNet50, self).__init__()
        self.in_planes = 64  # typical ResNet start
        
        # Initial 3×3 conv for CIFAR (instead of 7×7 for ImageNet)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Create the 4 layers of Bottleneck blocks
        self.layer1 = self._make_layer(64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        
        # The final feature dimension is 512 * Bottleneck.expansion = 512 * 4 = 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 2 FC layers to map 2048 → 1
        self.fc1 = nn.Linear(2048, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def _make_layer(self, planes, num_blocks, stride):
        """
        Creates a layer with `num_blocks` Bottleneck blocks.
        The first block may apply downsampling if stride != 1 
        or the in/out planes don't match.
        """
        downsample = None
        # If we need to change resolution (stride != 1) or #channels
        if stride != 1 or self.in_planes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion)
            )
        
        layers = []
        # First block in this layer, possibly with downsampling
        layers.append(Bottleneck(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * Bottleneck.expansion
        
        # Remaining blocks in this layer
        for _ in range(num_blocks - 1):
            layers.append(Bottleneck(self.in_planes, planes))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv + BN + GELU
        out = F.gelu(self.bn1(self.conv1(x)))
        
        # 4 layers of Bottleneck blocks
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Global average pooling to get (B, 2048)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)  # shape = (B, 2048)
        
        # 2 FC layers -> scalar
        out = F.gelu(self.fc1(out))
        out = self.fc2(out)         # shape = (B, 1)
        return out
    
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # If number of channels differ, a 1×1 conv ensures shapes match
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))  # or F.relu or ...
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.gelu(out)
        return out

class EBM_ResNet(nn.Module):
    """
    A simple EBM that uses a stack of residual blocks and ends in a scalar potential.
    """
    def __init__(self, in_channels=1, hidden_dim=64, num_blocks=4):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        
        # Suppose we stack several ResBlocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResBlock(hidden_dim, hidden_dim))
        self.res_blocks = nn.Sequential(*blocks)
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final linear (or 1×1 conv) -> scalar
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = F.gelu(self.initial_conv(x))
        out = self.res_blocks(out)
        out = self.pool(out)         # shape (B, hidden_dim, 1, 1)
        out = out.view(out.size(0), -1)  # flatten to (B, hidden_dim)
        energy = self.fc(out)            # shape (B,1)
        return energy
    
################################################################################
# Example Time-Dependent CNN
#  (Similar structure to DeeperPureCNN, but now takes x + 1 time channel.)
################################################################################

class DeeperTimeDependentCNN(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=24):
        """
        in_channels is the data dimension (e.g. 1 for MNIST),
        but the final input to the conv layers will be (in_channels + 1)
        because we add time as an extra channel.
        """
        super(DeeperTimeDependentCNN, self).__init__()
        self.gelu = nn.GELU()

        # Now the first conv sees (in_channels+1) channels.
        self.conv1 = nn.Conv2d(in_channels + 1, hidden_dim, kernel_size=3, padding=1)
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1) for _ in range(4)
        ])
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, hidden_dim // 8)
        self.fc4 = nn.Linear(hidden_dim // 8, 1)
    
    def forward(self, x, t):
        """
        x: (B, C, H, W)
        t: (B, 1, H, W) - a broadcasted tensor of time
        We'll concatenate along channel dimension -> shape: (B, C+1, H, W)
        """
        # Concatenate time as extra channel
        xt = torch.cat([x, t], dim=1)

        out = self.gelu(self.conv1(xt))
        for conv in self.conv_layers:
            out = self.gelu(conv(out))
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.gelu(self.fc1(out))
        out = self.gelu(self.fc2(out))
        out = self.gelu(self.fc3(out))
        V = self.fc4(out)  # shape (B,1)
        return V
    


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, 
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        
        # If channels differ, 1×1 conv. Otherwise, identity.
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.gelu(out)
        return out

class EBM_ResNet_1200K(nn.Module):
    """
    ~1.2 Million parameters with in_channels=1, hidden_dim=129, num_blocks=4.
    """
    def __init__(self, in_channels=1, hidden_dim=128, num_blocks=4):
        super().__init__()
        # The initial conv uses bias=True by default (PyTorch default).
        self.initial_conv = nn.Conv2d(
            in_channels, hidden_dim, kernel_size=3, padding=1, bias=True
        )
        
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResBlock(hidden_dim, hidden_dim))
        self.res_blocks = nn.Sequential(*blocks)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_dim, 1)  # final scalar

        # Custom initialization so output isn't near 0 at start
        self._initialize_weights()

    def forward(self, x):
        out = F.gelu(self.initial_conv(x))
        out = self.res_blocks(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)  # shape (B, hidden_dim)
        energy = self.fc(out)           # shape (B,1)
        return energy

    def _initialize_weights(self):
        """
        A custom weight init routine:
         - Kaiming (He) init for conv/linear
         - BN weights -> 1.0, BN bias -> 0.0
         - final fc.bias -> something > 0 (e.g. 1.0) so the output won't be ~0 initially
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    # Give each conv bias a small positive value
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                # This is the crucial part to ensure non-zero initial outputs
                nn.init.constant_(m.bias, 1.0)
                
                
################################################################################
# Large CNN Energy Model for CelebA 64x64, no time component, single scalar output
################################################################################

class DoubleConv(nn.Module):
    """
    Two consecutive conv(3×3) + GELU. No normalization layers.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=True)

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        return x


class Downsample(nn.Module):
    """
    A simple downsample via a 2×2 stride-2 conv (kernel_size=4, stride=2, pad=1).
    This halves spatial resolution.
    """
    def __init__(self, channels):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4,
                              stride=2, padding=1, bias=True)

    def forward(self, x):
        return self.conv(x)


class LargeCNNEnergyModel(nn.Module):
    """
    - Input: (B, 3, 64, 64)  [CelebA images]
    - Output: (B, 1)         [scalar energy]
    - No BN/GN. Just convs + GELU + skip connections + big FC.
    - Aims for ~100M params when channels and MLP sizes are large.
    """

    def __init__(self):
        super(LargeCNNEnergyModel, self).__init__()

        #############
        # Encoder
        #############
        # Stage 1: 3 -> 128, resolution 64x64
        self.doubleconv1 = DoubleConv(in_channels=3, out_channels=128)
        self.down1       = Downsample(128)  # 64->32

        # Stage 2: 128 -> 256, resolution 32x32
        self.doubleconv2 = DoubleConv(in_channels=128, out_channels=256)
        self.down2       = Downsample(256)  # 32->16

        # Stage 3: 256 -> 512, resolution 16x16
        self.doubleconv3 = DoubleConv(in_channels=256, out_channels=512)
        self.down3       = Downsample(512)  # 16->8

        # Stage 4: 512 -> 768 (or 1024), resolution 8x8
        # Increase if you want even bigger capacity (e.g. 1024).
        self.doubleconv4 = DoubleConv(in_channels=512, out_channels=768)
        # Optionally downsample to 4x4 if you want even more depth/capacity:
        # self.down4       = Downsample(768)

        ###########
        # Decoder (fully connected) 
        ###########
        # We will:
        # 1) Globally pool each skip feature => dimension = channels at that stage
        # 2) Concat them all => sum of (128 + 256 + 512 + 768) = 1664
        # 3) Pass that through a large MLP to get a single scalar.

        # First linear: 1664 -> 8192
        # Second linear: 8192 -> 8192
        # Third linear: 8192 -> 1
        # This alone can push parameter count near or over 100M.

        self.fc0 = nn.Linear(1664, 8192, bias=True)
        self.fc1 = nn.Linear(8192, 8192, bias=True)
        self.fc2 = nn.Linear(8192, 1,    bias=True)

    def forward(self, x):
        # -----------------
        # Encoder forward
        # -----------------
        # Stage 1
        s1 = self.doubleconv1(x)  # (B,128,64,64)
        # Save skip1 = s1, then downsample
        x = self.down1(s1)        # (B,128,32,32)

        # Stage 2
        s2 = self.doubleconv2(x)  # (B,256,32,32)
        x = self.down2(s2)        # (B,256,16,16)

        # Stage 3
        s3 = self.doubleconv3(x)  # (B,512,16,16)
        x = self.down3(s3)        # (B,512, 8, 8)

        # Stage 4
        s4 = self.doubleconv4(x)  # (B,768, 8, 8)
        # x = self.down4(s4)  # optional: if you want 4x4
        # For now, let's keep final resolution at 8x8 = s4

        # -----------------
        # Global average pool the skip features
        # -----------------
        # shape s1 = (B,128,64,64), s2 = (B,256,32,32), ...
        # We'll do adaptive_avg_pool2d -> shape (B, C, 1, 1), then flatten => (B, C)
        skip1 = F.adaptive_avg_pool2d(s1, (1, 1)).view(s1.size(0), -1)  # (B,128)
        skip2 = F.adaptive_avg_pool2d(s2, (1, 1)).view(s2.size(0), -1)  # (B,256)
        skip3 = F.adaptive_avg_pool2d(s3, (1, 1)).view(s3.size(0), -1)  # (B,512)
        skip4 = F.adaptive_avg_pool2d(s4, (1, 1)).view(s4.size(0), -1)  # (B,768)

        # Concat all skip-pooled vectors
        concat_skips = torch.cat([skip1, skip2, skip3, skip4], dim=1)  # (B, 1664)

        # -----------------
        # MLP for scalar output
        # -----------------
        out = F.gelu(self.fc0(concat_skips))  # (B, 8192)
        out = F.gelu(self.fc1(out))           # (B, 8192)
        energy = self.fc2(out)               # (B, 1)

        # The energy can be large positive or negative, no normalization applied.
        return energy
    
    ###############################################################################
# Helper Modules: DoubleConv & Downsample
###############################################################################

class DoubleConv(nn.Module):
    """
    Two consecutive conv(3×3) + GELU. No normalization layers.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=True
        )

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        return x


class Downsample(nn.Module):
    """
    Downsample by a factor of 2 using a stride-2 conv (kernel_size=4, stride=2, pad=1).
    This halves the spatial resolution.
    """
    def __init__(self, channels):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=4, stride=2, padding=1, bias=True
        )

    def forward(self, x):
        return self.conv(x)


###############################################################################
# 1) ~100M Parameter Model for CelebA 64×64 (3 input channels)
###############################################################################

class Network_100M_CelebA64x64(nn.Module):
    """
    - Input:  (B, 3, 64, 64)
    - Output: (B, 1) (scalar energy)
    - 4 stages of downsampling; skip connections are pooled & concatenated.
    - 2 hidden MLP layers at the end, sized to push total params ~100M.
    - No normalization, just conv + GELU + big MLP.
    """
    def __init__(self):
        super(Network_100M_CelebA64x64, self).__init__()
        ################
        # Encoder
        ################
        # Stage 1: (3 -> 128), 64x64
        self.doubleconv1 = DoubleConv(in_channels=3, out_channels=128)
        self.down1       = Downsample(128)  # 64x64 -> 32x32

        # Stage 2: (128 -> 256), 32x32
        self.doubleconv2 = DoubleConv(128, 256)
        self.down2       = Downsample(256)  # 32x32 -> 16x16

        # Stage 3: (256 -> 512), 16x16
        self.doubleconv3 = DoubleConv(256, 512)
        self.down3       = Downsample(512)  # 16x16 -> 8x8

        # Stage 4: (512 -> 768), 8x8
        self.doubleconv4 = DoubleConv(512, 768)
        # If you want extra depth, you could downsample again. We'll stop at 8x8.

        # The skip features will be: 128, 256, 512, 768 => total = 1664
        # We'll feed their global-pooled concatenation into an MLP.

        ################
        # MLP Decoder
        ################
        # 2 hidden layers, sized to approach ~100M total.
        # 1664 -> 6144 -> 6144 -> 1
        self.fc0 = nn.Linear(1664, 6144, bias=True)
        self.fc1 = nn.Linear(6144, 6144, bias=True)
        self.out = nn.Linear(6144, 1, bias=True)

    def forward(self, x):
        # --- Encoder ---
        s1 = self.doubleconv1(x)  # (B,128,64,64)
        x  = self.down1(s1)       # (B,128,32,32)

        s2 = self.doubleconv2(x)  # (B,256,32,32)
        x  = self.down2(s2)       # (B,256,16,16)

        s3 = self.doubleconv3(x)  # (B,512,16,16)
        x  = self.down3(s3)       # (B,512,8,8)

        s4 = self.doubleconv4(x)  # (B,768,8,8)

        # --- Global pool each skip feature ---
        skip1 = F.adaptive_avg_pool2d(s1, (1,1)).view(s1.size(0), -1)  # (B,128)
        skip2 = F.adaptive_avg_pool2d(s2, (1,1)).view(s2.size(0), -1)  # (B,256)
        skip3 = F.adaptive_avg_pool2d(s3, (1,1)).view(s3.size(0), -1)  # (B,512)
        skip4 = F.adaptive_avg_pool2d(s4, (1,1)).view(s4.size(0), -1)  # (B,768)

        concat_skips = torch.cat([skip1, skip2, skip3, skip4], dim=1)  # (B,1664)

        # --- MLP ---
        out = F.gelu(self.fc0(concat_skips))
        out = F.gelu(self.fc1(out))
        energy = self.out(out)  # (B,1)
        return energy


###############################################################################
# 2) ~20M Parameter Model for CIFAR 32×32 (3 input channels)
###############################################################################

class Network_20M_Cifar32x32(nn.Module):
    """
    - Input:  (B, 3, 32, 32)
    - Output: (B, 1)
    - 3 stages: (3->64->128->256)
    - Skips: 64 + 128 + 256 = 448
    - 2 hidden MLP layers, sized for ~20M total.
    """
    def __init__(self):
        super(Network_20M_Cifar32x32, self).__init__()
        ################
        # Encoder
        ################
        # Stage 1: (3 -> 64), 32x32
        self.doubleconv1 = DoubleConv(3, 64)
        self.down1       = Downsample(64)  # 32->16

        # Stage 2: (64 -> 128), 16x16
        self.doubleconv2 = DoubleConv(64, 128)
        self.down2       = Downsample(128) # 16->8

        # Stage 3: (128 -> 256), 8x8
        self.doubleconv3 = DoubleConv(128, 256)
        # Optionally downsample again if you want 4×4. We'll stop at 8×8.

        # skip dimension = 64 + 128 + 256 = 448

        ################
        # MLP Decoder
        ################
        # 2 hidden layers: 448 -> 2048 -> 2048 -> 1
        self.fc0 = nn.Linear(448, 2048, bias=True)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.out = nn.Linear(2048, 1, bias=True)

    def forward(self, x):
        s1 = self.doubleconv1(x)   # (B,64,32,32)
        x  = self.down1(s1)        # (B,64,16,16)

        s2 = self.doubleconv2(x)   # (B,128,16,16)
        x  = self.down2(s2)        # (B,128,8,8)

        s3 = self.doubleconv3(x)   # (B,256,8,8)

        # Global pool skip features
        skip1 = F.adaptive_avg_pool2d(s1, (1,1)).view(s1.size(0), -1)  # (B,64)
        skip2 = F.adaptive_avg_pool2d(s2, (1,1)).view(s2.size(0), -1)  # (B,128)
        skip3 = F.adaptive_avg_pool2d(s3, (1,1)).view(s3.size(0), -1)  # (B,256)

        concat_skips = torch.cat([skip1, skip2, skip3], dim=1)         # (B,448)

        out = F.gelu(self.fc0(concat_skips))
        out = F.gelu(self.fc1(out))
        energy = self.out(out)  # (B,1)
        return energy


###############################################################################
# 3) ~2M Parameter Model for MNIST 28×28 (1 input channel)
###############################################################################

class Network_2M_MNIST28x28(nn.Module):
    """
    - Input:  (B, 1, 28, 28)
    - Output: (B, 1)
    - 2 stages: (1->64) ->down-> (64->128)
    - skip dimension = 64 + 128 = 192
    - 2 hidden MLP layers sized to reach ~2M total params.
    """
    def __init__(self):
        super(Network_2M_MNIST28x28, self).__init__()
        ################
        # Encoder
        ################
        # Stage 1: (1 -> 64), 28x28
        self.doubleconv1 = DoubleConv(1, 64)
        self.down1       = Downsample(64)   # 28->14

        # Stage 2: (64 -> 128), 14x14
        self.doubleconv2 = DoubleConv(64, 128)
        self.down2       = Downsample(128)  # 14->7

        # skip dimension = (64 + 128) = 192

        ################
        # MLP Decoder
        ################
        # 2 hidden layers: 192 -> 1024 -> 1024 -> 1
        # This should push total ~2M (CNN ~0.6M, MLP ~1.4M).
        self.fc0 = nn.Linear(192, 1024, bias=True)
        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.out = nn.Linear(1024, 1, bias=True)

    def forward(self, x):
        s1 = self.doubleconv1(x)   # (B,64,28,28)
        x  = self.down1(s1)        # (B,64,14,14)

        s2 = self.doubleconv2(x)   # (B,128,14,14)
        x  = self.down2(s2)        # (B,128,7,7)

        # Global average pool skip features
        skip1 = F.adaptive_avg_pool2d(s1, (1,1)).view(s1.size(0), -1)  # (B,64)
        skip2 = F.adaptive_avg_pool2d(s2, (1,1)).view(s2.size(0), -1)  # (B,128)

        concat_skips = torch.cat([skip1, skip2], dim=1)                # (B,192)

        out = F.gelu(self.fc0(concat_skips))
        out = F.gelu(self.fc1(out))
        energy = self.out(out)  # (B,1)
        return energy
    
    
    

#------------------------------------------------------------------------------
# A “basic” residual block with 256 channels, but using GELU instead of ReLU.
# (Typically you might see ReLU in a ResNet, but we will respect your “keep GELU.”)
#------------------------------------------------------------------------------

class ResBlock256(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv3x3(256, 256)
        self.bn1   = nn.BatchNorm2d(256)
        self.conv2 = conv3x3(256, 256)
        self.bn2   = nn.BatchNorm2d(256)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Residual add & GELU
        out = out + residual
        out = F.gelu(out)
        return out

#------------------------------------------------------------------------------
# One trunk that processes a single (3,H,W) input and outputs a 256‐dim vector.
# It follows the “Conv → (ResBlock×2 → AvgPool)×3 → ResBlock×2 → GlobalPool” 
# pattern shown in the figure.  (Hence 8 total ResBlocks, each with 256 channels.)
#------------------------------------------------------------------------------

class Trunk256(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial 3×3 conv: 3->256
        self.initial_conv = nn.Conv2d(3, 256, 3, padding=1, bias=False)
        self.initial_bn   = nn.BatchNorm2d(256)

        # We will group the residual blocks in pairs, each followed by pooling.
        # That means 4 “pairs” of ResBlocks = 8 blocks total.
        self.blockgroup1 = nn.Sequential(
            ResBlock256(), ResBlock256()
        )
        self.blockgroup2 = nn.Sequential(
            ResBlock256(), ResBlock256()
        )
        self.blockgroup3 = nn.Sequential(
            ResBlock256(), ResBlock256()
        )
        self.blockgroup4 = nn.Sequential(
            ResBlock256(), ResBlock256()
        )

        self.avgpool = nn.AvgPool2d(2)

    def forward(self, x):
        # Initial conv + BN + GELU
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = F.gelu(x)

        # (ResBlock×2) → AvgPool
        x = self.blockgroup1(x)
        x = self.avgpool(x)  # down to half spatial size

        # (ResBlock×2) → AvgPool
        x = self.blockgroup2(x)
        x = self.avgpool(x)

        # (ResBlock×2) → AvgPool
        x = self.blockgroup3(x)
        x = self.avgpool(x)

        # (ResBlock×2) → Global average pool
        x = self.blockgroup4(x)
        # now do global average pool down to 1×1
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = x.view(x.size(0), -1)  # shape = (B,256)

        return x

#------------------------------------------------------------------------------
# The full ~30M‐param model, matching the “Large” spec:
#   • Three Trunk256 (for 3 input scales, e.g. (3,32,32), (3,16,16), (3,8,8))
#   • Concat 3 outputs → 768 dims
#   • MLP: 768→2048→128
#   • Final output: 1 dimension (e.g. a scalar “energy” or logit).
#
# If you only want a single‐scale CIFAR input, you could remove trunk2/trunk3,
# but then you would not match the original figure’s 30M design. 
#------------------------------------------------------------------------------

class Network_30M_Cifar(nn.Module):
    def __init__(self):
        super().__init__()
        # Three parallel trunks
        self.trunk1 = Trunk256()  # handles (3,32,32)
        self.trunk2 = Trunk256()  # handles (3,16,16)
        self.trunk3 = Trunk256()  # handles (3, 8, 8)

        # MLP(768→2048→128)
        self.fc1 = nn.Linear(768, 2048)
        self.fc2 = nn.Linear(2048, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, x32, x16, x8):
        # Each input scale goes through its own trunk
        h1 = self.trunk1(x32)   # (B,256)
        h2 = self.trunk2(x16)   # (B,256)
        h3 = self.trunk3(x8)    # (B,256)

        # Concatenate → (B,768)
        concat = torch.cat([h1, h2, h3], dim=1)

        # MLP with GELU
        out = F.gelu(self.fc1(concat))
        out = F.gelu(self.fc2(out))
        energy = self.out(out)  # (B,1)
        return energy


###############################################################################
# 1) 3×3 convolution helper, no normalization, with bias
###############################################################################
def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(
        in_ch, out_ch,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=True
    )

###############################################################################
# 2) Residual block with 512 channels, no batchnorm, using GELU
###############################################################################
class ResBlock512(nn.Module):
    """
    A basic residual block:
      out = x + Conv2( GELU( Conv1(x) ) )
      then final GELU
    """
    def __init__(self):
        super().__init__()
        self.conv1 = conv3x3(512, 512)
        self.conv2 = conv3x3(512, 512)

    def forward(self, x):
        out = F.gelu(self.conv1(x))
        out = self.conv2(out)
        out = F.gelu(out + x)
        return out

###############################################################################
# 3) A trunk that processes (3,H,W) => (B,512)
#    "Conv(3->512) -> (ResBlock x2 + AvgPool) x3 -> (ResBlock x2) -> GlobalPool"
#    That's 8 ResBlocks total, each with 512 channels, no normalization.
###############################################################################
class Trunk512(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial conv: 3->512
        self.initial_conv = nn.Conv2d(3, 512, kernel_size=3, padding=1, bias=True)

        # 8 total ResBlocks (grouped in pairs), each pair followed by AvgPool (x3)
        self.blockgroup1 = nn.Sequential(ResBlock512(), ResBlock512())
        self.blockgroup2 = nn.Sequential(ResBlock512(), ResBlock512())
        self.blockgroup3 = nn.Sequential(ResBlock512(), ResBlock512())
        self.blockgroup4 = nn.Sequential(ResBlock512(), ResBlock512())

        self.avgpool = nn.AvgPool2d(2)

    def forward(self, x):
        # Initial conv + GELU
        x = F.gelu(self.initial_conv(x))

        # (ResBlock×2) → AvgPool
        x = self.blockgroup1(x)
        x = self.avgpool(x)

        # (ResBlock×2) → AvgPool
        x = self.blockgroup2(x)
        x = self.avgpool(x)

        # (ResBlock×2) → AvgPool
        x = self.blockgroup3(x)
        x = self.avgpool(x)

        # (ResBlock×2) → Global average pool
        x = self.blockgroup4(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = x.view(x.size(0), -1)  # (B, 512)
        return x

###############################################################################
# 4) Final multi‐scale network ~120M params for CelebA (64×64, 32×32, 16×16)
#    "3 parallel Trunk512" => (512 + 512 + 512) => MLP(1536->4096->256->1)
###############################################################################
class Network_120M_CelebA64x64(nn.Module):
    """
    Multi‐scale ResNet with ~120M parameters.

    Expecting 3 different scale inputs:
      x64 ~ (B,3,64,64)
      x32 ~ (B,3,32,32)
      x16 ~ (B,3,16,16)

    Each goes through a Trunk512 => 512‐dim vector.
    We concatenate => 1536 dims total => MLP(4096->256->1).
    """
    def __init__(self):
        super().__init__()
        self.trunk64 = Trunk512()   # for (3,64,64)
        self.trunk32 = Trunk512()   # for (3,32,32)
        self.trunk16 = Trunk512()   # for (3,16,16)

        # MLP: 1536->4096->256->1
        self.fc1 = nn.Linear(1536, 4096, bias=True)
        self.fc2 = nn.Linear(4096, 256, bias=True)
        self.out = nn.Linear(256, 1, bias=True)

    def forward(self, x64, x32, x16):
        # Each trunk => (B,512)
        h1 = self.trunk64(x64)
        h2 = self.trunk32(x32)
        h3 = self.trunk16(x16)

        # Concatenate => (B,1536)
        concat = torch.cat([h1, h2, h3], dim=1)

        # MLP with GELU
        out = F.gelu(self.fc1(concat))
        out = F.gelu(self.fc2(out))
        energy = self.out(out)  # (B,1)
        return energy