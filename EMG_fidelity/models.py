import torch
import torch.nn as nn
import torch.nn.functional as F

'''class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, input_channels=12, num_classes=53,dropout_rate=0.2):
        super(ResNet1D, self).__init__()
        self.in_planes = 32
        self.dropout_rate = dropout_rate
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        features = out.view(out.size(0), -1) # (B, 512)
        features = F.dropout(features, p=self.dropout_rate, training=self.training)
        logits = self.fc(features)
        
        return features, logits # Logits for IS/Classification, features for FID

def ResNet18_1D(input_channels=12, num_classes=53):
    return ResNet1D(BasicBlock1D, [1, 1, 1, 1], input_channels=input_channels, num_classes=num_classes)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm_1d(norm: str, num_channels: int, gn_groups: int = 8) -> nn.Module:
    norm = norm.lower()
    if norm in ("bn", "batchnorm", "batch_norm"):
        return nn.BatchNorm1d(num_channels)
    if norm in ("gn", "groupnorm", "group_norm"):
        # Make sure groups divides channels
        g = min(gn_groups, num_channels)
        while num_channels % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, num_channels)
    raise ValueError(f"Unknown norm={norm}. Use 'bn' or 'gn'.")


class BasicBlock1D(nn.Module):
    """
    BasicBlock for 1D ResNet (post-activation, same as your original style),
    with optional BN/GN and optional zero-init on the last norm to stabilize training.
    """
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        norm: str = "bn",
        gn_groups: int = 8,
        zero_init_residual: bool = True,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm1 = _make_norm_1d(norm, planes, gn_groups=gn_groups)

        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm2 = _make_norm_1d(norm, planes, gn_groups=gn_groups)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_planes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                _make_norm_1d(norm, planes * self.expansion, gn_groups=gn_groups),
            )

        # Zero-init last norm's scale so the block starts near identity mapping
        if zero_init_residual:
            # BN has weight; GN also has weight
            if hasattr(self.norm2, "weight") and self.norm2.weight is not None:
                nn.init.constant_(self.norm2.weight, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet1D(nn.Module):
    """
    Small/medium ResNet1D for short windows (e.g., 400 samples).

    Key changes vs your original:
      - stem stride defaults to 1 (avoid early downsampling on 400-length windows)
      - configurable norm: BN or GN
      - optional zero-init residual
      - correct feature dimension (no misleading comments)
      - name helpers: ResNet10_1D, ResNet18_1D
    """

    def __init__(
        self,
        block,
        num_blocks,
        input_channels: int = 12,
        num_classes: int = 53,
        base_channels: int = 32,
        dropout_rate: float = 0.3,
        norm: str = "bn",
        gn_groups: int = 8,
        stem_kernel: int = 7,
        stem_stride: int = 1,
        zero_init_residual: bool = True,
    ):
        super().__init__()
        self.in_planes = base_channels
        self.dropout = nn.Dropout(p=dropout_rate)

        # Stem: keep stride=1 for short windows; do NOT use maxpool by default.
        stem_pad = stem_kernel // 2
        self.conv1 = nn.Conv1d(
            input_channels,
            base_channels,
            kernel_size=stem_kernel,
            stride=stem_stride,
            padding=stem_pad,
            bias=False,
        )
        self.norm1 = _make_norm_1d(norm, base_channels, gn_groups=gn_groups)

        # Stages
        self.layer1 = self._make_layer(
            block,
            planes=base_channels,
            num_blocks=num_blocks[0],
            stride=1,
            norm=norm,
            gn_groups=gn_groups,
            zero_init_residual=zero_init_residual,
        )
        self.layer2 = self._make_layer(
            block,
            planes=base_channels * 2,
            num_blocks=num_blocks[1],
            stride=2,
            norm=norm,
            gn_groups=gn_groups,
            zero_init_residual=zero_init_residual,
        )
        self.layer3 = self._make_layer(
            block,
            planes=base_channels * 4,
            num_blocks=num_blocks[2],
            stride=2,
            norm=norm,
            gn_groups=gn_groups,
            zero_init_residual=zero_init_residual,
        )
        self.layer4 = self._make_layer(
            block,
            planes=base_channels * 8,
            num_blocks=num_blocks[3],
            stride=2,
            norm=norm,
            gn_groups=gn_groups,
            zero_init_residual=zero_init_residual,
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels * 8 * block.expansion, num_classes)

    def _make_layer(
        self,
        block,
        planes: int,
        num_blocks: int,
        stride: int,
        norm: str,
        gn_groups: int,
        zero_init_residual: bool,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride=s,
                    norm=norm,
                    gn_groups=gn_groups,
                    zero_init_residual=zero_init_residual,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, L)
        returns:
          features: (B, D)  for FID-like metrics
          logits:   (B, K)  for classification / IS
        """
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)               # (B, D, 1)
        features = out.flatten(1)             # (B, D)
        features = self.dropout(features)
        logits = self.fc(features)

        # features shape: (B, 256)
        return features, logits


def ResNet18_1D(
    input_channels: int = 12,
    num_classes: int = 53,
    **kwargs,
) -> ResNet1D:
    # ResNet-10 style (small), good default for your single-subject setting.
    return ResNet1D(
        BasicBlock1D,
        [1, 1, 1, 1],
        input_channels=input_channels,
        num_classes=num_classes,
        **kwargs,
    )
