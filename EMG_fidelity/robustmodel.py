import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnPool(nn.Module):
    """Attention pooling over time: (B,T,F) -> (B,F)"""
    def __init__(self, feat_dim: int):
        super().__init__()
        self.w = nn.Linear(feat_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B,T,F)
        a = self.w(h)                  # (B,T,1)
        a = torch.softmax(a, dim=1)    # (B,T,1)
        return (a * h).sum(dim=1)      # (B,F)

class EMGHandNet1D(nn.Module):
    """
    EMGHandNet-style classifier for EMG windows.

    Input:
      x: (B, C=12, L=2000)  (you can use any L, e.g., 2000)
    Output:
      features: (B, D)      embedding for FID-like use
      logits:   (B, K)      class logits for classification/IS-like use
    """
    def __init__(
        self,
        in_ch: int = 12,
        num_classes: int = 53,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout_cnn: float = 0.2,
        dropout_head: float = 0.3,
    ):
        super().__init__()

        # CNN feature extractor: (B,C,L) -> (B,256,T)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            #nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),  # L -> L/2

            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            #nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),  # -> L/4

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            #nn.GroupNorm(8, 256),
                
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),  # -> L/8

            nn.Dropout(p=dropout_cnn),
        )

        # BiLSTM over time: (B,T,256) -> (B,T,2H)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if lstm_layers > 1 else 0.0,
        )

        # Attention pooling: (B,T,2H) -> (B,2H)
        self.pool = AttnPool(feat_dim=2 * lstm_hidden)

        # Head: (B,2H) -> (B,K)
        self.head = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_head),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor):
        z = self.cnn(x)         # (B,256,T)
        z = z.transpose(1, 2)   # (B,T,256)
        h, _ = self.lstm(z)     # (B,T,2H)
        features = self.pool(h) # (B,2H)
        logits = self.head(features)
        
        # features shape: (B, 256) 
        return features, logits



class SimpleConvNet(nn.Module):
    def __init__(self, num_classes, in_channels=12):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Layer 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Layer 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            # Classifier
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Extract features from the second to last layer
        features = self.net[:-1](x)
        # Pass features through the last layer (Linear classifier)
        logits = self.net[-1](features)

        # features shape: (B, 256)
        return features, logits
    


import torch
import torch.nn as nn
from math import ceil

# 引用原有的开源组件
from .cross_models.cross_encoder import Encoder
from .cross_models.cross_embed import DSW_embedding

class CrossformerClassifier(nn.Module):
    def __init__(self, data_dim, in_len, num_classes, seg_len = 16, win_size=2,
                factor=8, d_model=128, d_ff=256, n_heads=4, e_layers=5, 
                dropout=0.0, device=torch.device('cuda:0')):
        super(CrossformerClassifier, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.seg_len = seg_len
        self.device = device

        # 1. 对齐原代码的 Padding 逻辑 (处理不可整除的序列长度)
        self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # 2. 保留 Embedding 部分
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_in_len // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        # 3. 保留 Encoder (多尺度特征提取骨干)
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth=1, 
                               dropout=dropout, in_seg_num=(self.pad_in_len // seg_len), factor=factor)
        
        # 4. 分类头：仅使用最后一层 Encoder 输出
        self.head = nn.Linear(d_model, num_classes)
        
    def forward(self, x_seq):
        batch_size = x_seq.shape[0]
        
        # 处理 Padding (原汁原味的逻辑)
        if (self.in_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim=1)

        # DSW 嵌入与位置编码
        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)
        
        # 通过 Encoder 提取特征
        # enc_out 是一个 list，包含了从细粒度到粗粒度的多层张量
        # 每个张量的 shape 为: [batch_size, ts_d, seg_num_l, d_model]
        enc_out = self.encoder(x_seq)

        # 取最后一层 Encoder 输出，shape: [batch_size, ts_d, seg_num, d_model]
        last_out = enc_out[-1]
        # 全局平均池化，shape: [batch_size, d_model]
        features = last_out.mean(dim=(1, 2))
        logits = self.head(features)

        # features shape: (B, d_model)
        return features, logits


class Crossformer1D(nn.Module):
    """
    Wrapper that adapts channel-first EMG input (B, C, L) to Crossformer's
    expected layout (B, L, C), so it can be used like other EMG_fidelity classifiers.
    """

    def __init__(
        self,
        num_classes: int,
        in_ch: int = 12,
        in_len: int = 400,
        seg_len: int = 16,
        win_size: int = 2,
        factor: int = 8,
        d_model: int = 128,
        d_ff: int = 256,
        n_heads: int = 4,
        e_layers: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model = CrossformerClassifier(
            data_dim=in_ch,
            in_len=in_len,
            num_classes=num_classes,
            seg_len=seg_len,
            win_size=win_size,
            factor=factor,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            e_layers=e_layers,
            dropout=dropout,
            device=torch.device("cpu"),
        )

    def forward(self, x: torch.Tensor):
        x_seq = x.transpose(1, 2).contiguous()
        return self.model(x_seq)
