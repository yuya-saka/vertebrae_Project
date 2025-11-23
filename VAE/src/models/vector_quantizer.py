"""
Vector Quantizer Layer for VQ-VAE
コードブックを用いた離散潜在表現の学習
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantization Layer

    Args:
        num_embeddings: コードブックのサイズ (離散的な潜在ベクトルの数)
        embedding_dim: 各埋め込みベクトルの次元
        commitment_cost: Commitment Lossの係数 (β)
        decay: Exponential Moving Average (EMA)の減衰率 (使用しない場合は0)
        epsilon: 数値安定性のための小さな値
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.0,
        epsilon: float = 1e-5,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # コードブック (学習可能な埋め込み)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        # EMA使用時の統計量
        if self.decay > 0.0:
            self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
            self.register_buffer('ema_w', torch.zeros(num_embeddings, embedding_dim))

    def forward(self, inputs: torch.Tensor) -> tuple:
        """
        Forward pass

        Args:
            inputs: Encoderからの出力 (B, C, D, H, W)

        Returns:
            quantized: 量子化された出力 (B, C, D, H, W)
            loss: VQ Loss (codebook loss + commitment loss)
            perplexity: コードブック使用率の指標
            encodings: 各位置で選択されたコードブックインデックス (B, D*H*W)
        """
        # 入力のshape: (B, C, D, H, W)
        input_shape = inputs.shape
        batch_size, channels, depth, height, width = input_shape

        # Flatten: (B, C, D, H, W) -> (B*D*H*W, C)
        flat_input = inputs.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, C)
        flat_input = flat_input.view(-1, self.embedding_dim)  # (B*D*H*W, C)

        # 最も近いコードブックベクトルを探す
        # L2距離の計算: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)  # ||z||^2
            + torch.sum(self.embedding.weight ** 2, dim=1)  # ||e||^2
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())  # 2*z*e
        )  # (B*D*H*W, num_embeddings)

        # 最も近いコードブックのインデックス
        encoding_indices = torch.argmin(distances, dim=1)  # (B*D*H*W,)

        # One-hot encoding
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()  # (B*D*H*W, num_embeddings)

        # コードブックから対応するベクトルを取得
        quantized = torch.matmul(encodings, self.embedding.weight)  # (B*D*H*W, C)

        # EMA更新 (訓練時のみ)
        if self.training and self.decay > 0.0:
            self._ema_update(flat_input, encodings)

        # Loss計算
        # 1. Codebook Loss: ||sg[z] - e||^2
        codebook_loss = F.mse_loss(quantized.detach(), flat_input)

        # 2. Commitment Loss: β * ||z - sg[e]||^2
        commitment_loss = self.commitment_cost * F.mse_loss(quantized, flat_input.detach())

        # 総合VQ Loss
        vq_loss = codebook_loss + commitment_loss

        # Straight-Through Estimator: 勾配をバイパス
        quantized = flat_input + (quantized - flat_input).detach()

        # 元のshapeに戻す
        quantized = quantized.view(batch_size, depth, height, width, channels)
        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, D, H, W)

        # Perplexity計算 (コードブックの使用多様性の指標)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, vq_loss, perplexity, encoding_indices

    def _ema_update(self, flat_input: torch.Tensor, encodings: torch.Tensor):
        """
        Exponential Moving Average (EMA) 更新

        Args:
            flat_input: フラット化された入力 (B*D*H*W, C)
            encodings: One-hot encodings (B*D*H*W, num_embeddings)
        """
        with torch.no_grad():
            # クラスタサイズの更新
            cluster_size = torch.sum(encodings, dim=0)  # (num_embeddings,)
            self.ema_cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

            # クラスタ重心の更新
            dw = torch.matmul(encodings.t(), flat_input)  # (num_embeddings, C)
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            # 埋め込みの更新
            n = torch.sum(self.ema_cluster_size)
            cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )
            embed_normalized = self.ema_w / cluster_size.unsqueeze(1)
            self.embedding.weight.data.copy_(embed_normalized)

    def get_codebook_usage(self, encoding_indices: torch.Tensor) -> dict:
        """
        コードブックの使用状況を取得

        Args:
            encoding_indices: エンコーディングインデックス (B*D*H*W,)

        Returns:
            usage_stats: 使用統計の辞書
        """
        unique_codes = torch.unique(encoding_indices)
        usage_ratio = len(unique_codes) / self.num_embeddings

        return {
            'unique_codes': len(unique_codes),
            'total_codes': self.num_embeddings,
            'usage_ratio': usage_ratio,
        }
