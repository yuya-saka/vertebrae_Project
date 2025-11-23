"""
3D VQ-VAE Model for Vertebral Volume Data
正常椎体の3D構造を学習する
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vector_quantizer import VectorQuantizer


class Encoder3D(nn.Module):
    """
    3D Encoder

    入力: (B, 1, 128, 128, 128)
    出力: (B, latent_dim, D', H', W')

    Args:
        in_channels: 入力チャネル数 (通常1)
        hidden_dims: 各層の隠れ層チャネル数リスト
        latent_dim: 潜在表現の次元
        dropout: Dropoutの確率
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dims: list = [32, 64, 128, 256],
        latent_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        modules = []
        prev_channels = in_channels

        # ダウンサンプリング層を構築
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(prev_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
                )
            )
            prev_channels = h_dim

        # 最終層: latent_dimに変換
        modules.append(
            nn.Sequential(
                nn.Conv3d(prev_channels, latent_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(latent_dim),
            )
        )

        self.encoder = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (B, 1, 128, 128, 128)

        Returns:
            latent: (B, latent_dim, D', H', W')
        """
        return self.encoder(x)


class Decoder3D(nn.Module):
    """
    3D Decoder (Encoderの逆構造)

    入力: (B, latent_dim, D', H', W')
    出力: (B, 1, 128, 128, 128)

    Args:
        latent_dim: 潜在表現の次元
        hidden_dims: 各層の隠れ層チャネル数リスト (逆順)
        out_channels: 出力チャネル数 (通常1)
        dropout: Dropoutの確率
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dims: list = [256, 128, 64, 32],
        out_channels: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        modules = []
        prev_channels = latent_dim

        # アップサンプリング層を構築
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(prev_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
                )
            )
            prev_channels = h_dim

        # 最終層: out_channelsに変換
        modules.append(
            nn.Sequential(
                nn.Conv3d(prev_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid(),  # [0, 1]の範囲に正規化
            )
        )

        self.decoder = nn.Sequential(*modules)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            z: (B, latent_dim, D', H', W')

        Returns:
            recon: (B, 1, 128, 128, 128)
        """
        return self.decoder(z)


class VQVAE3D(nn.Module):
    """
    3D Vector Quantized Variational Autoencoder

    Args:
        in_channels: 入力チャネル数
        hidden_dims: Encoder/Decoderの隠れ層次元
        latent_dim: 潜在表現の次元 (= embedding_dim)
        num_embeddings: コードブックサイズ
        commitment_cost: Commitment Lossの係数
        dropout: Dropout確率
        use_ema: EMAを使用するか
        ema_decay: EMAの減衰率
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dims: list = [32, 64, 128, 256],
        latent_dim: int = 256,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
        dropout: float = 0.1,
        use_ema: bool = False,
        ema_decay: float = 0.99,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        # Encoder
        self.encoder = Encoder3D(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        # Vector Quantizer
        self.vq_layer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
            decay=ema_decay if use_ema else 0.0,
        )

        # Decoder
        decoder_hidden_dims = hidden_dims[::-1]  # 逆順
        self.decoder = Decoder3D(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            out_channels=in_channels,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass

        Args:
            x: 入力ボリューム (B, 1, D, H, W)

        Returns:
            output: {
                'recon': 再構成ボリューム (B, 1, D, H, W),
                'vq_loss': VQ Loss,
                'perplexity': コードブック使用率,
                'encoding_indices': エンコーディングインデックス,
            }
        """
        # Encode
        z_e = self.encoder(x)  # (B, latent_dim, D', H', W')

        # Vector Quantization
        z_q, vq_loss, perplexity, encoding_indices = self.vq_layer(z_e)

        # Decode
        recon = self.decoder(z_q)  # (B, 1, D, H, W)

        return {
            'recon': recon,
            'vq_loss': vq_loss,
            'perplexity': perplexity,
            'encoding_indices': encoding_indices,
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力を潜在表現にエンコード

        Args:
            x: (B, 1, D, H, W)

        Returns:
            z_q: 量子化された潜在表現 (B, latent_dim, D', H', W')
        """
        z_e = self.encoder(x)
        z_q, _, _, _ = self.vq_layer(z_e)
        return z_q

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        潜在表現をデコード

        Args:
            z_q: (B, latent_dim, D', H', W')

        Returns:
            recon: (B, 1, D, H, W)
        """
        return self.decoder(z_q)

    def get_reconstruction_error(self, x: torch.Tensor, reduction: str = 'none') -> torch.Tensor:
        """
        再構成誤差を計算 (骨折検出用)

        Args:
            x: 入力ボリューム (B, 1, D, H, W)
            reduction: 'none', 'mean', 'sum'

        Returns:
            error: 再構成誤差マップ
                - 'none': (B, 1, D, H, W) ボクセル単位の誤差
                - 'mean': スカラー
                - 'sum': スカラー
        """
        with torch.no_grad():
            output = self.forward(x)
            recon = output['recon']

            # L1距離で再構成誤差を計算
            error = torch.abs(x - recon)

            if reduction == 'mean':
                return error.mean()
            elif reduction == 'sum':
                return error.sum()
            else:  # 'none'
                return error


def build_vqvae_3d(config: dict) -> VQVAE3D:
    """
    設定から3D VQ-VAEモデルを構築

    Args:
        config: モデル設定の辞書

    Returns:
        model: VQVAE3D instance
    """
    return VQVAE3D(
        in_channels=config.get('in_channels', 1),
        hidden_dims=config.get('hidden_dims', [32, 64, 128, 256]),
        latent_dim=config.get('latent_dim', 256),
        num_embeddings=config.get('num_embeddings', 512),
        commitment_cost=config.get('commitment_cost', 0.25),
        dropout=config.get('dropout', 0.1),
        use_ema=config.get('use_ema', False),
        ema_decay=config.get('ema_decay', 0.99),
    )
