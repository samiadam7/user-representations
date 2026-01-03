from typing import Any, Sequence, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        bn_dim: int,
        hidden_dims: int | Sequence[int],
        activation: nn.Module = nn.ReLU(),
        output_activation: None | nn.Module = None,
        dropout_rate: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        else:
            hidden_dims = list(hidden_dims)

        # ---------- ENCODER ----------
        encoder_layers: list[nn.Module] = []
        prev_dim = in_dim

        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(type(activation)())  # new instance
            if use_layer_norm:
                encoder_layers.append(nn.LayerNorm(h))
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = h

        self.encoder = nn.Sequential(*encoder_layers)

        # ---------- DECODER ----------
        decoder_layers = []
        prev_dim = bn_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(cast(Any, type(activation)()))
            prev_dim = h

        decoder_layers.append(nn.Linear(prev_dim, in_dim))
        if output_activation is not None:
            decoder_layers.append(cast(Any, output_activation))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        z = self.encoder(X)
        out = self.decoder(z)
        return out

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        return self.encoder(X)

    def decode(self, X: torch.Tensor) -> torch.Tensor:
        return self.decoder(X)


class TiedAutoEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        bn_dim: int,
        hidden_dims: int | Sequence[int],
        activation: nn.Module = nn.ReLU(),
        output_activation: nn.Module | None = None,
        encoder_bias: bool = True,
        decoder_bias: bool = False,
        dropout_rate: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.activation = activation
        self.output_activation = output_activation
        self.use_decoder_bias = decoder_bias

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        else:
            hidden_dims = list(hidden_dims)

        # ---------- ENCODER ----------
        encoder_layers: list[nn.Module] = []
        prev_dim = in_dim

        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h, bias=encoder_bias))
            encoder_layers.append(type(activation)())  # new instance
            if use_layer_norm:
                encoder_layers.append(nn.LayerNorm(h))
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = h

        bn_layer = nn.Linear(prev_dim, bn_dim, bias=encoder_bias)
        encoder_layers.append(bn_layer)

        self.encoder = nn.Sequential(*encoder_layers)

        # indices of the Linear layers in the encoder
        self.linear_idx = [
            i for i, m in enumerate(self.encoder) if isinstance(m, nn.Linear)
        ]

        # ---------- DECODER BIASES ----------
        if self.use_decoder_bias:
            # One bias per decoder Linear layer (in reverse order)
            bias_params: list[nn.Parameter] = []

            for idx in reversed(self.linear_idx):
                lin = cast(nn.Linear, self.encoder[idx])
                bias_params.append(nn.Parameter(torch.zeros(lin.in_features)))

            self.decoder_bias = nn.ParameterList(bias_params)

        else:
            self.decoder_bias = nn.ParameterList([])

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        return self.encoder(X)

    def decode(self, X: torch.Tensor) -> torch.Tensor:
        num_linear = len(self.linear_idx)

        for step, idx in enumerate(reversed(self.linear_idx)):
            lin = cast(nn.Linear, self.encoder[idx])  # nn.Linear
            W: torch.Tensor = lin.weight  # (in_dim, out_dim)
            W_t: torch.Tensor = W.t()  # (out_dim, in_dim)

            if self.use_decoder_bias:
                b: torch.Tensor = self.decoder_bias[step]
                X = F.linear(X, W_t, b)
            else:
                X = F.linear(X, W_t)

            is_last_step = step == num_linear - 1
            if not is_last_step:
                X = self.activation(X)

        if self.output_activation is not None:
            X = self.output_activation(X)

        return X

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Z = self.encode(X)
        out = self.decode(Z)
        return out
