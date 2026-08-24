"""Local extension of torch_harmonics' SFNO that exposes inner_skip/outer_skip.

torch_harmonics.examples.models.sfno.SphericalFourierNeuralOperator's block
construction loop hardcodes inner_skip="none", outer_skip="identity" for every
SphericalFourierNeuralOperatorBlock, so the paper's inner-skip formula
sigma(Decoder(Encoder(x) + Wx)) + x is unreachable through the public
constructor (see debug_train_8_26/stage0.md). This subclass calls the parent
constructor as-is (encoder/decoder/SHT setup are unaffected), then rebuilds
self.blocks with the exact same per-block arguments the parent used, just
threading inner_skip/outer_skip through.
"""
import torch
import torch.nn as nn
from torch_harmonics.examples.models.sfno import (
    SphericalFourierNeuralOperator as _SphericalFourierNeuralOperator,
    SphericalFourierNeuralOperatorBlock,
)


class SphericalFourierNeuralOperator(_SphericalFourierNeuralOperator):
    def __init__(
        self,
        *,
        inner_skip="linear",
        outer_skip="identity",
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_mlp=True,
        bias=False,
        **kwargs,
    ):
        super().__init__(
            mlp_ratio=mlp_ratio, drop_rate=drop_rate, drop_path_rate=drop_path_rate,
            use_mlp=use_mlp, bias=bias, **kwargs,
        )

        dpr = [t.item() for t in torch.linspace(0, drop_path_rate, self.num_layers)]
        blocks = nn.ModuleList([])
        for i in range(self.num_layers):
            first_layer = i == 0
            last_layer = i == self.num_layers - 1
            blocks.append(SphericalFourierNeuralOperatorBlock(
                self.trans_down if first_layer else self.trans,
                self.itrans_up if last_layer else self.itrans,
                self.embed_dim, self.embed_dim,
                mlp_ratio=mlp_ratio, drop_rate=drop_rate, drop_path=dpr[i],
                act_layer=self.activation_function, norm_layer=self.normalization_layer,
                use_mlp=use_mlp, bias=bias,
                inner_skip=inner_skip, outer_skip=outer_skip,
            ))
        self.blocks = blocks
