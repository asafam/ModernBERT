# Copyright 2024 **AUTHORS_TODO**
# License: Apache-2.0

# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2023, Tri Dao.

import torch
import torch.nn as nn

from .configuration_bert import FlexBertConfig, maybe_add_padding
from .activation import get_act_fn
from .normalization import get_norm_layer


class BertResidualGLU(nn.Module):
    """Applies the FFN at the end of each Mosaic BERT layer.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality, but
    introduces Gated Linear Units.

    Note: Mosaic BERT adds parameters in order to implement Gated Linear Units. To keep parameter count consistent with that of a
    standard Hugging Face BERT, scale down `config.intermediate_size` by 2/3. For example, a Mosaic BERT constructed with
    `config.intermediate_size=2048` will have the same parameter footprint as its Hugging Face BERT counterpart constructed
    with the `config.intermediate_size=3072`.
    However, in most cases it will not be necessary to adjust `config.intermediate_size` since, despite the increased
    parameter size, Mosaic BERT typically offers a net higher throughput than a Hugging Face BERT built from the same `config`.
    """

    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        self.gated_layers = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.act = get_act_fn(config.hidden_act)
        self.wo = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layernorm = get_norm_layer(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute new hidden states from current hidden states.

        Args:
            hidden_states (torch.Tensor): The (unpadded) hidden states from
                the attention layer [nnz, dim].
        """
        residual_connection = hidden_states
        # compute the activation
        hidden_states = self.gated_layers(hidden_states)
        gated = hidden_states[:, : self.config.intermediate_size]
        non_gated = hidden_states[:, self.config.intermediate_size :]
        hidden_states = self.act(gated) * non_gated
        hidden_states = self.dropout(hidden_states)
        # multiply by the second matrix
        hidden_states = self.wo(hidden_states)
        # add the residual connection and post-LN
        hidden_states = self.layernorm(hidden_states + residual_connection)
        return hidden_states


class FlexBertMLPBase(nn.Module):
    """A FlexBERT MLP base class for type hints."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This is a base class and should not be used directly.")


class FlexBertMLP(FlexBertMLPBase):
    """Applies the MLP at the end of each FlexBERT layer.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality.
    """

    def __init__(self, config: FlexBertConfig):
        super().__init__()
        self.Wi = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_in_bias)
        self.act = get_act_fn(config)
        self.drop = nn.Dropout(config.mlp_dropout_prob) if config.mlp_dropout_prob > 0.0 else nn.Identity()
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_out_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute new hidden states from current hidden states.

        Args:
            hidden_states (torch.Tensor): The (unpadded) hidden states from
                the attention layer [nnz, dim].
        """
        return self.Wo(self.drop(self.act(self.Wi(hidden_states))))


class FlexBertGLU(FlexBertMLPBase):
    """Applies the GLU at the end of each FlexBERT layer.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality.
    """

    def __init__(self, config: FlexBertConfig):
        super().__init__()
        self.Wi = nn.Linear(config.hidden_size, int(config.intermediate_size), bias=config.mlp_in_bias)
        self.act = get_act_fn(config)
        self.drop = nn.Dropout(config.mlp_dropout_prob) if config.mlp_dropout_prob > 0.0 else nn.Identity()
        self.Wo = nn.Linear(config.intermediate_size // 2, config.hidden_size, bias=config.mlp_out_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(input) * gate))


class FlexBertParallelGLU(FlexBertMLPBase):
    """Applies the GLU at the end of each FlexBERT layer using intermediate_ff computed in parallel of the attention.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality.
    """

    def __init__(self, config: FlexBertConfig):
        super().__init__()
        self.act = get_act_fn(config)
        self.drop = nn.Dropout(config.mlp_dropout_prob) if config.mlp_dropout_prob > 0.0 else nn.Identity()
        self.Wo = nn.Linear(config.intermediate_size // 2, config.hidden_size, bias=config.mlp_out_bias)

    def forward(self, intermediate_ff: torch.Tensor) -> torch.Tensor:
        input, gate = intermediate_ff.chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(input) * gate))

MLP2CLS = {
    "mlp": FlexBertMLP,
    "glu": FlexBertGLU,
    "parallel_glu": FlexBertParallelGLU,
}


def get_mlp_layer(config: FlexBertConfig) -> FlexBertMLPBase:
    try:
        return MLP2CLS[config.mlp_layer](config)
    except KeyError:
        raise ValueError(f"Invalid MLP layer type: {config.mlp_layer=}, must be one of {MLP2CLS.keys()}.")
