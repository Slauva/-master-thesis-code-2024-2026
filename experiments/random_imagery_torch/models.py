"""PyTorch spectral adaptations of the ARL EEGModels architectures.

The layer families, filter counts, depthwise/separable structure, activations,
dropout, and max-norm intent follow ARL EEGModels. The input geometry and
pooling are adapted for the project's frequency-domain tensors, so these
modules are not numerically identical TensorFlow translations.

Upstream: https://github.com/vlawhern/arl-eegmodels
License: ``ARL_EEGMODELS_LICENSE.txt`` in this package.
"""

from dataclasses import dataclass
from math import prod
from typing import Literal, TypeAlias

import torch
from torch import nn
from torch.nn import functional as F

ArchitectureName: TypeAlias = Literal[
    "eegnet",
    "deep-convnet",
    "shallow-convnet",
    "eegnet-ssvep",
    "eegnet-v1",
]
DropoutType: TypeAlias = Literal["dropout", "spatial"]

PRIMARY_ARCHITECTURES: tuple[ArchitectureName, ...] = (
    "eegnet",
    "deep-convnet",
    "shallow-convnet",
)
EXPLORATORY_ARCHITECTURES: tuple[ArchitectureName, ...] = (
    "eegnet-ssvep",
    "eegnet-v1",
)
ALL_ARCHITECTURES = PRIMARY_ARCHITECTURES + EXPLORATORY_ARCHITECTURES


@dataclass(frozen=True, slots=True)
class SpectralModelShape:
    input_planes: int
    electrodes: int
    width: int

    def __post_init__(self) -> None:
        if min(self.input_planes, self.electrodes, self.width) < 1:
            raise ValueError("Every spectral model input dimension must be positive")

    @property
    def tensor_shape(self) -> tuple[int, int, int]:
        return self.input_planes, self.electrodes, self.width


@dataclass(frozen=True, slots=True)
class MaxNormConstraint:
    module: nn.Conv2d | nn.Linear
    maximum: float

    def __post_init__(self) -> None:
        if self.maximum <= 0:
            raise ValueError("Max-norm bounds must be positive")


class SpectralModel(nn.Module):
    architecture: ArchitectureName

    def __init__(
        self,
        *,
        input_shape: SpectralModelShape,
        n_outputs: int = 36,
    ) -> None:
        super().__init__()
        if n_outputs < 1:
            raise ValueError("`n_outputs` must be positive")
        self.input_shape = input_shape
        self.n_outputs = n_outputs

    def _validate_input(self, inputs: torch.Tensor) -> None:
        if inputs.ndim != 4:
            raise ValueError(
                "Spectral model input must have shape "
                "(batch, planes, electrodes, spectral_time)"
            )
        if tuple(inputs.shape[1:]) != self.input_shape.tensor_shape:
            raise ValueError(
                f"{self.architecture} expected input shape "
                f"(batch, {self.input_shape.input_planes}, "
                f"{self.input_shape.electrodes}, {self.input_shape.width}), "
                f"got {tuple(inputs.shape)}"
            )
        if not torch.is_floating_point(inputs):
            raise TypeError("Spectral model inputs must use a floating-point dtype")

    def max_norm_constraints(self) -> tuple[MaxNormConstraint, ...]:
        return ()

    @torch.no_grad()
    def project_max_norm_(self) -> None:
        for constraint in self.max_norm_constraints():
            project_weight_max_norm_(constraint.module, constraint.maximum)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class SamePadConv2d(nn.Conv2d):
    """TensorFlow-style SAME padding, including strided and even kernels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_height, input_width = inputs.shape[-2:]
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.stride
        pad_height = _same_padding(input_height, kernel_height, stride_height)
        pad_width = _same_padding(input_width, kernel_width, stride_width)
        padded = F.pad(
            inputs,
            (
                pad_width[0],
                pad_width[1],
                pad_height[0],
                pad_height[1],
            ),
        )
        return F.conv2d(
            padded,
            self.weight,
            self.bias,
            self.stride,
            0,
            self.dilation,
            self.groups,
        )


class EEGNet(SpectralModel):
    architecture: ArchitectureName = "eegnet"

    def __init__(
        self,
        *,
        input_shape: SpectralModelShape,
        n_outputs: int = 36,
        dropout_rate: float = 0.5,
        kernel_length: int = 64,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        head_max_norm: float = 0.25,
        dropout_type: DropoutType = "dropout",
    ) -> None:
        super().__init__(input_shape=input_shape, n_outputs=n_outputs)
        _validate_architecture_parameters(
            dropout_rate=dropout_rate,
            kernel_length=kernel_length,
            filter_counts=(f1, depth_multiplier, f2),
        )
        effective_kernel = min(kernel_length, input_shape.width)
        depthwise_filters = f1 * depth_multiplier
        self.temporal_conv = SamePadConv2d(
            input_shape.input_planes,
            f1,
            (1, effective_kernel),
            bias=False,
        )
        self.temporal_batch_norm = nn.BatchNorm2d(f1, momentum=0.01)
        self.spatial_depthwise = nn.Conv2d(
            f1,
            depthwise_filters,
            (input_shape.electrodes, 1),
            groups=f1,
            bias=False,
        )
        self.spatial_batch_norm = nn.BatchNorm2d(depthwise_filters, momentum=0.01)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = _build_dropout(dropout_type, dropout_rate)
        self.separable_depthwise = SamePadConv2d(
            depthwise_filters,
            depthwise_filters,
            (1, min(16, max(1, input_shape.width // 4))),
            groups=depthwise_filters,
            bias=False,
        )
        self.separable_pointwise = nn.Conv2d(
            depthwise_filters,
            f2,
            kernel_size=1,
            bias=False,
        )
        self.separable_batch_norm = nn.BatchNorm2d(f2, momentum=0.01)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = _build_dropout(dropout_type, dropout_rate)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(f2, n_outputs)
        self.head_max_norm = head_max_norm
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _reset_spectral_modules(self)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._validate_input(inputs)
        features = self.temporal_batch_norm(self.temporal_conv(inputs))
        features = self.spatial_depthwise(features)
        features = F.elu(self.spatial_batch_norm(features))
        features = self.dropout1(self.pool1(features))
        features = self.separable_depthwise(features)
        features = self.separable_pointwise(features)
        features = F.elu(self.separable_batch_norm(features))
        features = self.dropout2(self.pool2(features))
        return self.head(self.global_pool(features).flatten(1))

    def max_norm_constraints(self) -> tuple[MaxNormConstraint, ...]:
        return (
            MaxNormConstraint(self.spatial_depthwise, 1.0),
            MaxNormConstraint(self.head, self.head_max_norm),
        )


class EEGNetSSVEP(EEGNet):
    architecture: ArchitectureName = "eegnet-ssvep"

    def __init__(
        self,
        *,
        input_shape: SpectralModelShape,
        n_outputs: int = 36,
        dropout_rate: float = 0.5,
        kernel_length: int = 256,
        f1: int = 96,
        depth_multiplier: int = 1,
        f2: int = 96,
        dropout_type: DropoutType = "dropout",
    ) -> None:
        super().__init__(
            input_shape=input_shape,
            n_outputs=n_outputs,
            dropout_rate=dropout_rate,
            kernel_length=kernel_length,
            f1=f1,
            depth_multiplier=depth_multiplier,
            f2=f2,
            dropout_type=dropout_type,
        )

    def max_norm_constraints(self) -> tuple[MaxNormConstraint, ...]:
        return (MaxNormConstraint(self.spatial_depthwise, 1.0),)


class DeepConvNet(SpectralModel):
    architecture: ArchitectureName = "deep-convnet"

    def __init__(
        self,
        *,
        input_shape: SpectralModelShape,
        n_outputs: int = 36,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__(input_shape=input_shape, n_outputs=n_outputs)
        _validate_architecture_parameters(dropout_rate=dropout_rate)
        self.temporal_conv = SamePadConv2d(
            input_shape.input_planes,
            25,
            (1, min(5, input_shape.width)),
        )
        self.spatial_conv = nn.Conv2d(
            25,
            25,
            (input_shape.electrodes, 1),
        )
        self.batch_norm1 = nn.BatchNorm2d(25, eps=1e-5, momentum=0.1)
        self.pool1 = nn.MaxPool2d((1, 2))
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = SamePadConv2d(25, 50, (1, 5))
        self.batch_norm2 = nn.BatchNorm2d(50, eps=1e-5, momentum=0.1)
        self.pool2 = nn.MaxPool2d((1, 2))
        self.dropout2 = nn.Dropout(dropout_rate)

        self.conv3 = SamePadConv2d(50, 100, (1, 5))
        self.batch_norm3 = nn.BatchNorm2d(100, eps=1e-5, momentum=0.1)
        self.pool3 = nn.MaxPool2d((1, 2))
        self.dropout3 = nn.Dropout(dropout_rate)

        self.conv4 = SamePadConv2d(100, 200, (1, 5))
        self.batch_norm4 = nn.BatchNorm2d(200, eps=1e-5, momentum=0.1)
        self.pool4 = nn.MaxPool2d((1, 2))
        self.dropout4 = nn.Dropout(dropout_rate)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(200, n_outputs)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _reset_spectral_modules(self)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._validate_input(inputs)
        features = self.temporal_conv(inputs)
        features = self.spatial_conv(features)
        features = self.dropout1(self.pool1(F.elu(self.batch_norm1(features))))
        features = self.dropout2(self.pool2(F.elu(self.batch_norm2(self.conv2(features)))))
        features = self.dropout3(self.pool3(F.elu(self.batch_norm3(self.conv3(features)))))
        features = self.dropout4(self.pool4(F.elu(self.batch_norm4(self.conv4(features)))))
        return self.head(self.global_pool(features).flatten(1))

    def max_norm_constraints(self) -> tuple[MaxNormConstraint, ...]:
        return tuple(
            MaxNormConstraint(module, 2.0)
            for module in (
                self.temporal_conv,
                self.spatial_conv,
                self.conv2,
                self.conv3,
                self.conv4,
            )
        ) + (MaxNormConstraint(self.head, 0.5),)


class ShallowConvNet(SpectralModel):
    architecture: ArchitectureName = "shallow-convnet"

    def __init__(
        self,
        *,
        input_shape: SpectralModelShape,
        n_outputs: int = 36,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__(input_shape=input_shape, n_outputs=n_outputs)
        _validate_architecture_parameters(dropout_rate=dropout_rate)
        self.temporal_conv = SamePadConv2d(
            input_shape.input_planes,
            40,
            (1, min(13, input_shape.width)),
        )
        self.spatial_conv = nn.Conv2d(
            40,
            40,
            (input_shape.electrodes, 1),
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(40, eps=1e-5, momentum=0.1)
        self.pool = nn.AvgPool2d((1, min(35, input_shape.width)), stride=(1, 7))
        self.dropout = nn.Dropout(dropout_rate)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(40, n_outputs)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _reset_spectral_modules(self)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._validate_input(inputs)
        features = self.temporal_conv(inputs)
        features = self.spatial_conv(features)
        features = torch.square(self.batch_norm(features))
        features = self.pool(features)
        features = torch.log(torch.clamp(features, min=1e-7, max=10_000.0))
        features = self.dropout(features)
        return self.head(self.global_pool(features).flatten(1))

    def max_norm_constraints(self) -> tuple[MaxNormConstraint, ...]:
        return (
            MaxNormConstraint(self.temporal_conv, 2.0),
            MaxNormConstraint(self.spatial_conv, 2.0),
            MaxNormConstraint(self.head, 0.5),
        )


class EEGNetV1(SpectralModel):
    architecture: ArchitectureName = "eegnet-v1"

    def __init__(
        self,
        *,
        input_shape: SpectralModelShape,
        n_outputs: int = 36,
        dropout_rate: float = 0.25,
        kernels: tuple[tuple[int, int], tuple[int, int]] = ((2, 32), (8, 4)),
        strides: tuple[int, int] = (2, 4),
    ) -> None:
        super().__init__(input_shape=input_shape, n_outputs=n_outputs)
        _validate_architecture_parameters(dropout_rate=dropout_rate)
        if len(kernels) != 2 or any(len(kernel) != 2 or min(kernel) < 1 for kernel in kernels):
            raise ValueError("EEGNet-v1 requires two positive two-dimensional kernels")
        if len(strides) != 2 or min(strides) < 1:
            raise ValueError("EEGNet-v1 strides must contain two positive values")

        self.spatial_conv = nn.Conv2d(
            input_shape.input_planes,
            16,
            (input_shape.electrodes, 1),
        )
        self.batch_norm1 = nn.BatchNorm2d(16, momentum=0.01)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = SamePadConv2d(
            1,
            4,
            kernels[0],
            stride=strides,
        )
        self.batch_norm2 = nn.BatchNorm2d(4, momentum=0.01)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.conv3 = SamePadConv2d(
            4,
            4,
            kernels[1],
            stride=strides,
        )
        self.batch_norm3 = nn.BatchNorm2d(4, momentum=0.01)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(4, n_outputs)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _reset_spectral_modules(self)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._validate_input(inputs)
        features = self.dropout1(F.elu(self.batch_norm1(self.spatial_conv(inputs))))
        features = features.permute(0, 2, 1, 3)
        features = self.dropout2(F.elu(self.batch_norm2(self.conv2(features))))
        features = self.dropout3(F.elu(self.batch_norm3(self.conv3(features))))
        return self.head(self.global_pool(features).flatten(1))


def build_spectral_model(
    architecture: ArchitectureName,
    *,
    input_shape: SpectralModelShape,
    n_outputs: int = 36,
    dropout_rate: float | None = None,
) -> SpectralModel:
    if architecture == "eegnet":
        return EEGNet(
            input_shape=input_shape,
            n_outputs=n_outputs,
            dropout_rate=0.5 if dropout_rate is None else dropout_rate,
        )
    if architecture == "deep-convnet":
        return DeepConvNet(
            input_shape=input_shape,
            n_outputs=n_outputs,
            dropout_rate=0.5 if dropout_rate is None else dropout_rate,
        )
    if architecture == "shallow-convnet":
        return ShallowConvNet(
            input_shape=input_shape,
            n_outputs=n_outputs,
            dropout_rate=0.5 if dropout_rate is None else dropout_rate,
        )
    if architecture == "eegnet-ssvep":
        return EEGNetSSVEP(
            input_shape=input_shape,
            n_outputs=n_outputs,
            dropout_rate=0.5 if dropout_rate is None else dropout_rate,
        )
    if architecture == "eegnet-v1":
        return EEGNetV1(
            input_shape=input_shape,
            n_outputs=n_outputs,
            dropout_rate=0.25 if dropout_rate is None else dropout_rate,
        )
    raise ValueError(f"Unsupported spectral architecture: {architecture!r}")


@torch.no_grad()
def project_weight_max_norm_(
    module: nn.Conv2d | nn.Linear,
    maximum: float,
) -> None:
    if maximum <= 0:
        raise ValueError("Max-norm bound must be positive")
    if not isinstance(module, (nn.Conv2d, nn.Linear)):
        raise TypeError("Max-norm projection supports Conv2d and Linear modules")
    weight = module.weight
    dimensions = tuple(range(1, weight.ndim))
    norms = torch.linalg.vector_norm(weight, dim=dimensions, keepdim=True)
    epsilon = torch.finfo(weight.dtype).eps
    projected_maximum = maximum * (1.0 - 8.0 * epsilon)
    scales = torch.where(
        norms > maximum,
        projected_maximum / torch.clamp(norms, min=epsilon),
        torch.ones_like(norms),
    )
    weight.mul_(scales)


def _same_padding(
    input_size: int,
    kernel_size: int,
    stride: int,
) -> tuple[int, int]:
    output_size = (input_size + stride - 1) // stride
    total = max((output_size - 1) * stride + kernel_size - input_size, 0)
    before = total // 2
    return before, total - before


def _build_dropout(dropout_type: DropoutType, rate: float) -> nn.Module:
    if dropout_type == "dropout":
        return nn.Dropout(rate)
    if dropout_type == "spatial":
        return nn.Dropout2d(rate)
    raise ValueError("`dropout_type` must be 'dropout' or 'spatial'")


def _validate_architecture_parameters(
    *,
    dropout_rate: float,
    kernel_length: int | None = None,
    filter_counts: tuple[int, ...] = (),
) -> None:
    if not 0 <= dropout_rate < 1:
        raise ValueError("Dropout rate must be in [0, 1)")
    if kernel_length is not None and kernel_length < 1:
        raise ValueError("Kernel length must be positive")
    if any(count < 1 for count in filter_counts):
        raise ValueError("Filter counts and depth multipliers must be positive")


def _reset_spectral_modules(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.BatchNorm2d):
            nn.init.ones_(child.weight)
            nn.init.zeros_(child.bias)


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(prod(parameter.shape) for parameter in module.parameters() if parameter.requires_grad)
