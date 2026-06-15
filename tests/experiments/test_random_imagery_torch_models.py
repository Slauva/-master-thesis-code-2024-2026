from pathlib import Path

import pytest
import torch
from torch import nn

from experiments.random_imagery_torch import (
    ALL_ARCHITECTURES,
    DeepConvNet,
    EEGNet,
    EEGNetSSVEP,
    EEGNetV1,
    ShallowConvNet,
    SpectralModel,
    SpectralModelShape,
    build_spectral_model,
    project_weight_max_norm_,
    trainable_parameter_count,
)
from experiments.random_imagery_torch.models import ArchitectureName

INPUT_SHAPES = {
    "fft": SpectralModelShape(input_planes=1, electrodes=63, width=39),
    "morlet": SpectralModelShape(input_planes=39, electrodes=63, width=49),
    "superlet": SpectralModelShape(input_planes=39, electrodes=63, width=46),
    "stft": SpectralModelShape(input_planes=39, electrodes=63, width=51),
}

PARAMETER_SNAPSHOTS = {
    "eegnet": {
        "fft": 2_412,
        "morlet": 17_436,
        "superlet": 16_484,
        "stft": 18_060,
    },
    "deep-convnet": {
        "fft": 179_136,
        "morlet": 183_886,
        "superlet": 183_886,
        "stft": 183_886,
    },
    "shallow-convnet": {
        "fft": 102_916,
        "morlet": 122_676,
        "superlet": 122_676,
        "stft": 122_676,
    },
    "eegnet-ssvep": {
        "fft": 23_940,
        "morlet": 203_940,
        "superlet": 192_612,
        "stft": 211_428,
    },
    "eegnet-v1": {
        "fft": 2_028,
        "morlet": 40_332,
        "superlet": 40_332,
        "stft": 40_332,
    },
}

MODEL_TYPES = {
    "eegnet": EEGNet,
    "deep-convnet": DeepConvNet,
    "shallow-convnet": ShallowConvNet,
    "eegnet-ssvep": EEGNetSSVEP,
    "eegnet-v1": EEGNetV1,
}


@pytest.mark.parametrize("architecture", ALL_ARCHITECTURES)
@pytest.mark.parametrize("method", tuple(INPUT_SHAPES))
def test_all_architectures_forward_backward_on_every_input_geometry(
    architecture: ArchitectureName,
    method: str,
) -> None:
    torch.manual_seed(7)
    shape = INPUT_SHAPES[method]
    model = build_spectral_model(
        architecture,
        input_shape=shape,
        dropout_rate=0.0,
    )
    inputs = torch.randn(2, *shape.tensor_shape, requires_grad=True)

    logits = model(inputs)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        torch.randint(0, 2, logits.shape, dtype=torch.float32),
    )
    loss.backward()

    assert logits.shape == (2, 36)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert gradients and all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients
    )


@pytest.mark.parametrize("architecture", ALL_ARCHITECTURES)
@pytest.mark.parametrize("method", tuple(INPUT_SHAPES))
def test_parameter_count_snapshots(
    architecture: ArchitectureName,
    method: str,
) -> None:
    model = build_spectral_model(architecture, input_shape=INPUT_SHAPES[method])

    assert model.parameter_count == PARAMETER_SNAPSHOTS[architecture][method]
    assert trainable_parameter_count(model) == PARAMETER_SNAPSHOTS[architecture][method]


@pytest.mark.parametrize("architecture", ALL_ARCHITECTURES)
def test_factory_returns_expected_model_type(architecture: ArchitectureName) -> None:
    model = build_spectral_model(architecture, input_shape=INPUT_SHAPES["fft"])

    assert isinstance(model, MODEL_TYPES[architecture])
    assert isinstance(model, SpectralModel)
    assert model.architecture == architecture


@pytest.mark.parametrize(
    ("model_type", "expected_groups", "expected_outputs"),
    [
        (EEGNet, 8, 16),
        (EEGNetSSVEP, 96, 96),
    ],
)
def test_eegnet_depthwise_grouping(
    model_type: type[EEGNet],
    expected_groups: int,
    expected_outputs: int,
) -> None:
    model = model_type(input_shape=INPUT_SHAPES["fft"])

    assert model.spatial_depthwise.groups == expected_groups
    assert model.spatial_depthwise.in_channels == expected_groups
    assert model.spatial_depthwise.out_channels == expected_outputs
    assert model.separable_depthwise.groups == expected_outputs
    assert model.separable_depthwise.in_channels == expected_outputs
    assert model.separable_depthwise.out_channels == expected_outputs


@pytest.mark.parametrize("architecture", ALL_ARCHITECTURES)
def test_initialization_is_deterministic_for_a_fixed_seed(
    architecture: ArchitectureName,
) -> None:
    torch.manual_seed(1234)
    first = build_spectral_model(architecture, input_shape=INPUT_SHAPES["fft"])
    first_state = {
        name: tensor.detach().clone()
        for name, tensor in first.state_dict().items()
    }
    torch.manual_seed(1234)
    second = build_spectral_model(architecture, input_shape=INPUT_SHAPES["fft"])

    assert first_state.keys() == second.state_dict().keys()
    for name, expected in first_state.items():
        torch.testing.assert_close(second.state_dict()[name], expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("architecture", ALL_ARCHITECTURES)
def test_model_rejects_wrong_input_geometry(architecture: ArchitectureName) -> None:
    model = build_spectral_model(architecture, input_shape=INPUT_SHAPES["fft"])

    with pytest.raises(ValueError, match="expected input shape"):
        model(torch.randn(2, 1, 63, 40))


@pytest.mark.parametrize(
    "architecture",
    ("eegnet", "deep-convnet", "shallow-convnet", "eegnet-ssvep"),
)
def test_model_max_norm_projection_enforces_every_constraint(
    architecture: ArchitectureName,
) -> None:
    model = build_spectral_model(architecture, input_shape=INPUT_SHAPES["fft"])
    constraints = model.max_norm_constraints()
    assert constraints
    with torch.no_grad():
        for constraint in constraints:
            constraint.module.weight.fill_(10.0)

    model.project_max_norm_()

    for constraint in constraints:
        weight = constraint.module.weight
        dimensions = tuple(range(1, weight.ndim))
        norms = torch.linalg.vector_norm(weight, dim=dimensions)
        assert torch.all(norms <= constraint.maximum + 1e-6)
        assert torch.any(norms >= constraint.maximum - 1e-5)


def test_standalone_max_norm_projection_validates_inputs() -> None:
    linear = nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        linear.weight.fill_(2.0)

    project_weight_max_norm_(linear, 0.5)

    torch.testing.assert_close(
        torch.linalg.vector_norm(linear.weight, dim=1),
        torch.full((3,), 0.5),
    )
    with pytest.raises(ValueError, match="positive"):
        project_weight_max_norm_(linear, 0.0)
    with pytest.raises(TypeError, match="Conv2d and Linear"):
        project_weight_max_norm_(nn.BatchNorm1d(3), 1.0)  # type: ignore[arg-type]


def test_eegnet_supports_spatial_dropout() -> None:
    model = EEGNet(
        input_shape=INPUT_SHAPES["fft"],
        dropout_type="spatial",
    )

    assert isinstance(model.dropout1, nn.Dropout2d)
    assert isinstance(model.dropout2, nn.Dropout2d)


def test_upstream_license_and_adaptation_notice_are_retained() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    package_dir = project_dir / "experiments" / "random_imagery_torch"
    license_text = (package_dir / "ARL_EEGMODELS_LICENSE.txt").read_text(encoding="utf-8")
    notice_text = (package_dir / "ARL_EEGMODELS_NOTICE.md").read_text(encoding="utf-8")
    model_text = (package_dir / "models.py").read_text(encoding="utf-8")

    assert "Creative Commons Zero (CC0) License" in license_text
    assert "Apache License" in license_text
    assert "END OF TERMS AND CONDITIONS" in license_text
    assert "https://github.com/vlawhern/arl-eegmodels" in notice_text
    assert "not numerically identical translations" in notice_text
    assert "ARL_EEGMODELS_LICENSE.txt" in model_text
    assert not (project_dir / "eegnet-tesnorflow.py").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("architecture", ALL_ARCHITECTURES)
def test_cuda_forward_backward_smoke(architecture: ArchitectureName) -> None:
    device = torch.device("cuda")
    shape = INPUT_SHAPES["fft"]
    model = build_spectral_model(
        architecture,
        input_shape=shape,
        dropout_rate=0.0,
    ).to(device)
    inputs = torch.randn(2, *shape.tensor_shape, device=device, requires_grad=True)

    logits = model(inputs)
    logits.square().mean().backward()
    torch.cuda.synchronize()

    assert logits.device.type == "cuda"
    assert torch.isfinite(logits).all()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
