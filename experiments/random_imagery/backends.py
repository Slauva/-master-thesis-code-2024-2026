from experiments.random_imagery.classifier_backend import (
    LinearSVMBackend,
    RidgeClassifierBackend,
)
from experiments.random_imagery.contracts import AnyRandomImageryModelBackend
from experiments.random_imagery.regression_backend import (
    ElasticNetIndependentBackend,
    ElasticNetMultiOutputBackend,
    PLSRegressionMultiOutputBackend,
    RandomForestIndependentBackend,
    RandomForestMultiOutputBackend,
    RidgeRegressionIndependentBackend,
    RidgeRegressionMultiOutputBackend,
)

_BACKEND_TYPES = {
    "linear-svm-independent": LinearSVMBackend,
    "ridge-classifier-independent": RidgeClassifierBackend,
    "ridge-regression-independent": RidgeRegressionIndependentBackend,
    "ridge-regression-multioutput": RidgeRegressionMultiOutputBackend,
    "elastic-net-independent": ElasticNetIndependentBackend,
    "elastic-net-multioutput": ElasticNetMultiOutputBackend,
    "random-forest-independent": RandomForestIndependentBackend,
    "random-forest-multioutput": RandomForestMultiOutputBackend,
    "pls-regression-multioutput": PLSRegressionMultiOutputBackend,
}


def build_model_backend(model_id: str) -> AnyRandomImageryModelBackend:
    try:
        backend_type = _BACKEND_TYPES[model_id]
    except KeyError as error:
        supported = ", ".join(_BACKEND_TYPES)
        raise ValueError(
            f"Model {model_id!r} has no trainable schema-v3 backend; "
            f"expected one of: {supported}"
        ) from error
    return backend_type()


__all__ = ["build_model_backend"]
