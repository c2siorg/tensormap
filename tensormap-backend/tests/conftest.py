"""Test configuration for backend test imports."""

import sys
import types
from pathlib import Path

# Ensure backend root (containing app/) is importable when pytest is invoked from any cwd.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _install_tensorflow_stub() -> None:
    """Install a minimal tensorflow stub for unit tests when tensorflow is unavailable."""

    class _Tensor:
        def __init__(self, name: str, shape=None):
            self.name = name
            self.shape = shape

    class _BaseLayer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, input_tensor):
            return _Tensor(name=self.kwargs.get("name", "layer"), shape=getattr(input_tensor, "shape", None))

    class _Dense(_BaseLayer):
        pass

    class _Flatten(_BaseLayer):
        pass

    class _Conv2D(_BaseLayer):
        pass

    class _Concatenate:
        def __init__(self, axis=-1):
            self.axis = axis

        def __call__(self, tensors):
            return _Tensor(name="concat", shape=getattr(tensors[0], "shape", None))

    class _Model:
        def __init__(self, inputs, outputs):
            self.inputs = inputs
            self.outputs = outputs

        def to_json(self) -> str:
            return '{"class_name": "Model", "config": {"name": "stub-model"}}'

    keras_layers = types.SimpleNamespace(
        Dense=_Dense,
        Flatten=_Flatten,
        Conv2D=_Conv2D,
        Concatenate=_Concatenate,
    )
    keras_mod = types.SimpleNamespace(
        Input=lambda shape, name: _Tensor(name=name, shape=shape),
        layers=keras_layers,
        Model=_Model,
    )
    sys.modules["tensorflow"] = types.SimpleNamespace(keras=keras_mod)


try:
    __import__("tensorflow")
except ModuleNotFoundError:
    _install_tensorflow_stub()
