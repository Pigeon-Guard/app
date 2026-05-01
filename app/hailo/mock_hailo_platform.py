from __future__ import annotations
import hashlib
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np


class HailoSchedulingAlgorithm:
    NONE = "NONE"


class HailoStreamInterface:
    PCIe = "PCIe"


class FormatType:
    FLOAT32 = "FLOAT32"
    UINT8 = "UINT8"


@dataclass
class _VStreamInfo:
    name: str
    shape: tuple


class VDeviceParams:
    def __init__(self):
        self.scheduling_algorithm = HailoSchedulingAlgorithm.NONE


class VDevice:
    @staticmethod
    def create_params():
        return VDeviceParams()

    def __init__(self, params=None):
        self.params = params or VDeviceParams()

    def configure(self, hef, configure_params):
        return [FakeNetworkGroup(hef)]


class HEF:
    def __init__(self, hef_path: str):
        self.hef_path = hef_path

        self._input_infos = [
            _VStreamInfo(name="input_0", shape=(224, 224, 3))
        ]
        self._output_infos = [
            _VStreamInfo(name="output_0", shape=(1000,))
        ]

    def get_input_vstream_infos(self):
        return self._input_infos

    def get_output_vstream_infos(self):
        return self._output_infos


class ConfigureParams:
    @staticmethod
    def create_from_hef(hef, interface):
        return SimpleNamespace(hef=hef, interface=interface)


class InputVStreamParams:
    @staticmethod
    def make(network_group, quantized=False, format_type=FormatType.FLOAT32):
        return SimpleNamespace(
            quantized=quantized,
            format_type=format_type,
            direction="input",
        )


class OutputVStreamParams:
    @staticmethod
    def make(network_group, quantized=False, format_type=FormatType.FLOAT32):
        return SimpleNamespace(
            quantized=quantized,
            format_type=format_type,
            direction="output",
        )


class _ActivatedNetworkGroup:
    def __init__(self, network_group, params):
        self.network_group = network_group
        self.params = params

    def __enter__(self):
        self.network_group._is_active = True
        return self.network_group

    def __exit__(self, exc_type, exc, tb):
        self.network_group._is_active = False
        return False


class FakeNetworkGroup:
    def __init__(self, hef: HEF):
        self.hef = hef
        self._is_active = False

    def create_params(self):
        return SimpleNamespace(kind="fake-network-group-params")

    def activate(self, params):
        return _ActivatedNetworkGroup(self, params)


class InferVStreams:
    def __init__(self, network_group, input_vstreams_params, output_vstreams_params):
        self.network_group = network_group
        self.input_vstreams_params = input_vstreams_params
        self.output_vstreams_params = output_vstreams_params

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def infer(self, input_data: dict[str, np.ndarray]):
        if not self.network_group._is_active:
            raise RuntimeError("Network group must be activated before infer().")

        input_infos = self.network_group.hef.get_input_vstream_infos()
        output_infos = self.network_group.hef.get_output_vstream_infos()

        input_name = input_infos[0].name
        output_name = output_infos[0].name
        output_shape = output_infos[0].shape

        x = input_data[input_name]
        x = np.asarray(x, dtype=np.float32)

        if x.ndim != 4:
            raise ValueError(f"Expected batched input with ndim=4, got shape={x.shape}")

        batch_size = x.shape[0]
        outputs = np.empty((batch_size, *output_shape), dtype=np.float32)

        for i in range(batch_size):
            sample = x[i]
            h = hashlib.sha256(sample.tobytes()).digest()
            seed = int.from_bytes(h[:8], "little")
            rng = np.random.default_rng(seed)

            y = rng.standard_normal(output_shape, dtype=np.float32)

            y += sample.mean() * 0.1
            y -= sample.std() * 0.05

            outputs[i] = y

        return {output_name: outputs}