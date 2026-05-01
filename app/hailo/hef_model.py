import cv2
import logging
import torch
import numpy as np
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

try:
    import hailo_platform as hpf
except ImportError:
    import app.hailo.mock_platform as hpf
    logging.warning("hailo_platform not available, using mock implementation")

class HEFModel:
    def __init__(self, hef_path: str):

        # Setting VDevice params to disable the HailoRT service feature
        params = hpf.VDevice.create_params()
        params.scheduling_algorithm = hpf.HailoSchedulingAlgorithm.NONE

        # The target can be used as a context manager ("with" statement) to ensure it's released on time.
        # Here it's avoided for the sake of simplicity
        self.target = hpf.VDevice(params=params)

        # Loading compiled HEFs to device:
        self.hef = hpf.HEF(hef_path)
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = self.hef.get_output_vstream_infos()

        # Get the "network groups" (connectivity groups, aka. "different networks") information from the .hef
        configure_params = hpf.ConfigureParams.create_from_hef(hef=self.hef, interface=hpf.HailoStreamInterface.PCIe)
        network_groups = self.target.configure(self.hef, configure_params)

        self.network_group = network_groups[0]
        self.network_group_params = self.network_group.create_params()

        # Create input and output virtual streams params
        # Quantized argument signifies whether or not the incoming data is already quantized.
        # Data is quantized by HailoRT if and only if quantized == False .
        self.input_vstreams_params = hpf.InputVStreamParams.make(self.network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)
        self.output_vstreams_params = hpf.OutputVStreamParams.make(self.network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)

    def _infer_raw(self, x):
        """Returns full raw output dict + orig_shape"""
        x = np.asarray(x)
        orig_shape = x.shape[:2]  # (H,W)

        x = cv2.resize(x, (224, 224)).astype(np.float32) / 255.0
        batch = np.expand_dims(x, axis=0)  # (1,224,224,3)

        input_data = {self.input_vstream_info.name: batch}

        with hpf.InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                y_raw = infer_pipeline.infer(input_data)  # Dict of ALL outputs

        return y_raw, orig_shape

    def _postprocess(self, y_raw, orig_shape):
        boxes_raw = np.asarray(y_raw['best/concat18']).astype(np.float32)
        obj_raw = np.asarray(y_raw['best/activation2']).astype(np.float32)

        boxes_raw = np.squeeze(boxes_raw, axis=1)  # (1, 1029, 64)
        obj_raw = np.squeeze(obj_raw, axis=1)      # (1, 1029, 1)

        print("boxes_raw.shape =", boxes_raw.shape)
        print("obj_raw.shape   =", obj_raw.shape)
        print("boxes_raw[0, 0, :10] =", boxes_raw[0, 0, :10])
        print("boxes_raw[0, 1, :10] =", boxes_raw[0, 1, :10])
        print("obj_raw[0, :10, 0]   =", obj_raw[0, :10, 0])

        raise RuntimeError("Need decoded tensor semantics before mapping to Ultralytics format")

    # TODO: Use yolo postprocessing from hailo_model_zoo instead of reinventing the wheel here.
    def predict(self, x):
        """Full pipeline: preprocess → infer → postprocess → Results"""
        y_raw, orig_shape = self._infer_raw(x)
        return self._postprocess(y_raw, orig_shape)