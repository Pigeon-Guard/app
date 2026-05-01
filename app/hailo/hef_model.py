import logging
import numpy as np

try:
    import hailo_platform as hpf
except ImportError:
    import mock_hailo_platform as hpf
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
        self.output_vstream_info = self.hef.get_output_vstream_infos()[0]

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

    def predict(self, x):
        assert x.shape == (224, 224, 3)

        batch = np.expand_dims(x, axis=0).astype(np.float32)
        input_data = { self.input_vstream_info.name: batch }

        y = None
        with hpf.InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                output_data = infer_pipeline.infer(input_data)
                y = output_data[self.output_vstream_info.name][0]

        return y