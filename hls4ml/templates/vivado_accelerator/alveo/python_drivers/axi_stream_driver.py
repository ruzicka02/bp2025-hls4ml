from datetime import datetime

import numpy as np
from pynq import Overlay, allocate


class NeuralNetworkOverlay(Overlay):
    def __init__(self, xclbin_name, dtbo=None, download=True, ignore_version=False, device=None):
        super().__init__(xclbin_name, dtbo=dtbo, download=download, ignore_version=ignore_version, device=device)
        self.input_buffer = None
        self.output_buffer = None
        self.weight_buffer = None
        self.weight_size = None

    def allocate_mem(self, X_shape, y_shape, dtype=np.float32, trg_in=None, trg_out=None):
        """Buffer allocation in the accelerator's memory.

        Args:
            X_shape (list): Input buffer shape.
            y_shape (list): Output buffer shape.
            w_shape (list): Weights buffer shape.
            dtype (dtype, optional): The data type of the elements of the input/output tensors. Must be an instance of
                numpy dtype. Defaults to np.float32.

                It should be set depending on the interface of the accelerator; if it uses 'float'
                data type for the 'data' AXI-Stream field, 'np.float32' dtype must be used. Instead if it uses
                'ap_fixed<A,B>', 'np.intA' is the correct dtype to use. Note that A cannot any integer value, but it can
                assume power of 2 values, i.e., {..., 8, 16, 32, ...}. Check `numpy` documentation for more information.
                In this case the encoding/decoding has to be computed by the host machine. For example for
                'ap_fixed<16,6>' type the following 2 functions are the correct one to use for encode/decode
                'float' -> 'ap_fixed<16,6>'::

                    def encode(xi):
                        return np.int16(round(xi * 2**10)) # note 2**10 = 2**(A-B)
                    def decode(yi):
                        return yi * 2**-10
                    encode_v = np.vectorize(encode) # to apply them element-wise
                    decode_v = np.vectorize(decode)

            trg_in (optional): Input buffer target memory. By default the v++ command set it to HBM[0] for
                alveo-u50. Defaults to None.
            trg_out (optional): Output buffer target memory. By default the v++ command set it to HBM[0] for
                alveo-u50. Defaults to None.
        """
        self.input_buffer = allocate(shape=X_shape, dtype=dtype, target=trg_in)
        self.output_buffer = allocate(shape=y_shape, dtype=dtype, target=trg_out)

    def fill_weights(self, data: np.ndarray):
        assert data.dtype == np.int16, "Incorrect data type, stuff may fail"
        # assert np.prod(data.shape) == CONSTANT_PASSED_FROM_HLS4ML, "Data size must match the buffer size"

        self.weight_size = data.shape

        self.weight_buffer = allocate(shape=data.shape, dtype=data.dtype, target=None)
        self.weight_buffer[:] = data.flatten()
        self.weight_buffer.sync_to_device()

    def predict(self, X, y_shape, dtype=np.float32, debug=False, profile=False, encode=None, decode=None):
        """Obtain the predictions of the NN implemented in the FPGA.

        Args:
            X (ndarray): The input tensor.
            y_shape (list): The shape of the output tensor, needed by the accelerator to set the TLAST bit properly.
            dtype (dtype, optional): The data type of the elements of the input/output tensors. Must be an instance of
                numpy dtype. Defaults to np.float32.
            debug (bool, optional): If set, the function will print information about the data transfers status.
                Defaults to False.
            profile (bool, optional): If set, the function will print the performance of the algorithm in terms of
                inference/s. Defaults to False.
            encode (Callable, optional): Function to transform the input tensor. Defaults to None.
            decode (Callable, optional): Function to transform the output tensor. Defaults to None.

        Returns:
            _type_: A ``np.ndarray`` with a shape equal of ``y_shape`` and ``dtype`` data type.
        """
        self.allocate_mem(X_shape=X.shape, y_shape=y_shape, dtype=dtype)
        if profile:
            timea = datetime.now()
        if encode is not None:
            X = encode(X)
        in_size = np.prod(X.shape)
        out_size = np.prod(y_shape)
        self.input_buffer[:] = X
        self.input_buffer.sync_to_device()
        if debug:
            print("Send OK")
        self.krnl_rtl_1.call(self.input_buffer, self.weight_buffer, self.output_buffer, in_size, self.weight_size, out_size)
        if debug:
            print("Kernel call OK")
        self.output_buffer.sync_from_device()
        if debug:
            print("Recieve OK")
        result = self.output_buffer.copy()
        if profile:
            timeb = datetime.now()
            dts, rate = self._print_dt(timea, timeb, len(X))
            self.input_buffer.flush()
            self.output_buffer.flush()
            self.free()
            return result, dts, rate
        self.input_buffer.flush()
        self.output_buffer.flush()
        return result

    def free_overlay(self):
        self.free()

    def _print_dt(self, timea, timeb, N):
        dt = timeb - timea
        dts = dt.seconds + dt.microseconds * 10**-6
        rate = N / dts
        print(f"Classified {N} samples in {dts} seconds ({rate} inferences / s)")
        print(f"Or {1 / rate * 1e6} us / inferences")
        return dts, rate


if __name__ == "__main__":
    with open("tb_weights.dat") as f:
        weights: list[float] = [float(x) for x in f.read().split()]
    print(weights)

    # assume ap_fixed<16, 6> == 1 bit sign, 6 bit integer part, 9 bit fraction part
    # numpy doesn't have fixed point numbers, so we use ints
    fixed_point_convertor = 2 ** 9
    weights_fixed = np.ndarray(len(weights), np.int16)
    weights_fixed[:] = [x * fixed_point_convertor for x in weights]

    print(weights_fixed / fixed_point_convertor)

    with open("tb_input_features.dat") as f:
        # 4 values per row, assume normalized
        inputs: list[float] = [float(x) for x in f.read().split()]

    inputs_fixed = np.ndarray(len(inputs), np.int16)
    inputs_fixed[:] = [x * fixed_point_convertor for x in inputs]
    inputs_fixed = inputs_fixed.reshape(-1, 4)

    # one-hot encoding of 3 classes
    output_shape = (inputs_fixed.shape[0], 3)

    nn = NeuralNetworkOverlay('iris_hls4ml_prj_kernel.xclbin')
    nn.fill_weights(weights_fixed)
    y_hw, latency, throughput = nn.predict(inputs_fixed, output_shape, dtype=np.int16, profile=True)
