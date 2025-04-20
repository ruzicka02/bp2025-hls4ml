from datetime import datetime

import numpy as np
from pynq import Overlay, allocate


# === Utility functions used within the overlay ===
def quantize(data: np.ndarray) -> np.ndarray:
    """
    Quantize a numpy array of floats to a numpy array of uint16.
    Assumes an interface dtype ap_fixed<16,6>.

    Follows the Vitis HLS implementation as described in:
    https://docs.amd.com/r/en-US/ug1399-vitis-hls/Overview-of-Arbitrary-Precision-Fixed-Point-Data-Types
    """
    if not np.issubdtype(data.dtype, np.floating):
        raise ValueError("Input array must have dtype np.floating")

    # 10 fractional bits
    return (data * (2**10)).astype(np.int16).astype(np.uint16)


def dequantize(data: np.ndarray) -> np.ndarray:
    if data.dtype not in [np.uint16, np.int16]:
        raise ValueError("Input array must be a 16b integer")

    # return back signs for a correct float conversion
    if data.dtype == np.uint16:
        data = data.astype(np.int16)

    # 10 fractional bits
    return data.astype(np.float64) / (2**10)


def bundled_shape(shape: tuple) -> tuple:
    """
    Takes a shape tuple and returns the shape after bundling.
    The last dimension is halved with rounding up due to 16b -> 32b bundling.
    """
    if len(shape) == 0:
        return shape

    return (*shape[:-1], int(np.ceil(shape[-1] / 2)))


def bundle_for_transfer(data: np.ndarray) -> np.ndarray:
    """
    Takes an array of np.uint16 and merges them into np.uint32 across the last dimension.
    In the single uint32, lo bits are the 1st and hi bits are the 2nd value.
    When the value count is odd, fill with zeros.
    """
    if data.dtype != np.uint16:
        raise ValueError("Input array must have dtype np.uint16")

    if data.shape[-1] % 2 == 1:
        padding = np.zeros((*data.shape[:-1], 1), dtype=np.uint16)
        data = np.concatenate((data, padding), axis=-1)

    lo = data[..., ::2].astype(np.uint32)
    hi = data[..., 1::2].astype(np.uint32) << 16

    return lo | hi


def unbundle_from_transfer(data: np.ndarray) -> np.ndarray:
    """
    Takes an array of np.uint32 and splits it into np.uint16 across the last dimension.
    The lo bits become the 1st value and hi bits become the 2nd value.
    """
    if data.dtype != np.uint32:
        raise ValueError("Input array must have dtype np.uint32")

    lo = (data & 0xFFFF).astype(np.uint16)
    hi = (data >> 16).astype(np.uint16)

    return np.stack((lo, hi), axis=-1).reshape(*data.shape[:-1], -1)


class NeuralNetworkOverlay(Overlay):
    def __init__(
        self,
        bitfile_name,
        weights,
        x_shape,
        y_shape,
        dtbo=None,
        download=True,
        ignore_version=False,
        device=None,
    ):
        super().__init__(
            bitfile_name,
            dtbo=dtbo,
            download=download,
            ignore_version=ignore_version,
            device=device,
        )

        self.sendchannel = self.hier_0.axi_dma_0.sendchannel
        self.recvchannel = self.hier_0.axi_dma_0.recvchannel
        self.input_buffer = allocate(shape=bundled_shape(x_shape), dtype=np.uint32)
        self.output_buffer = allocate(shape=bundled_shape(y_shape), dtype=np.uint32)

        self.odd_output = y_shape[-1] % 2 == 1

        weights = weights.flatten()
        if np.issubdtype(weights.dtype, np.floating):
            weights = quantize(weights)

        # assert weights.dtype == np.uint16

        self.weights_buffer = allocate(shape=weights.shape, dtype=weights.dtype)
        self.weights_buffer[: len(weights)] = weights

        weights_addr = self.weights_buffer.device_address

        # lo
        self.hier_0.myproject_axi_0.register_map.weights_1 = weights_addr % 2**32
        # hi
        self.hier_0.myproject_axi_0.register_map.weights_2 = weights_addr // 2**32

    def _print_dt(self, timea, timeb, N):
        dt = timeb - timea
        dts = dt.seconds + dt.microseconds * 10**-6
        rate = N / dts
        print(f"Classified {N} samples in {dts} seconds ({rate} inferences / s)")
        return dts, rate

    def predict(self, X, debug=False, profile=False):
        """
        Obtain the predictions of the NN implemented in the FPGA.
        Parameters:
        - X : the input vector. Should be numpy ndarray.
        - dtype : the data type of the elements of the input/output vectors.
                  Note: it should be set depending on the interface of the accelerator; if it uses 'float'
                  types for the 'data' AXI-Stream field, 'np.float32' dtype is the correct one to use.
                  Instead if it uses 'ap_fixed<A,B>', 'np.intA' is the correct one to use (note that A cannot
                  any integer value, but it can assume {..., 8, 16, 32, ...} values. Check `numpy`
                  doc for more info).
                  In this case the encoding/decoding has to be computed by the PS. For example for
                  'ap_fixed<16,6>' type the following 2 functions are the correct one to use for encode/decode
                  'float' -> 'ap_fixed<16,6>':
                  ```
                    def encode(xi):
                        return np.int16(round(xi * 2**10)) # note 2**10 = 2**(A-B)
                    def decode(yi):
                        return yi * 2**-10
                    encode_v = np.vectorize(encode) # to apply them element-wise
                    decode_v = np.vectorize(decode)
                  ```
        - profile : boolean. Set it to `True` to print the performance of the algorithm in term of `inference/s`.
        - encode/decode: function pointers. See `dtype` section for more information.
        - return: an output array based on `np.ndarray` with a shape equal to `y_shape` and a `dtype` equal to
                  the namesake parameter.
        """
        if profile:
            timea = datetime.now()

        if np.issubdtype(X.dtype, np.floating):
            X = quantize(X)

        if X.dtype == np.uint16:
            X = bundle_for_transfer(X)

        assert X.dtype == np.uint32, f"Unsupported dtype {X.dtype}"

        self.input_buffer[:] = X
        self.recvchannel.transfer(self.output_buffer)

        # time

        self.sendchannel.transfer(self.input_buffer)
        if debug:
            print("Transfer OK")

        self.recvchannel.wait()
        if debug:
            print("Receive OK.")

        # time

        self.sendchannel.wait()
        if debug:
            print("Send OK")

        result = unbundle_from_transfer(self.output_buffer)
        if self.odd_output:
            result = result[..., :-1]
        result = dequantize(result)

        if profile:
            timeb = datetime.now()
            dts, rate = self._print_dt(timea, timeb, len(X))
            return result, dts, rate
        else:
            return result
