#ifndef MYPROJECT_AXI_H_
#define MYPROJECT_AXI_H_

#include <iostream>
#include "ap_axi_sdata.h"
#include "ap_fixed.h"
// hls-fpga-machine-learning insert include

// hls-fpga-machine-learning insert definitions

void iris_hls4ml_prj_axi(
    hls::stream<input_axi_t>  &in,
    T_in                      *weights,
    hls::stream<output_axi_t> &out
);

#endif
