#ifndef MYPROJECT_AXI_H_
#define MYPROJECT_AXI_H_

#include <iostream>
#include "ap_axi_sdata.h"
// hls-fpga-machine-learning insert include

// hls-fpga-machine-learning insert definitions

void myproject_axi(hls::stream<input_axi_t> &in, hls::stream<output_axi_t> &out);
#endif
