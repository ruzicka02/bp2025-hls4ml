#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "firmware/myproject.h"
#include "firmware/myproject_axi.h"
#include "firmware/nnet_utils/nnet_helpers.h"

// hls-fpga-machine-learning insert bram

#define CHECKPOINT 5000

namespace nnet {
bool trace_enabled = true;
std::map<std::string, void *> *trace_outputs = NULL;
size_t trace_type_size = sizeof(double);
} // namespace nnet

int main(int argc, char **argv) {
    // hls-fpga-machine-learning insert namespace

    // load input data from text file
    std::ifstream fin("tb_data/tb_input_features.dat");
    // load weights from text file
    std::ifstream fwe("tb_data/tb_weights.dat");
    // load predictions from text file
    std::ifstream fpr("tb_data/tb_output_predictions.dat");

#ifdef RTL_SIM
    std::string RESULTS_LOG = "tb_data/rtl_cosim_results.log";
#else
    std::string RESULTS_LOG = "tb_data/csim_results.log";
#endif
    std::ofstream fout(RESULTS_LOG);

    std::string iline;
    std::string wline;
    std::string pline;
    int e = 0;

    if (fin.is_open() && fpr.is_open() && fwe.is_open()) {
        model_default_t weights_buffer[LAYER_WEIGHTS_SIZE];
        int i = 0;
        // load weights from file
        // space and newline are treated identically here (one big array is loaded)
        while (std::getline(fwe, wline)) {
            char *cstr = const_cast<char *>(wline.c_str());
            char *current;
            current = strtok(cstr, " ");
            while (current != NULL && i < LAYER_WEIGHTS_SIZE) {
                weights_buffer[i] = model_default_t(atof(current));
                current = strtok(NULL, " ");
                i++;
            }
            if (i >= LAYER_WEIGHTS_SIZE) {
                break;
            }
        }
        std::cout << i << " / " << LAYER_WEIGHTS_SIZE << " weights loaded" << std::endl;

        while (std::getline(fin, iline) && std::getline(fpr, pline)) {
            if (e % CHECKPOINT == 0)
                std::cout << "Processing input " << e << std::endl;
            char *cstr = const_cast<char *>(iline.c_str());
            char *current;
            std::vector<float> in;
            current = strtok(cstr, " ");
            while (current != NULL) {
                in.push_back(atof(current));
                current = strtok(NULL, " ");
            }
            cstr = const_cast<char *>(pline.c_str());
            std::vector<float> pr;
            current = strtok(cstr, " ");
            while (current != NULL) {
                pr.push_back(atof(current));
                current = strtok(NULL, " ");
            }

            // hls-fpga-machine-learning insert data

            // hls-fpga-machine-learning insert top-level-function

            if (e % CHECKPOINT == 0) {
                std::cout << "Predictions" << std::endl;
                // hls-fpga-machine-learning insert predictions
                std::cout << "Quantized predictions" << std::endl;
                // hls-fpga-machine-learning insert quantized
            }
            e++;

            // hls-fpga-machine-learning insert tb-output
        }
        fin.close();
        fpr.close();
    } else {
        std::cout << "INFO: Unable to open input/predictions file, using default input." << std::endl;

        // hls-fpga-machine-learning insert zero

        // hls-fpga-machine-learning insert top-level-function

        // hls-fpga-machine-learning insert output

        // hls-fpga-machine-learning insert tb-output
    }

    fout.close();
    std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

    return 0;
}
