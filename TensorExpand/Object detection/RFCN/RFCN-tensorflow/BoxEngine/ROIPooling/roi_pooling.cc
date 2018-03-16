// Copyright 2017 Robert Csordas. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include <math.h>
#include <algorithm>
#include <iostream>
#include "roi_pooling_gpu.h"

using namespace tensorflow;

void PosRoiPoolingCPU(float *output, const float *input, const int *boxes, const int nBoxes, const int *roi_size, const int outChannels, const int imageWidth, const int imageHeight){
    for (int chan=0; chan<outChannels; ++chan){
        for (int box=0; box<nBoxes; ++box){
            for (int boxY=0; boxY<roi_size[0]; ++boxY){
                for (int boxX=0; boxX<roi_size[1]; ++boxX){
                    int x0 = std::min(std::max(boxes[box*4+0],0), imageWidth-1);
                    int y0 = std::min(std::max(boxes[box*4+1],0), imageHeight-1);
                    int x1 = std::min(std::max(boxes[box*4+2],0), imageWidth-1);
                    int y1 = std::min(std::max(boxes[box*4+3],0), imageHeight-1);
                    float wBox = float(x1 - x0 + 1) / float(roi_size[1]);
                    float hBox = float(y1 - y0 + 1) / float(roi_size[0]);

                    x0 = int(float(x0) + wBox * float(boxX));
                    y0 = int(float(y0) + hBox * float(boxY));

                    int wBox_int = (int)ceil(wBox);
                    int hBox_int = (int)ceil(hBox);

                    int boxIndex = boxX + boxY * roi_size[1];

                    const float *inPlane = input + imageWidth*imageHeight*(boxIndex*outChannels + chan);

                    float acc = 0;
                    for (int y=0; y<hBox_int; ++y){
                        const float *rStart=inPlane + (y+y0)*imageWidth + x0;
                        for (int x=0; x<wBox_int; ++x){
                            acc += rStart[x];
                        }
                    }

                    acc /= hBox_int*wBox_int;
                    output[(box*roi_size[0]*roi_size[1] + boxIndex)*outChannels + chan] = acc;
                }
            }
        }
    }
}


void ComputePosRoiPooling(OpKernelContext* context, bool useGPU) {
    //Inputs
    const Tensor& input_tensor = context->input(0);
    const Tensor& boxes = context->input(1);

    const Tensor& roi_size_tensor = context->input(2);
    int * roi_size = (int*)roi_size_tensor.tensor_data().data();

    //Input check
    OP_REQUIRES(context, input_tensor.dims() == 4, errors::InvalidArgument("Features should be a 4d input tensor"));
    OP_REQUIRES(context, boxes.dims() == 2, errors::InvalidArgument("Boxes sould be a 2d tensor"));
    OP_REQUIRES(context, boxes.dim_size(1) == 4, errors::InvalidArgument("Box should have 4 coordinates"));
    OP_REQUIRES(context, roi_size_tensor.dims() == 1, errors::InvalidArgument("Output size should be 1d"));
    OP_REQUIRES(context, roi_size_tensor.dim_size(0) == 2, errors::InvalidArgument("Output size should be 2"));
    OP_REQUIRES(context, input_tensor.dim_size(0) == 1, errors::InvalidArgument("Batches not supported"));

    //Calculate dimensions
    int num_channels = input_tensor.dim_size(1);
    int image_width = input_tensor.dim_size(3);
    int image_height = input_tensor.dim_size(2);

    int n_boxes = boxes.dim_size(0);
    int box_h = roi_size[0];
    int box_w = roi_size[1];
    int n_cells = box_h*box_w;

    OP_REQUIRES(context, num_channels % n_cells == 0, errors::InvalidArgument("Number of input channels should be divisable by output window size"));

    int n_output_channels = num_channels/n_cells;

    //Allocate output
    Tensor* output_tensor = NULL;
    TensorShape output_shape = {n_boxes, box_h, box_w, n_output_channels};
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

    //Raw data
    const float *input = (float*) input_tensor.tensor_data().data();
    float *output = (float*) output_tensor->tensor_data().data();
    const int *raw_boxes = (int*) boxes.tensor_data().data();

    if (useGPU){
        #if USE_GPU
        PosRoiPoolingGPUKernelLauncher(output, input, raw_boxes, n_boxes, roi_size, n_output_channels, image_width, image_height);
        #endif
    } else {
        PosRoiPoolingCPU(output, input, raw_boxes, n_boxes, roi_size, n_output_channels, image_width, image_height);
    }
}


class PosRoiPoolingCpu : public OpKernel {
    public:
    explicit PosRoiPoolingCpu(OpKernelConstruction* context) : OpKernel(context) { }

    void Compute(OpKernelContext* context) override {
        // Compute using CPU
        ComputePosRoiPooling(context, false);
    }
};

#if USE_GPU
class PosRoiPoolingGpu : public OpKernel {
    public:
    explicit PosRoiPoolingGpu(OpKernelConstruction* context) : OpKernel(context) { }

    void Compute(OpKernelContext* context) override {
        // Compute using GPU
        ComputePosRoiPooling(context, true);
    }
};
#endif

REGISTER_OP("PosRoiPooling")
    .Input("conv_features: float")
    .Input("boxes: int32")
    .Input("roi_size: int32")
    .Output("output_tensor: float");

REGISTER_KERNEL_BUILDER(Name("PosRoiPooling").Device(DEVICE_CPU), PosRoiPoolingCpu);
#if USE_GPU
REGISTER_KERNEL_BUILDER(Name("PosRoiPooling").Device(DEVICE_GPU).HostMemory("roi_size"), PosRoiPoolingGpu);
#endif