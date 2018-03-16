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

#include "roi_pooling_gpu.h"

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

__global__ void PosRoiPoolingGPUKernel(float *output, const float *input, const int *boxes, const int nBoxes, const int *roi_size, const int outChannels, const int imageWidth, const int imageHeight){
    int N = outChannels*nBoxes*roi_size[0]*roi_size[1];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        int j = i;
        int boxX = j % roi_size[1];
        j /= roi_size[1];
        int boxY = j % roi_size[0];
        j /= roi_size[0];
        int box = j % nBoxes;
        j /= nBoxes;
        int chan = j;

    
        int x0 = min(max(boxes[box*4+0],0), imageWidth-1);
        int y0 = min(max(boxes[box*4+1],0), imageHeight-1);
        int x1 = min(max(boxes[box*4+2],0), imageWidth-1);
        int y1 = min(max(boxes[box*4+3],0), imageHeight-1);
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

void PosRoiPoolingGPUKernelLauncher(float *output, const float *input, const int *boxes, const int nBoxes, const int *roi_size, const int outChannels, const int imageWidth, const int imageHeight){
    PosRoiPoolingGPUKernel<<<32, 256>>>(output, input, boxes, nBoxes, roi_size, outChannels, imageWidth, imageHeight);
}

#endif
