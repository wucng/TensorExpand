#ifndef _ROI_POOLING_GPU_H_
#define _ROI_POOLING_GPU_H_

void PosRoiPoolingGPUKernelLauncher(float *output, const float *input, const int *boxes, const int nBoxes, const int *roi_size, const int outChannels, const int imageWidth, const int imageHeight);
void PosRoiPoolingGradGPUKernelLauncher(float *input, const float *grad, const int *boxes, const int nBoxes, const int *roi_size, const int outChannels, const int imageWidth, const int imageHeight);

#endif
