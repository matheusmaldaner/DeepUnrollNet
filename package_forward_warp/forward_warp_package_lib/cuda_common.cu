#include "cuda_common.h"
#include <cuda_runtime_api.h>
#include <stdio.h>

// Modern texture objects (replacing deprecated texture references)
cudaTextureObject_t texObj_u1 = 0;
cudaTextureObject_t texObj_f1 = 0;
cudaTextureObject_t texObj_i2 = 0;
cudaTextureObject_t texObj_i4 = 0;
cudaTextureObject_t texObj_u2 = 0;
cudaTextureObject_t texObj_f2 = 0;
cudaTextureObject_t texObj_f4 = 0;
cudaTextureObject_t texObj_u4 = 0;

inline int getNumTiles(int totalSize, int tileSize)
{
    const int div = totalSize / tileSize;
    return totalSize % tileSize == 0 ? div : div + 1;
}

template<class T>
__global__ void kernel_texture_to_memory(T* output, tex_type type, int nOutputChans, int H, int W, 
                                        cudaTextureObject_t texObj_u1, cudaTextureObject_t texObj_u2, 
                                        cudaTextureObject_t texObj_u4, cudaTextureObject_t texObj_i2, 
                                        cudaTextureObject_t texObj_i4, cudaTextureObject_t texObj_f1, 
                                        cudaTextureObject_t texObj_f2, cudaTextureObject_t texObj_f4) {
    const int X = blockIdx.x * blockDim.x + threadIdx.x;
    const int Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (!(X < W && Y < H)) return;

    int index = (Y * W + X) * nOutputChans;

    switch (type) {
        case TEX_U1: {
            output[index] = tex2D<unsigned char>(texObj_u1, X, Y);
            break;
        }
        case TEX_U2: {
            uchar2 data = tex2D<uchar2>(texObj_u2, X, Y);
            const unsigned char data_array[2] = {data.x, data.y};
            for (int i = 0; i < nOutputChans; i++) {
                output[index + i] = data_array[i];
            }
            break;
        }
        case TEX_U4: {
            uchar4 data = tex2D<uchar4>(texObj_u4, X, Y);
            const unsigned char data_array[4] = {data.x, data.y, data.z, data.w};
            for (int i = 0; i < nOutputChans; i++) {
                output[index + i] = data_array[i];
            }
            break;
        }
        case TEX_I2: {
            int2 data = tex2D<int2>(texObj_i2, X, Y);
            const int data_array[2] = {data.x, data.y};
            for (int i = 0; i < nOutputChans; i++) {
                output[index + i] = data_array[i];
            }
            break;
        }
        case TEX_I4: {
            int4 data = tex2D<int4>(texObj_i4, X, Y);
            const int data_array[4] = {data.x, data.y, data.z, data.w};
            for (int i = 0; i < nOutputChans; i++) {
                output[index + i] = data_array[i];
            }
            break;
        }
        case TEX_F1: {
            float data = tex2D<float>(texObj_f1, X, Y);
            output[index] = data;
            break;
        }
        case TEX_F2: {
            float2 data = tex2D<float2>(texObj_f2, X, Y);
            float data_array[2] = {data.x, data.y};
            for (int i = 0; i < nOutputChans; i++) {
                output[index + i] = data_array[i];
            }
            break;
        }
        case TEX_F4: {
            float4 data = tex2D<float4>(texObj_f4, X, Y);
            float data_array[4] = {data.x, data.y, data.z, data.w};
            for (int i = 0; i < nOutputChans; i++) {
                output[index + i] = data_array[i];
            }
            break;
        }
    }
}

template<class T>
__global__ void from_pytorch_mem_layout_kernel(int B, int C, int H, int W, const T* src_data_ptr, T* dst_data_ptr)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= B*C*H*W) return;

    int bn = i / (C*H*W); int remaining = i % (C*H*W);
    int ch = remaining / (H*W); remaining = remaining % (H*W);
    int r = remaining / W; remaining = remaining % W;
    int c = remaining;
    dst_data_ptr[bn * (H*W*C) + r * W * C + c * C + ch] = src_data_ptr[i];
}

template<class T>
__global__ void to_pytorch_mem_layout_kernel(int B, int C, int H, int W, const T* src_data_ptr, T* dst_data_ptr)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= B*C*H*W) return;
    int bn = i / (C*H*W); int remaining = i % (C*H*W);
    int r = remaining / (W*C); remaining = remaining % (W*C);
    int c = remaining / C;
    int ch = remaining % C;
    dst_data_ptr[bn*H*W*C + ch*H*W + r * W + c] = src_data_ptr[i];
}


template<class T>
void cudaTextureToCudaMem(const cudaArray* input, T* output, tex_type type, int nOutputChans, int H, int W)
{
    // Create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = const_cast<cudaArray*>(input);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    // Create all texture objects for kernel parameters
    cudaTextureObject_t texObjs[8] = {0};
    texObjs[0] = texObj; // for the current type
    
    dim3 gridDim = dim3(getNumTiles(W, 32), getNumTiles(H, 32));
    dim3 blockDim = dim3(32, 32);
    kernel_texture_to_memory<T><<<gridDim, blockDim>>>(output, type, nOutputChans, H, W, 
                                                        texObjs[0], texObjs[1], texObjs[2], texObjs[3],
                                                        texObjs[4], texObjs[5], texObjs[6], texObjs[7]);
    cudaDeviceSynchronize();

    // Cleanup
    cudaDestroyTextureObject(texObj);
}


template<class T>
void from_pytorch_mem_layout(int B, int C, int H, int W, const T* src_data_ptr, T* dst_data_ptr)
{
    const int threads = 512;
    const dim3 blocks ((B*C*H*W) / threads + 1);
    //printf("kernel BCHW %d %d %d %d\n", B, C, H, W);
    from_pytorch_mem_layout_kernel<<<blocks, threads>>>(B, C, H, W, src_data_ptr, dst_data_ptr);
    cudaDeviceSynchronize();
}

template<class T>
void to_pytorch_mem_layout(int B, int C, int H, int W, const T* src_data_ptr, T* dst_data_ptr)
{
    const int threads = 512;
    const dim3 blocks ((B*C*H*W) / threads + 1);
    to_pytorch_mem_layout_kernel<<<blocks, threads>>>(B, C, H, W, src_data_ptr, dst_data_ptr);
    cudaDeviceSynchronize();
}

/** Instatiate template functions **/
template void cudaTextureToCudaMem<unsigned char>(const cudaArray* input, unsigned char* output, tex_type type, int nOutputChans, int H, int W);
template void cudaTextureToCudaMem<int>(const cudaArray* input, int* output, tex_type type, int nOutputChans, int H, int W);
template void cudaTextureToCudaMem<float>(const cudaArray* input, float* output, tex_type type, int nOutputChans, int H, int W);

template void from_pytorch_mem_layout<int>(int B, int C, int H, int W, const int* src_data_ptr, int* dst_data_ptr);
template void from_pytorch_mem_layout<float>(int B, int C, int H, int W, const float* src_data_ptr, float* dst_data_ptr);
template void to_pytorch_mem_layout<float>(int B, int C, int H, int W, const float* src_data_ptr, float* dst_data_ptr);
template void to_pytorch_mem_layout<unsigned char>(int B, int C, int H, int W, const unsigned char* src_data_ptr, unsigned char* dst_data_ptr);