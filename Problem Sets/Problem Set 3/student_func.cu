/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

#define DEBUG 0

#if DEBUG == 1
#include <sys/time.h>
#endif

#include <cstdio>
#include <cassert>

template<typename T>
struct addF {
    __device__ T operator() (const T& a, const T& b) {
        return a + b;
    }
};

template<typename T>
struct minF {
    __device__ T operator() (const T& a, const T& b) {
        return (a < b) ? a : b;
    }
};

template<typename T>
struct maxF {
    __device__ T operator() (const T& a, const T& b) {
        return (a < b) ? b : a;
    }
};

template<typename T, typename F>
__device__ T reduce_shared(T* sdata, int n) {
    int tid = threadIdx.x;

    for (int s = n/2; s>0; s /=2) {
        if (tid < s) {
            T a = sdata[tid];
            T b = sdata[tid+s];
            sdata[tid] = F()(a, b);
        }
        __syncthreads();
    }
    
    T r;
    if (tid == 0) {
        r = sdata[0];
    } else {
        r = 0;
    }
    return r;
}

template<typename T, typename F>
__global__ void reduce_kernel(const T* const d_in, T* const d_out, T identity, int length) {

    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int tid = threadIdx.x;
    int n = blockDim.x;

    extern __shared__ __align__(sizeof(T)) unsigned char smem[];
    T *sdata = reinterpret_cast<T *>(smem); // size should be n

    if (idx < length) {
        sdata[tid] = d_in[idx];
    } else {
        sdata[tid] = identity;
    }
    if (idx + n < length) {
        sdata[tid] = F()(sdata[tid], d_in[idx + n]);
    }
    __syncthreads();

    T r = reduce_shared<T, F>(sdata, n);

    if (tid == 0) {
        d_out[blockIdx.x] = r;
    }
}

__global__ void reduce_min_max_kernel(const float* const d_in, float* const d_min, float* const d_max, int length) {
    
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int tid = threadIdx.x;
    int n = blockDim.x;

    extern __shared__ __align__(sizeof(float)) unsigned char smem[];
    float *sdata = reinterpret_cast<float *>(smem); // size should be n

    float* sdata_min = sdata;
    float* sdata_max = sdata + n;

    if (idx < length) {
        float t = d_in[idx];
        sdata_min[tid] = t;
        sdata_max[tid] = t;
    } else {
        sdata_min[tid] = 275.0;
        sdata_max[tid] = 0.0;
    }
    if (idx + n < length) {
        sdata_min[tid] = minF<int>()(sdata[tid], d_in[idx + n]);
        sdata_max[tid] = maxF<int>()(sdata[tid], d_in[idx + n]);
    }
    __syncthreads();

    for (int s = n/2; s>0; s /=2) {
        if (tid < s) {
            float a = sdata_min[tid];
            float b = sdata_min[tid+s];
            sdata_min[tid] = min(a, b);
            
            a = sdata_max[tid];
            b = sdata_max[tid+s];
            sdata_max[tid] = max(a, b);
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_min[blockIdx.x] = sdata_min[0];
        d_max[blockIdx.x] = sdata_max[0];
    }
}

__global__ void histogram_local_reduce_atomic(const float* const d_logLuminance,
                                                unsigned int* const d_histogram,
                                                const float min_logLum,
                                                const float max_logLum,
                                                const size_t length,
                                                const size_t numBins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    //int n = blockDim.x;
    float range_logLum = max_logLum - min_logLum;

    extern __shared__ __align__(sizeof(unsigned int)) unsigned char smem[];
    unsigned int *sdata = reinterpret_cast<unsigned int *>(smem); // size should be numBins

    for(int i=tid; i<numBins; i += blockDim.x) {
        sdata[i] = 0;
    }
    __syncthreads();

    for (int i=idx; i<length; i += gridDim.x * blockDim.x) {
        int bin = minF<unsigned int>()(static_cast<unsigned int>((d_logLuminance[i] - min_logLum) / range_logLum * numBins), static_cast<unsigned int>(numBins-1));
        //d_buf[bin * length + idx] += 1;
        atomicAdd(&sdata[bin], 1);
    }

    __syncthreads();
    for(int i=tid; i<numBins; i += blockDim.x) {
        atomicAdd(&d_histogram[i], sdata[i]);
    }
    //for (int i=0; i<numBins; ++i) {
    //    __syncthreads();
    //    sdata[tid] = (idx < length)? d_buf[i * length + idx] : 0;
    //    __syncthreads();
    //    unsigned int r = reduce_shared< unsigned int, addF<unsigned int> >(sdata, n);
    //    if (tid == 0) atomicAdd(&d_histogram[i], r);
    //}
}

template<typename T, typename F>
__global__ void hills_steele_scan_kernel(const T* const d_in, T* const d_out, int length) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    int n = blockDim.x; // size of one shared mem array

    extern __shared__ __align__(sizeof(T)) unsigned char smem[];
    T *sdata = reinterpret_cast<T *>(smem); // size should be n

    if (idx < length) {
        sdata[tid] = d_in[idx];
    }
    __syncthreads();

    for (int s = 1, pin = 0, pout = 1; s < blockDim.x; s <<= 1, pin = 1-pin, pout = 1-pout) {
        if (tid >= s) {
            T a = sdata[pin*n + tid - s];
            T b = sdata[pin*n + tid];
            sdata[pout*n + tid] = F()(a, b);
        } else {
            sdata[pout*n + tid] = sdata[pin*n + tid];
        }
        __syncthreads();
    }

    if (idx < length) {
        d_out[idx] = sdata[tid];
    }
}

template<typename T, typename F>
__global__ void blelloch_scan_kernel(T* const d_in, T* const d_out, int length, T identity) {
    
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int ltid = threadIdx.x;

    int n = 2*blockDim.x;
    
    extern __shared__ __align__(sizeof(T)) unsigned char smem[];
    T *sdata = reinterpret_cast<T *>(smem); // size should be n
    
    sdata[ltid * 2] = (gtid * 2 < length)? d_in[gtid * 2] : identity;
    sdata[ltid * 2 + 1] = (gtid * 2 + 1< length)? d_in[gtid * 2 + 1] : identity;
    __syncthreads();

    int s = 2;
    for (; s <= n; s <<= 1) {
        int pos = ltid * s + s - 1;
        if (pos < n) {
            T a = sdata[pos - s/2];
            T b = sdata[pos];
            sdata[pos] = F()(a, b);
        }
        __syncthreads();
    }

    if (ltid == 0) {
        if(d_out) d_out[blockIdx.x] = sdata[n-1];
        sdata[n - 1] = identity; // identity item
    }
    __syncthreads();
    
    s = n;
    for (; s > 1; s >>= 1) {
        int pos = ltid * s + s - 1;
        if (pos < n) {
            T a = sdata[pos - s/2];
            T b = sdata[pos];
            sdata[pos] = F()(a, b);
            sdata[pos - s/2] = b;
        }
        __syncthreads();
    }
    
    if (gtid * 2 < length) {
        d_in[gtid * 2] = sdata[ltid * 2];
    }
    if (gtid * 2 + 1 < length) {
        d_in[gtid * 2 + 1] = sdata[ltid * 2 + 1];
    }
}

template<typename T, typename F>
__global__ void scan_update_kernel(T* const d_in, T* const d_buf, int length) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int idx = bid * blockDim.x + tid;

    __shared__ T sdata;

    if (tid == 0) {
        sdata = d_buf[bid];
    }
    __syncthreads();

    if (idx < length) {
        d_in[idx] =  F()(d_in[idx], sdata);
    }
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

    int length = numRows * numCols;

    int blockSize = 512;
    int gridSize = (length + blockSize - 1)/blockSize;
    
    float* d_buf1;
    unsigned int* d_buf2;

#if DEBUG == 1
    struct timeval tb, te;
    gettimeofday(&tb, 0);
#endif    
    checkCudaErrors(cudaMalloc((void **)&d_buf1, length * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_buf2, numBins * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_buf2, 0, numBins * sizeof(unsigned int)));
   
#if DEBUG == 1
    cudaDeviceSynchronize();
    gettimeofday(&te, 0);
    printf("memset time: %ld\n", (te.tv_sec-tb.tv_sec)*1000000 + te.tv_usec-tb.tv_usec);
#endif    

    // find min_logLum
#if DEBUG == 1
    printf("reduce_min: grid: %d, block: %d, length: %d\n", gridSize, blockSize, length);
    gettimeofday(&tb, 0);
#endif    

    reduce_kernel< float, minF<float> ><<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_logLuminance, d_buf1, 275.0, length);
    checkCudaErrors(cudaGetLastError());

    int gs = gridSize;
    while(gs > 1) {
        int l = gs;
        gs = (gs+blockSize-1)/blockSize;
        printf("reduce_min: grid: %d, block: %d, length: %d\n", gs, blockSize, l);
        reduce_kernel<float, minF<float> ><<<gs, blockSize, blockSize * sizeof(float)>>>(d_buf1, d_buf1, 275.0, l);
        checkCudaErrors(cudaGetLastError());
    }
#if DEBUG == 1
    cudaDeviceSynchronize();
    gettimeofday(&te, 0);
    printf("reduce min time: %ld\n", (te.tv_sec-tb.tv_sec)*1000000 + te.tv_usec-tb.tv_usec);
#endif    

    checkCudaErrors(cudaMemcpy(&min_logLum, d_buf1, sizeof(float), cudaMemcpyDeviceToHost));
    printf("min_logLum = %f\n", min_logLum);
    
    // find max logLum
#if DEBUG == 1
    printf("reduce_max: grid: %d, block: %d, length: %d\n", gridSize, blockSize, length);
    gettimeofday(&tb, 0);
#endif    

    reduce_kernel<float, maxF<float> ><<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_logLuminance, d_buf1, 0.0, length);
    checkCudaErrors(cudaGetLastError());

    gs = gridSize;
    while(gs > 1) {
        int l = gs;
        gs = (gs+blockSize-1)/blockSize;
        printf("reduce_max: grid: %d, block: %d, length: %d\n", gs, blockSize, l);
        reduce_kernel<float, maxF<float> ><<<gs, blockSize, blockSize * sizeof(float)>>>(d_buf1, d_buf1, 0.0, l);
        checkCudaErrors(cudaGetLastError());
    }

#if DEBUG == 1
    cudaDeviceSynchronize();
    gettimeofday(&te, 0);
    printf("reduce max time: %ld\n", (te.tv_sec-tb.tv_sec)*1000000 + te.tv_usec-tb.tv_usec);
#endif    

    checkCudaErrors(cudaMemcpy(&max_logLum, d_buf1, sizeof(float), cudaMemcpyDeviceToHost));
    printf("max_logLum = %f\n", max_logLum);

    // compute histogram
    gridSize = (length + blockSize - 1)/blockSize;

#if DEBUG == 1
    printf("histogram: grid: %d, block: %d, length: %d\n", gridSize, blockSize, length);
    gettimeofday(&tb, 0);
#endif    

    histogram_local_reduce_atomic<<<gridSize, blockSize, numBins * sizeof(unsigned int)>>>(d_logLuminance, d_cdf, min_logLum, max_logLum, length, numBins);
    checkCudaErrors(cudaGetLastError());
    
#if DEBUG == 1
    cudaDeviceSynchronize();
    gettimeofday(&te, 0);
    printf("histogram time: %ld\n", (te.tv_sec-tb.tv_sec)*1000000 + te.tv_usec-tb.tv_usec);
#endif    
   
    // exclusive scan to produce cdf
    blockSize = 1024;
    gridSize = (numBins + blockSize - 1)/blockSize;
    assert(gridSize <= blockSize);

#if DEBUG == 1
    printf("scan: grid: %d, block: %d, length: %lu\n", gridSize, blockSize, numBins);
    gettimeofday(&tb, 0);
#endif    

    blelloch_scan_kernel<unsigned int, addF<unsigned int> ><<<gridSize, blockSize, blockSize * 2 * sizeof(unsigned int)>>>(d_cdf, d_buf2, numBins, 0);
    checkCudaErrors(cudaGetLastError());
    
    if (gridSize > 1) {
#if DEBUG == 1
        printf("scan: grid: %d, block: %d, length: %d\n", (gridSize+blockSize-1)/blockSize, blockSize, gridSize);
#endif    
        blelloch_scan_kernel<unsigned int, addF<unsigned int> ><<<(gridSize+blockSize-1)/blockSize, blockSize, blockSize * 2 * sizeof(unsigned int)>>>(d_buf2, NULL, gridSize, 0);
        checkCudaErrors(cudaGetLastError());
#if DEBUG == 1
        printf("scan_update: grid: %d, block: %d\n", gridSize, blockSize);
#endif    
        scan_update_kernel<unsigned int, addF<unsigned int> ><<<gridSize, blockSize>>>(d_cdf, d_buf2, numBins);
        checkCudaErrors(cudaGetLastError());
    }
    
#if DEBUG == 1
    cudaDeviceSynchronize();
    gettimeofday(&te, 0);
    printf("scan time: %ld\n", (te.tv_sec-tb.tv_sec)*1000000 + te.tv_usec-tb.tv_usec);
#endif    
    
    checkCudaErrors(cudaFree(d_buf1));
    checkCudaErrors(cudaFree(d_buf2));
}
