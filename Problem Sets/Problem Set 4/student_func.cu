//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

#include <cassert>
#include <cstdio>

#define DEBUG 0
#if DEBUG == 1
#include <sys/time.h>
#endif

template<typename T>
__device__ T blelloch_scan_shared(T* const sdata, size_t n, T identity) { // n must be power of 2
    
    T r;
    int tid = threadIdx.x;
    int s = 2;
    for (; s <= n; s <<= 1) {
        int pos = tid * s + s - 1;
        if (pos < n) {
            T a = sdata[pos - s/2];
            T b = sdata[pos];
            sdata[pos] = a + b;
        }
        __syncthreads();
    }

    if (tid == 0) {
        r = sdata[n - 1];
        sdata[n - 1] = identity; // identity item
    }
    __syncthreads();
    
    s = n;
    for (; s > 1; s >>= 1) {
        int pos = tid * s + s - 1;
        if (pos < n) {
            T a = sdata[pos - s/2];
            T b = sdata[pos];
            sdata[pos] = a + b;
            sdata[pos - s/2] = b;
        }
        __syncthreads();
    }

    return r;
}

template<typename T>
__global__ void one_warp_exlusive_scan_kernel(T* const d_in, size_t n, T identity) {
    
    extern __shared__ T s_data[];

    int tid = threadIdx.x;
    s_data[tid] = (tid < n)? d_in[tid] : identity;
    
    // per_warp scan, no need to sync
    // __syncthreads();
    
    blelloch_scan_shared<T>(s_data, n, identity);
    
    if (tid < n) d_in[tid] = s_data[tid];
}

template<size_t NB>
__global__ void per_block_predicate_scan_kernel(unsigned int* const d_inputVals,
                                                unsigned int* const d_predicate,
                                                unsigned int* const d_buf, // size = gridDim.x * (1<<NB)
                                                const size_t numElems, const int shift) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ unsigned int sdata[]; // size = blockDim.x * (1<<NB)

    // 1) compute per thread histogram, predicate
    for(size_t i=0; i<(1<<NB); ++i) {
        sdata[i*blockDim.x + tid] = 0;
    }
    unsigned int v, b;
    if (idx < numElems) {
        v = d_inputVals[idx];
        b = (v >> shift) & ((1<<NB) - 1);
        sdata[b*blockDim.x + tid] = 1; // this is predicate array
    }
    __syncthreads();

    // 2) scan the predicate array, compute global histogram
    for(size_t i=0; i<(1<<NB); ++i) {
        unsigned int r = blelloch_scan_shared< unsigned int >(&sdata[i*blockDim.x], blockDim.x, 0);
        if (tid == 0) d_buf[i * gridDim.x + blockIdx.x] = r;
    }

    // 3) write out the scanned predicate array
    if (idx < numElems) d_predicate[idx] = sdata[b * blockDim.x + tid];
}

template<size_t NB>
__global__ void second_level_predicate_scan_kernel(unsigned int* const d_buf, // size = gridDim.x * (1<<NB)
                                                    unsigned int* const d_histo, 
                                                    const size_t numElems) {
    int tid = threadIdx.x;
    extern __shared__ unsigned int sdata[]; // size = blockDim.x 
   
    for(size_t i=0; i<(1<<NB); ++i) {
        // 1) load the intermediate data to shared mem
        sdata[tid] = (tid < numElems)? d_buf[i*numElems + tid] : 0;
        __syncthreads();
        // 2) scan
        unsigned int r = blelloch_scan_shared< unsigned int >(sdata, blockDim.x, 0);
        if (tid < numElems) d_buf[i*numElems+ tid] = sdata[tid];
        __syncthreads();
        if (tid == 0) d_histo[i] = r;
    }
}

template<size_t NB>
__global__ void fixup_and_scatter(unsigned int* const d_inputVals,
                                  unsigned int* const d_outputVals,
                                  unsigned int* const d_inputPos,
                                  unsigned int* const d_outputPos,
                                  unsigned int* const d_predicate,
                                  unsigned int* const d_buf, // size = gridDim.x * (1<<NB)
                                  unsigned int* const d_cdf, 
                                  const size_t numElems, const int shift) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ unsigned int sdata[]; // size = blockDim.x 
    
    if (idx >= numElems) {
        return;
    }

    unsigned int v = d_inputVals[idx];
    unsigned int b = (v >> shift) & ((1<<NB) - 1);

    // FIXME: d_buf in shared mem?
    unsigned int r = d_buf[b * gridDim.x + blockIdx.x];
    unsigned int p = d_predicate[idx] + r + d_cdf[b];

    d_outputVals[p] = v;
    d_outputPos[p] = d_inputPos[idx];
}

#define NBITS 2
#define NCOMB (1<<NBITS)

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
  //PUT YOUR SORT HERE
    int blockSize = 256;
    int gridSize = (numElems + blockSize - 1) / blockSize;
    unsigned int* d_predicate;
    unsigned int* d_buf;
    unsigned int* d_histo;

    unsigned int* d_iov[2] = {d_inputVals, d_outputVals};
    unsigned int* d_iop[2] = {d_inputPos, d_outputPos};

#if DEBUG == 1
    printf("numElems: %u\n", numElems);
    struct timeval tb, te;
    gettimeofday(&tb, 0);
#endif    
    assert(gridSize <= 1024);
    assert(NCOMB <= 32);

    checkCudaErrors(cudaMalloc((void **)&d_predicate, numElems * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void **)&d_buf, gridSize * NCOMB * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void **)&d_histo, NCOMB * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_histo, 0, NCOMB * sizeof(unsigned int)));
#if DEBUG == 1
    checkCudaErrors(cudaDeviceSynchronize());
    gettimeofday(&te, 0);
    printf("memset time: %ld\n", (te.tv_sec-tb.tv_sec)*1000000 + te.tv_usec-tb.tv_usec);
#endif    

    for (int shift=0, pi=0, po=1; shift < 8*sizeof(unsigned int); shift+=NBITS, pi=1-pi, po=1-po) {
#if DEBUG == 1
        printf("@@@shift: %d\n", shift);
        printf("per_block_scan: grid: %d, block: %d, length: %d\n", gridSize, blockSize, numElems);
        gettimeofday(&tb, 0);
#endif    
        per_block_predicate_scan_kernel<NBITS><<<gridSize, blockSize, NCOMB*blockSize*sizeof(unsigned int)>>>(d_iov[pi], d_predicate, d_buf, numElems, shift);
#if DEBUG == 1
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&te, 0);
        printf("per_block_scan time: %ld\n", (te.tv_sec-tb.tv_sec)*1000000 + te.tv_usec-tb.tv_usec);
#endif    

#if DEBUG == 1
        printf("second_level_scan: grid: %d, block: %d, length: %d\n", 1, 1024, gridSize);
        gettimeofday(&tb, 0);
#endif    
        second_level_predicate_scan_kernel<NBITS><<<1, 1024, 1024*sizeof(unsigned int)>>>(d_buf, d_histo, gridSize);
#if DEBUG == 1
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&te, 0);
        printf("second_level_scan time: %ld\n", (te.tv_sec-tb.tv_sec)*1000000 + te.tv_usec-tb.tv_usec);
        unsigned int* h_histo = (unsigned int*)malloc(NCOMB*sizeof(unsigned int));
        checkCudaErrors(cudaMemcpy(h_histo, d_histo, NCOMB*sizeof(unsigned int), cudaMemcpyDeviceToHost));
        printf("histo:");
        for(int i=0; i<NCOMB; ++i) {
            printf("%d, ", h_histo[i]);
        }
        printf("\n");
#endif    

#if DEBUG == 1
        printf("one_warp_scan: grid: %d, block: %d, length: %d\n", 1, 32, NCOMB);
        gettimeofday(&tb, 0);
#endif    
        one_warp_exlusive_scan_kernel<unsigned int><<<1, 32, 32*sizeof(unsigned int)>>>(d_histo, NCOMB, 0);
#if DEBUG == 1
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&te, 0);
        printf("one_warp_scan time: %ld\n", (te.tv_sec-tb.tv_sec)*1000000 + te.tv_usec-tb.tv_usec);
        checkCudaErrors(cudaMemcpy(h_histo, d_histo, NCOMB*sizeof(unsigned int), cudaMemcpyDeviceToHost));
        printf("cdf:");
        for(int i=0; i<NCOMB; ++i) {
            printf("%d, ", h_histo[i]);
        }
        printf("\n");
#endif    

#if DEBUG == 1
        printf("fixup_and_scatter: grid: %d, block: %d, length: %d\n", gridSize, blockSize, numElems);
        gettimeofday(&tb, 0);
#endif    
        fixup_and_scatter<NBITS><<<gridSize, blockSize, blockSize*sizeof(unsigned int)>>>(d_iov[pi], d_iov[po], d_iop[pi], d_iop[po], d_predicate, d_buf, d_histo, numElems, shift);
#if DEBUG == 1
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&te, 0);
        printf("fixup_and_scatter time: %ld\n", (te.tv_sec-tb.tv_sec)*1000000 + te.tv_usec-tb.tv_usec);
#endif    
    }

    if ((sizeof(unsigned int)*8/NBITS) != 0) {
        checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    }
    
    checkCudaErrors(cudaFree(d_predicate));
    checkCudaErrors(cudaFree(d_buf));
    checkCudaErrors(cudaFree(d_histo));
}
