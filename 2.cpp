#include <cluda.h>
// modified from pytorch
// https://github.com/pytorch/pytorch/master/blob/torch/lib/THC/THCTensorTopK.cuh
// original license below:
/*
Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/


#if __CUDA_ARCH__ < 350
#define __ldg(ptr) (*(ptr))
#endif


typedef ptrdiff_t ssize_t;


__device__ __forceinline__ int lane_id() {
  int id;
  asm("mov.s32 %0, %laneid;" : "=r"(id) );
  return id;
}

__device__ __forceinline__ unsigned lane_mask_lt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__ unsigned lane_mask_le() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__ unsigned lane_mask_gt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__ unsigned lane_mask_ge() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_ge;" : "=r"(mask));
  return mask;
}

template <typename T>
struct Bitfield {};

template <>
struct Bitfield<unsigned int> {
  static __device__ __forceinline__
  unsigned int get(unsigned int val, int pos, int len) {
    unsigned int ret;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
    return ret;
  }

  static __device__ __forceinline__
  unsigned int set(unsigned int val, unsigned int toInsert, int pos, int len) {
    unsigned int ret;
    asm("bfi.b32 %0, %1, %2, %3, %4;" :
        "=r"(ret) : "r"(toInsert), "r"(val), "r"(pos), "r"(len));
    return ret;
  }
};

template <>
struct Bitfield<unsigned long long int> {
  static __device__ __forceinline__
  unsigned long long int get(unsigned long long int val, int pos, int len) {
    unsigned long long int ret;
    asm("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) : "l"(val), "r"(pos), "r"(len));
    return ret;
  }

  static __device__ __forceinline__
  unsigned long long int set(unsigned long long int val, unsigned long long int toInsert, int pos, int len) {
    unsigned long long int ret;
    asm("bfi.b64 %0, %1, %2, %3, %4;" :
        "=l"(ret) : "l"(toInsert), "l"(val), "r"(pos), "r"(len));
    return ret;
  }
};


template <typename T>
struct RadixConfig {
// Converts a type (maybe float) to an integer representation with the same
// sorting; i.e., for floats f1, f2:
// if f1 < f2 then convert(f1) < convert(f2)
// We use this to enable radix selection of floating-point values.
// This also gives a relative order for NaNs, but that's ok, as they
// will all be adjacent
  typedef unsigned int RadixType;
  static inline __device__ RadixType convert(T v) {
      return (RadixType)v;
  }

  static inline __device__ float deconvert(RadixType v) {
      return (T)v;
  }
};

template <>
struct RadixConfig<float> {
  typedef unsigned int RadixType;

  static inline __device__ RadixType convert(float v) {
    RadixType x = __float_as_int(v);
    RadixType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;

    return (x ^ mask);
  }

  static inline __device__ float deconvert(RadixType v) {
    RadixType mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;

    return __int_as_float(v ^ mask);
  }
};

/* Inspired by specialization for float, above.
 * NB: This specialization needs that ga_half is defined.
 * So, this code should be included only after #include <cluda.h> */
template <>
struct RadixConfig<ga_half> {
  typedef unsigned int RadixType;

  static inline __device__ RadixType convert(ga_half v) {
    RadixType x = __float_as_int(ga_half2float(v));
    RadixType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;

    return (x ^ mask);
  }

  static inline __device__ ga_half deconvert(RadixType v) {
    RadixType mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;

    return ga_float2half(__int_as_float(v ^ mask));
  }
};

template <>
struct RadixConfig<double> {
  typedef unsigned long long RadixType;

  static inline __device__ RadixType convert(double v) {
    RadixType x = __double_as_longlong(v);
    RadixType mask = -((x >> 63)) | 0x8000000000000000;
    return (x ^ mask);
  }

  static inline __device__ double deconvert(RadixType v) {
    RadixType mask = ((v >> 63) - 1) | 0x8000000000000000;
    return __longlong_as_double(v ^ mask);
  }
};


template <>
struct RadixConfig<char> {
  typedef unsigned int RadixType;

  static inline __device__ RadixType convert(char v) {
    return 128u + v;
  }

  static inline __device__ char deconvert(RadixType v) {
    return v - 128;
  }
};

template <>
struct RadixConfig<short> {
  typedef unsigned int RadixType;

  static inline __device__ RadixType convert(short v) {
    assert(sizeof(short) == 2);
    return 32768u ^ v;
  }

  static inline __device__ short deconvert(RadixType v) {
    return v - 32768;
  }
};

template <>
struct RadixConfig<int> {
  typedef unsigned int RadixType;

  static inline __device__ RadixType convert(int v) {
    assert(sizeof(int) == 4);
    return 2147483648u + v;
  }

  static inline __device__ int deconvert(RadixType v) {
    return v - 2147483648u;
  }
};

template <>
struct RadixConfig<long long> {
  typedef unsigned long long RadixType;

  static inline __device__ RadixType convert(long long v) {
    assert(sizeof(long long) == 8);
    return 9223372036854775808ull + v;
  }

  static inline __device__ long long deconvert(RadixType v) {
    return v - 9223372036854775808ull;
  }
};

#define USE_HALF 1

#if USE_HALF == 1
// since half is ushort, using macro to protect this part is necessary
template <>
struct RadixConfig<unsigned short> {
  typedef unsigned int RadixType;

  static inline __device__ RadixType convert(unsigned short v) {
    RadixType mask = -(((RadixType)v >> 15)) | 0x8000;
    return (v ^ mask);
  }

  static inline __device__ unsigned short deconvert(RadixType v) {
    RadixType mask = ((v >> 15) - 1) | 0x8000;
    return (unsigned short)(v ^ mask);
  }
};
#endif // USE_HALF

// $inp_t should be replaced in c_code
// we cannot use templated kernel because gpuarray API does not support it
#define NDIM            1
#define INPUT_TYPE      ga_half
#define INDEX_TYPE      ga_byte
#define bitsof(T)       (sizeof(T)*8)
#define radix_t         RadixConfig<INPUT_TYPE>::RadixType
#define WRITE_VALUE     0
#define WRITE_INDEX     1

#if RADIX_SIZE > 32
#error "RADIX_SIZE must be smaller than warp size (32)"
#endif

void __device__ atomicAdd(long long *dst, long long &src) {
    atomicAdd(
        reinterpret_cast<unsigned long long*>(dst),
        reinterpret_cast<unsigned long long&>(src));
}

template <typename T>
static inline __device__ T binary_cumsum(
    int idx, int warp_id, T* smem, bool value) {
    // cumsum within 1D thread block, which adds up `value` of all threads
    // whose id is *no greater than* the current thread
    // binary_cumsum(1, 0, 1, 0, 1) -> (1, 1, 2, 2, 3)

    // cumsum within warp
    unsigned int warp_bits = __ballot(value);
    T warp_sum = __popc(lane_mask_le() & warp_bits);

    if (lane_id() == 0)
        smem[warp_id] = __popc(warp_bits);

    local_barrier();

    // cumsum across warps in one thread
    if (idx == 0) {
        T sum = smem[0];
        for (int i = 1; i < blockDim.x / GA_WARP_SIZE; ++i) {
            sum += smem[i];
            smem[i] = sum;
        }
    }

    local_barrier();

    // load the carry from the preceding warp
    if (warp_id >= 1) {
        warp_sum = warp_sum+smem[warp_id - 1];
    }

    return warp_sum;
}

template <typename T>
static inline __device__ T binary_cumsum_exclusive(
    int idx, int warp_id, T* smem, bool value) {
    // cumsum within 1D thread block, which adds up `value` of all threads
    // whose id is *less than* the current thread
    // binary_cumsum_excl(1, 0, 1, 0, 1) -> (0, 1, 1, 2, 2)

    // cumsum within warp
    unsigned int warp_bits = __ballot(value);
    T warp_sum = __popc(lane_mask_lt() & warp_bits);

    if (lane_id() == 0)
        smem[warp_id] = __popc(warp_bits);

    local_barrier();

    // cumsum across warps in one thread
    if (idx == 0) {
        T sum = smem[0];
        for (int i = 1; i < blockDim.x / GA_WARP_SIZE; ++i) {
            sum += smem[i];
            smem[i] = sum;
        }
    }

    local_barrier();

    // load the carry from the preceding warp
    if (warp_id >= 1)
        warp_sum += smem[warp_id - 1];

    return warp_sum;
}

// apply raw(byte) offset to pointer
template <typename T>
static __device__ inline T* ptr_add(T *ptr, ssize_t offset) {
    return (T*)((char*)ptr + offset);
}

// get array element using raw(byte) offset
template <typename T>
static __device__ inline T& ptr_at(T *ptr, ssize_t offset) {
    return *((T*)((char*)ptr + offset));
}

// read array element using raw(byte) offset
template <typename T>
static __device__ inline T ptr_read_cached(T *ptr, ssize_t offset) {
    return __ldg(((T*)((char*)ptr + offset)));
}

#define RADIX_BITS 4
#define RADIX_SIZE      (1<<RADIX_BITS)
#define RADIX_MASK(n)   ((RADIX_SIZE-1) << (n*RADIX_BITS))
#define RADIX_DIGITS(T) (bitsof(T)/RADIX_BITS)

// works when length on axis is within max allowed threads in block (1024)
extern "C" __global__ void k_topk_dense(
        
        // size_t dims_1, ssize_t dims_2, ... , dims_${NDIM}
        
        // INPUT_TYPE *dstv
        
        // size_t offset
        
        // ssize_t dstv_strides_0, ssize_t dstv_strides_1, ... , dstv_strides_${NDIM}
        INDEX_TYPE *dsti,
        // INDEX_TYPE *dsti
        size_t dsti_offset,
        // size_t offset
        ssize_t dsti_strides_0, 
        // ssize_t dsti_strides_0, ssize_t dsti_strides_1, ... , dsti_strides_${NDIM}
        ssize_t k,
        INPUT_TYPE* src,
	size_t src_offset,
        ssize_t src_strides_0, 
        // ssize_t src_strides_0, ssize_t src_strides_1, ... , src_strides_${NDIM}
        size_t size) {
    __shared__ int smem[32 * RADIX_SIZE];
    __shared__ int k2;
    const unsigned int idx = threadIdx.x;
    bool is_topk= (idx < size);
    bool is_topkth = is_topk;
    size_t out_idx;

    const unsigned char warp_id = idx / GA_WARP_SIZE;
    // 0. get the slice for thread block to work on

    size_t gid = blockIdx.x, gidx;
    
                dsti = ptr_add(dsti, dsti_offset);
                
                src = ptr_add(src, src_offset);
            
    // $set_slice expands into:
    //for(int i=1; i<NDIM; i++) {
        // gidx = gid % dims_${i};
        // gid /= dims_${i};
        // dsti = ptr_add(dsti, gidx*dsti_strides_${i};
        // dstv = ptr_add(dstv, gidx*dstv_strides_${i};
        // src = ptr_add(src, gidx*src_strides_${i});
    //}

    // get input and its radix friendly form
    const INPUT_TYPE xval = is_topk ? ptr_at(src, idx*src_strides_0) : (INPUT_TYPE)0;
    radix_t x = RadixConfig<INPUT_TYPE>::convert(xval);

    // resolve negative k
    if (k<0) { x = ~x; k = -k; }
    if (idx==0)
        k2 = k;

    // 1. filter is_topk and is_topkth using radix select

    #pragma unroll
    for (int i=bitsof(INPUT_TYPE)-RADIX_BITS; i>=0; i-=RADIX_BITS) {
        const int digit = Bitfield<radix_t>::get(x, i, RADIX_BITS);
        /*int digit = (x>>i) & (RADIX_SIZE-1);*/
        // count within warp
        #pragma unroll
        for (int bin=0; bin<RADIX_SIZE; ++bin) {
            bool vote = (bin == digit) && is_topkth;
            unsigned int votes = __ballot(vote);
            if (lane_id()==0)
                smem[bin + RADIX_SIZE*warp_id] = __popc(votes);
        }
        local_barrier();
        // sum counts across all warps
        if (idx < RADIX_SIZE) {
            int sum = smem[idx];
            #pragma unroll
            for(int w=RADIX_SIZE; w<blockDim.x*RADIX_SIZE / GA_WARP_SIZE; w+=RADIX_SIZE)
                sum += smem[idx + w];
            smem[idx] = sum;
        }
        local_barrier();

        // find the bucket and update k2
        // smem[:RADIX_SIZE:-1] = k2 - cumsum(smem[:RADIX_SIZE-1:-1])
        if (idx == 0) {
            int sum = k2;
            #pragma unroll
            for (int bin=RADIX_SIZE-1; bin>=0; --bin) {
                sum -= smem[bin];
                smem[bin] = sum;
                k2 = (sum > 0) ? sum : k2;
            }
            smem[RADIX_SIZE] = 1;
        }
        local_barrier();

        if (is_topkth) {
            is_topk &= (smem[digit+1] > 0);
            is_topkth &= (smem[digit] <= 0) && (smem[digit+1] > 0);
        }
        local_barrier();
    }

    // set k2 as number of exceeding values
    if (idx==0) {
        #pragma unroll
        for (int bin=RADIX_SIZE-1; bin>=0; --bin) {
            if (smem[bin] <= 0)
                break;
            k2 = smem[bin];
        }
    }
    local_barrier();

    // 2. find the index of output array, if exists

    if (k2 != 0) {
        // top_kth value may not be unique, so we need to
        // perform binary cumsum on is_topkth to drop exceeding top-kth values
        out_idx = binary_cumsum_exclusive(idx, warp_id, smem, is_topkth);
        if ((out_idx >= k2) && is_topkth)
            is_topk = false;
        local_barrier();
    }

    // perform binary cumsum on is_topk to determine the indices to put result
    out_idx = binary_cumsum_exclusive(idx, warp_id, smem, is_topk);

    if (is_topk) {
#if WRITE_VALUE == 1
        ptr_at(dstv, out_idx * dstv_strides_0) = xval;
#endif
#if WRITE_INDEX == 1
        ptr_at(dsti, out_idx * dsti_strides_0) = (INDEX_TYPE)idx;
#endif
    }
}
NVRTC compile log::
default_program(434): error: no suitable constructor exists to convert from "int" to "ga_half"

default_program(418): warning: variable "gid" was declared but never referenced

default_program(418): warning: variable "gidx" was declared but never referenced

1 error detected in the compilation of "default_program".
