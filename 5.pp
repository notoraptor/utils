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

/* NB: __ldg is not defined for ga_half, so we must specialize ptr_read_cached.
 * To do it, I try to use a built-in type that should have the same size as ga_half.
 * Based on current ga_half implementation (2017/11/27), it should be ga_ushort.
 * This code must be updated every time ga_half implementation size changes,
 * until a better code be provided. */
#define GA_HALF_STD_TYPE ga_ushort
static __device__ inline ga_half ptr_read_cached(ga_half *ptr, ssize_t offset) {

    int check_ga_half_std_type[ ( ( sizeof(GA_HALF_STD_TYPE) - sizeof(ga_half) ) ? -1 : 1 ) ];

    return RadixConfig<ga_half>::deconvert(__ldg(((GA_HALF_STD_TYPE*)((char*)ptr + offset))));

}
#undef GA_HALF_STD_TYPE
#define RADIX_BITS 2
#define RADIX_SIZE      (1<<RADIX_BITS)
#define RADIX_DIGITS(T) (bitsof(T)/RADIX_BITS)

#define COUNT_TYPE int
#define KERNEL_NAME k_topk_dense_large

// if count_t is int, work for array size within [1025, 2^31-1]
// if count_t is long long, work for array size within [2^31, 2^63-1]
template <typename DataType, typename RadixType, typename CountType>
__device__ DataType find_pattern(DataType* smem,
                             DataType* data,
                             CountType slice_size,
                             CountType stride,
                             RadixType known_bits,
                             RadixType known_bits_mask) {
    if (threadIdx.x < 32)
        smem[threadIdx.x] = RadixConfig<DataType>::deconvert(0);

    local_barrier();

    // All threads participate in the loop, in order to sync on the flag
    for (CountType i = threadIdx.x; i < (slice_size + (CountType)blockDim.x-1); i += blockDim.x) {
        bool in_range = (i < slice_size);
        DataType v = in_range ? ptr_read_cached(data, i*stride) : 0;

        if (in_range && ((RadixConfig<DataType>::convert(v) & known_bits_mask) == known_bits)) {
            // There should not be conflicts if we are using find_pattern,
            // since the result is unique
            smem[0] = 1;
            smem[1] = v; // can't use val as the flag, since it could be 0
        }

        local_barrier();

        DataType found = smem[0];
        DataType val = smem[1];

        local_barrier();

        // Check to see if a thread found the value
        if (found != 0)
            return val;
    }
    return 0;
}

// This function counts the distribution of all input values in a
// slice we are selecting by radix digit at `radix_digit_pos`, but only
// those that pass the filter `((v & known_bits_mask) == known_bits)`.
// This produces and broadcasts the seen counts for a single block only.
// `smem` must have at least `RADIX_SIZE` elements.
template <typename DataType, typename RadixType, typename CountType>
__device__ void count_radix_masked(CountType counts[RADIX_SIZE],
                                    CountType* smem,
                                    RadixType known_bits,
                                    RadixType known_bits_mask,
                                    int radix_digit_pos,
                                    CountType slice_size,
                                    CountType stride,
                                    DataType* data) {
    // Clear out per-thread counts from a previous round
#pragma unroll
    for (int i = 0; i < RADIX_SIZE; ++i)
        counts[i] = 0;

    if (threadIdx.x < RADIX_SIZE)
        smem[threadIdx.x] = 0;

    local_barrier();

    // Scan over all the data. Upon a read, the warp will accumulate
    // counts per each digit in the radix using warp voting.
    for (CountType i = threadIdx.x; i < slice_size; i += blockDim.x) {
        RadixType val = RadixConfig<DataType>::convert(ptr_read_cached(data, i*stride));

        bool has_val = ((val & known_bits_mask) == known_bits);
        RadixType digit_in_radix = Bitfield<RadixType>::get(val, radix_digit_pos, RADIX_BITS);

        #pragma unroll
        for (int j = 0; j < RADIX_SIZE; ++j) {
            bool vote = has_val && (digit_in_radix == j);
            counts[j] += __popc(__ballot(vote));
        }
    }

    // Now, for each warp, sum values
    if (lane_id() == 0) {
        for (int i=0; i<RADIX_SIZE; ++i)
            atomicAdd(&smem[i], counts[i]);
    }
    /*
    // not sure why, but this just give wrong results
    if (lane_id() < RADIX_SIZE)
        atomicAdd(&smem[lane_id()], counts[lane_id()]);
        */

    local_barrier();

    // For each thread, read in the total counts
    #pragma unroll
    for (unsigned int i = 0; i < RADIX_SIZE; ++i)
        counts[i] = smem[i];

    local_barrier();
}

template <typename DataType, typename RadixType, typename CountType>
__device__ void radix_select(DataType* data,
                            CountType k,
                            bool order,
                            CountType slice_size,
                            CountType stride,
                            CountType* smem,
                            DataType* top_kth) {
    // Per-thread buckets into which we accumulate digit counts in our
    // radix
    register CountType counts[RADIX_SIZE];

    // We only consider elements x such that (x & known_bits_mask) == known_bits
    // Initially, we consider all elements of the array, so the above
    // statement is true regardless of input.
    RadixType known_bits = 0, known_bits_mask = 0;

    // We are looking for the top k_to_find-th element when iterating over
    // digits; this count gets reduced by elimination when counting
    // successive digits
    CountType k_to_find = abs(k);

    // We start at the most significant digit in our radix, scanning
    // through to the least significant digit
    #pragma unroll
    for (int digit_pos = bitsof(DataType) - RADIX_BITS;
            digit_pos >= 0; digit_pos -= RADIX_BITS) {

        // Count radix distribution for the current position and reduce
        // across all threads
        count_radix_masked<DataType, RadixType, CountType>(
                    counts, smem,
                    known_bits, known_bits_mask, digit_pos,
                    slice_size, stride, data);

        // All threads participate in the comparisons below to know the
        // final result

        #define CHECK_RADIX(i) \
            int count = counts[i]; \
            /* All threads have the same value in counts here, so all */  \
            /* threads will return from the function. */  \
            if (count == 1 && k_to_find == 1) {  \
                /* There is a unique answer. */  \
                known_bits = Bitfield<RadixType>::set(  \
                    known_bits, i, digit_pos, RADIX_BITS);  \
                known_bits_mask = Bitfield<RadixType>::set(  \
                    known_bits_mask, RADIX_SIZE-1, digit_pos, RADIX_BITS);  \
                /* The answer is now the unique element v such that: */  \
                /* (v & known_bits_mask) == known_bits */  \
                /* However, we do not yet know what the actual element is. We */  \
                /* need to perform a search through the data to find the */  \
                /* element that matches this pattern. */  \
                *top_kth = find_pattern<DataType, RadixType, CountType>(  \
                        (DataType*) smem, data, slice_size,  \
                        stride, known_bits, known_bits_mask);  \
                return;  \
            }  \
            if (count >= k_to_find) {  \
                known_bits = Bitfield<RadixType>::set(known_bits, i, digit_pos, RADIX_BITS);  \
                known_bits_mask = Bitfield<RadixType>::set(  \
                    known_bits_mask, RADIX_SIZE-1, digit_pos, RADIX_BITS);  \
                /* The top-Kth element v must now be one such that: */  \
                /* (v & known_bits_mask == known_bits) */  \
                /* but we haven't narrowed it down; we must check the next */  \
                /* least-significant digit */  \
                break;  \
            }  \
            k_to_find -= count

        if (order) {
            #pragma unroll
            for (int i=RADIX_SIZE - 1; i >= 0; --i) {
                CHECK_RADIX(i);
            }
        } else {
            #pragma unroll
            for (int i=0; i < RADIX_SIZE; ++i) {
                CHECK_RADIX(i);
            }
        }
        #undef CHECK_RADIX
    } // end digit_pos for

    // There is no unique result, but there is a non-unique result
    // matching `known_bits` exactly
    *top_kth = RadixConfig<DataType>::deconvert(known_bits);
}

extern "C" __global__ void KERNEL_NAME(
        
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
    __shared__ COUNT_TYPE smem[32];
    INPUT_TYPE topkth_value;

    const bool order = (k>0);
    k = (order ? k : -k);
    const int idx = threadIdx.x;
    const int warp_id = idx / GA_WARP_SIZE;

    // get the slice for thread block to work on
    // size <- the axis to work on
    // dims_1+ <- batched dimensions
    unsigned int gid = blockIdx.x, gidx;
    
                dsti = ptr_add(dsti, dsti_offset);
                
                src = ptr_add(src, src_offset);
            
    // $set_slice expands into:
    //for(int i=1; i<NDIM; i++) {
        // gidx = gid % dims_${i};
        // gid /= dims_${i};
        // dsti = ptr_add(dsti, gidx*dsti_strides_${i});
        // dstv = ptr_add(dstv, gidx*dstv_strides_${i});
        // src = ptr_add(src, gidx*src_strides_${i});
    //}

    radix_select<INPUT_TYPE, radix_t, COUNT_TYPE>(
        src, k, order, size, src_strides_0,
        smem, &topkth_value);

    // Every value that is strictly less/greater than `pattern`
    // (depending on sort dir) in sorted int format is in the top-K.
    // The top-K value itself might not be unique.
    //
    // Since there are a variable number of elements that we see that
    // are within the top-k, we don't know at what index to write out
    // the resulting values.
    // In order to get this, we perform an exclusive cumsum of
    // `has_topk`. This will return the resulting index into which we
    // need to write the result, if a thread has a result.

    // All threads need to participate in the loop and the cumsum
    // but not necessarily in the load; hence loop bounds being rounded
    // up to a multiple of the block dim.
    COUNT_TYPE iter_bound = size + blockDim.x-1;
    INDEX_TYPE write_base = 0;

    for (int i = idx; i < iter_bound; i += blockDim.x) {
        bool in_range = (i < size);
        INPUT_TYPE v = in_range ? ptr_read_cached(src, i*src_strides_0) : 0;
        bool has_topk;
        if (order) {
            has_topk = in_range && (v > topkth_value);
        } else {
            has_topk = in_range && (v < topkth_value);
        }

        int index = binary_cumsum_exclusive(idx, warp_id, smem, has_topk);
        int carry = smem[blockDim.x / 32 - 1];

        if (has_topk) {
            COUNT_TYPE write_idx = write_base + index;
#if WRITE_VALUE == 1
            ptr_at(dstv, write_idx * dstv_strides_0) = v;
#endif
#if WRITE_INDEX == 1
            ptr_at(dsti, write_idx * dsti_strides_0) = (INDEX_TYPE)i;
#endif
        }

        write_base += carry;
    }

    COUNT_TYPE topk_remaining = (k - write_base);

    for (COUNT_TYPE i = idx; i < iter_bound; i += blockDim.x) {
        bool in_range = (i < size);
        INPUT_TYPE v = in_range ? ptr_read_cached(src, i*src_strides_0) : 0;
        bool has_topk = in_range && (v == topkth_value);

        int index = binary_cumsum_exclusive(idx, warp_id, smem, has_topk);
        int carry = smem[blockDim.x / 32 - 1];

        if (has_topk && index < topk_remaining) {
            COUNT_TYPE write_idx = write_base + index;
#if WRITE_VALUE == 1
            ptr_at(dstv, write_idx * dstv_strides_0) = v;
#endif
#if WRITE_INDEX == 1
            ptr_at(dsti, write_idx * dsti_strides_0) = (INDEX_TYPE)i;
#endif
        }

        if (carry >= topk_remaining)
            break;

        topk_remaining -= carry;
        write_base += carry;
    }
}

NVRTC compile log::
default_program(389): warning: variable "check_ga_half_std_type" was declared but never referenced

default_program(661): error: operand types are incompatible ("ga_half" and "int")

default_program(664): error: no operator ">" matches these operands
            operand types are: ga_half > ga_half

default_program(666): error: no operator "<" matches these operands
            operand types are: ga_half < ga_half

default_program(689): error: operand types are incompatible ("ga_half" and "int")

default_program(690): error: no operator "==" matches these operands
            operand types are: ga_half == ga_half

default_program(623): warning: variable "gid" was declared but never referenced

default_program(623): warning: variable "gidx" was declared but never referenced

default_program(419): error: operand types are incompatible ("ga_half" and "int")
          detected during:
            instantiation of "DataType find_pattern(DataType *, DataType *, CountType, CountType, RadixType, RadixType) [with DataType=ga_half, RadixType=unsigned int, CountType=int]" 
(575): here
            instantiation of "void radix_select<DataType,RadixType,CountType>(DataType *, CountType, __nv_bool, CountType, CountType, CountType *, DataType *) [with DataType=ga_half, RadixType=unsigned int, CountType=int]" 
(638): here

default_program(424): error: no operator "=" matches these operands
            operand types are: ga_half = int
          detected during:
            instantiation of "DataType find_pattern(DataType *, DataType *, CountType, CountType, RadixType, RadixType) [with DataType=ga_half, RadixType=unsigned int, CountType=int]" 
(575): here
            instantiation of "void radix_select<DataType,RadixType,CountType>(DataType *, CountType, __nv_bool, CountType, CountType, CountType *, DataType *) [with DataType=ga_half, RadixType=unsigned int, CountType=int]" 
(638): here

default_program(436): error: no operator "!=" matches these operands
            operand types are: ga_half != int
          detected during:
            instantiation of "DataType find_pattern(DataType *, DataType *, CountType, CountType, RadixType, RadixType) [with DataType=ga_half, RadixType=unsigned int, CountType=int]" 
(575): here
            instantiation of "void radix_select<DataType,RadixType,CountType>(DataType *, CountType, __nv_bool, CountType, CountType, CountType *, DataType *) [with DataType=ga_half, RadixType=unsigned int, CountType=int]" 
(638): here

default_program(439): error: no suitable constructor exists to convert from "int" to "ga_half"
          detected during:
            instantiation of "DataType find_pattern(DataType *, DataType *, CountType, CountType, RadixType, RadixType) [with DataType=ga_half, RadixType=unsigned int, CountType=int]" 
(575): here
            instantiation of "void radix_select<DataType,RadixType,CountType>(DataType *, CountType, __nv_bool, CountType, CountType, CountType *, DataType *) [with DataType=ga_half, RadixType=unsigned int, CountType=int]" 
(638): here

9 errors detected in the compilation of "default_program".

ERROR 3: nvrtcCompileProgram: NVRTC_ERROR_COMPILATION
