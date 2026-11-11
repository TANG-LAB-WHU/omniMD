//! Whole networks in one kernel (plan I-3).
//!
//! Layer-by-layer GEMMs write every intermediate activation to device memory
//! and need a dozen launches per network, which limits large batches by
//! bandwidth and small ones by launch latency. The two kernels here keep the
//! intermediates in shared memory and take one launch for any number of
//! same-shaped networks: each row carries the index of the network it uses.
//!
//! - [`ScalarNetwork`]: one input per row, value and derivative of every
//!   output (the se_e2_a embedding nets, one per type pair).
//! - [`GradientNetwork`]: one output per row, value and gradient with respect
//!   to the inputs (the fitting nets, one per center type).
//!
//! Same layer semantics as [`crate::DenseLayer`].
use crate::dense::validate;
use crate::{check_len, cuda_error, Activation, Device, Scalar};
use cudarc::driver::{
    CudaFunction, CudaSlice, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use omnimd_ml_contract::{Error, Result};

const ACTIVATE: &str = r#"
template <typename T>
__device__ void activate(int act, T z, T* v, T* s) {
    if (act == 1) {
        *v = tanh(z);
        *s = T(1) - *v * *v;
    } else if (act == 2 || act == 3) {
        T e = exp(z >= T(0) ? -z : z);
        T sigmoid = z >= T(0) ? T(1) / (T(1) + e) : e / (T(1) + e);
        if (act == 2) {
            *v = sigmoid;
            *s = sigmoid * (T(1) - sigmoid);
        } else {
            *v = (z > T(0) ? z : T(0)) + log1p(e);
            *s = sigmoid;
        }
    } else {
        *v = z;
        *s = T(1);
    }
}
"#;

const SCALAR_SOURCE: &str = r#"
// table: 7 ints per layer {in, out, activation, residual, weights, bias, timestep}
// with the last three as offsets into one network's parameter block; weights
// are [in][out]. Network k's block starts at params + k * net_size.
//
// One warp per row. Each lane owns up to C output neurons at a time
// (n, n+32, ...) and keeps their sums in registers while the loop over the
// inputs broadcasts one activation and its tangent from shared memory, so
// every weight load (coalesced across the lanes) feeds two multiply-adds.
// A row with a negative id is skipped: nothing is written for it.
template <typename T, int C>
__device__ void scalar_network(const T* x, const int* ids, long long rows, const int* table,
        int layers, const T* params, int net_size, int width, T* out, T* dout) {
    extern __shared__ double shared[];
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    long long row = (long long)blockIdx.x * warps + warp;
    if (row >= rows || (ids && ids[row] < 0)) return;
    const T* block = params + (ids ? (long long)ids[row] * net_size : 0);
    T* y = (T*)shared + (long long)warp * 4 * width;
    T* dy = y + width;
    T* next = dy + width;
    T* dnext = next + width;
    if (lane == 0) { y[0] = x[row]; dy[0] = T(1); }
    __syncwarp();
    for (int l = 0; l < layers; ++l) {
        const int* e = table + 7 * l;
        int in = e[0], on = e[1], act = e[2], residual = e[3];
        const T* w = block + e[4];
        const T* bias = block + e[5];
        const T* dt = block + e[6];
        bool last = l + 1 == layers;
        for (int n0 = lane; n0 < on; n0 += 32 * C) {
            T z[C], d[C];
            for (int c = 0; c < C; ++c) { z[c] = T(0); d[c] = T(0); }
            const T* wk = w;
            for (int k = 0; k < in; ++k, wk += on) {
                T yk = y[k], dk = dy[k];
                for (int c = 0; c < C; ++c) {
                    int n = n0 + 32 * c;
                    if (n < on) { T v = wk[n]; z[c] += yk * v; d[c] += dk * v; }
                }
            }
            for (int c = 0; c < C; ++c) {
                int n = n0 + 32 * c;
                if (n >= on) break;
                T v, s;
                activate(act, z[c] + bias[n], &v, &s);
                v *= dt[n];
                T dv = s * dt[n] * d[c];
                if (residual) { v += y[n % in]; dv += dy[n % in]; }
                if (last) {
                    out[row * on + n] = v;
                    dout[row * on + n] = dv;
                } else {
                    next[n] = v;
                    dnext[n] = dv;
                }
            }
        }
        __syncwarp();
        T* t = y; y = next; next = t;
        t = dy; dy = dnext; dnext = t;
    }
}

// Float32 is limited by loads, so four neurons per lane; float64 is limited
// by the double-precision units and measures fastest with one.
#define INSTANTIATE(T, S, C)                                                                  \
extern "C" __global__ void scalar_network_##S(const T* x, const int* ids, long long rows,     \
        const int* table, int layers, const T* params, int net_size, int width, T* out,       \
        T* dout) {                                                                            \
    scalar_network<T, C>(x, ids, rows, table, layers, params, net_size, width, out, dout);    \
}
INSTANTIATE(float, f32, 4)
INSTANTIATE(double, f64, 1)
"#;

const GRADIENT_SOURCE: &str = r#"
template <typename T>
__device__ T warp_sum(T v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

// sum over k < count of a[k] * w[k * stride], in U independent partial sums.
// With few rows the kernel is bound by the latency of this chain in float32
// (U = 8); float64 is bound by the double-precision units and measures
// fastest with a single sum.
template <typename T, int U>
__device__ T strided_dot(const T* a, const T* w, int count, int stride) {
    T acc[U];
    for (int u = 0; u < U; ++u) acc[u] = T(0);
    int k = 0;
    for (; k + U <= count; k += U, w += U * (long long)stride)
        for (int u = 0; u < U; ++u) acc[u] += a[k + u] * w[u * (long long)stride];
    for (; k < count; ++k, w += stride) acc[0] += a[k] * *w;
    T total = acc[0];
    for (int u = 1; u < U; ++u) total += acc[u];
    return total;
}

// table: 10 ints per layer {in, out, activation, residual, weights, transposed
// weights, bias, timestep, y, slope}: the middle four are offsets into one
// network's parameter block (weights [in][out], transposed [out][in]), the
// last two offsets into shared memory of that layer's outputs and slopes
// (hidden layers only). Shared memory: the input row, every hidden layer's
// outputs and slopes, two adjoint buffers of the widest hidden layer, one
// partial sum per warp.
//
// One block per row. Forward: thread n owns output neuron n, the loop over the
// inputs broadcasts one activation from shared memory and reads one weight
// row coalesced across the block. Backward: thread k owns input k and reads
// the transposed weights the same way, so the last pass writes the gradient
// row coalesced.
//
// With `partial` the first layer's products are done elsewhere (the tile
// kernels below): its pre-activations are the bias plus `tiles` partial sums
// per neuron, partial[row][tile][n]; the input row is not needed and not kept
// (the table's shared-memory offsets still count it); and what goes to
// `gradient` is delta[row][n], the adjoint of the first layer's
// pre-activations, for the kernel that multiplies by the weights.
template <typename T, int U>
__device__ void gradient_network(const T* x, const int* ids, long long rows, const int* table,
        int layers, const T* params, int net_size, int adjoint, T* value, T* gradient,
        const T* partial, int tiles) {
    extern __shared__ double shared[];
    long long row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x, threads = blockDim.x;
    const T* block = params + (ids ? (long long)ids[row] * net_size : 0);
    int in0 = table[0];
    T* sh = (T*)shared;
    const T* current = sh;
    int hidden = 0;
    if (partial) {
        // The first layer from its partial sums (it has no residual here).
        sh -= in0;
        int on = table[1];
        const T* bias = block + table[6];
        const T* dt = block + table[7];
        T* y = sh + table[8];
        T* slope = sh + table[9];
        const T* sums = partial + row * tiles * on;
        for (int n = tid; n < on; n += threads) {
            T z = bias[n];
            for (int t = 0; t < tiles; ++t) z += sums[(long long)t * on + n];
            T v, s;
            activate(table[2], z, &v, &s);
            y[n] = v * dt[n];
            slope[n] = s * dt[n];
        }
        current = y;
        hidden = 1;
    } else {
        for (int k = tid; k < in0; k += threads) sh[k] = x[row * in0 + k];
    }
    __syncthreads();
    for (int l = hidden; l + 1 < layers; ++l) {
        const int* e = table + 10 * l;
        int in = e[0], on = e[1], act = e[2], residual = e[3];
        const T* w = block + e[4];
        const T* bias = block + e[6];
        const T* dt = block + e[7];
        T* y = sh + e[8];
        T* slope = sh + e[9];
        for (int n = tid; n < on; n += threads) {
            T z = bias[n] + strided_dot<T, U>(current, w + n, in, on);
            T v, s;
            activate(act, z, &v, &s);
            v *= dt[n];
            if (residual) v += current[n % in];
            y[n] = v;
            slope[n] = s * dt[n];
        }
        __syncthreads();
        current = y;
    }
    // Output layer: one value per row, reduced over the block.
    const int* last = table + 10 * (layers - 1);
    int width = last[0];
    const T* w_out = block + last[4];
    T* a = sh + adjoint;
    T* b = a + table[10 * layers];
    T* partial_sum = b + table[10 * layers];
    T* seed = partial_sum + (threads >> 5);
    T sum = T(0);
    for (int k = tid; k < width; k += threads) sum += current[k] * w_out[k];
    sum = warp_sum(sum);
    if ((tid & 31) == 0) partial_sum[tid >> 5] = sum;
    __syncthreads();
    if (tid == 0) {
        T z = (block + last[6])[0];
        for (int wi = 0; wi < (threads >> 5); ++wi) z += partial_sum[wi];
        T v, s;
        activate(last[2], z, &v, &s);
        T scale = (block + last[7])[0];
        value[row] = v * scale;
        *seed = s * scale;
    }
    __syncthreads();
    T g = *seed;
    if (layers == 1) {
        for (int k = tid; k < width; k += threads) gradient[row * in0 + k] = g * w_out[k];
        return;
    }
    for (int k = tid; k < width; k += threads) a[k] = g * w_out[k];
    __syncthreads();
    for (int l = layers - 2; l >= 0; --l) {
        const int* e = table + 10 * l;
        int in = e[0], on = e[1], residual = e[3];
        const T* wt = block + e[5];
        T* slope = sh + e[9];
        if (partial && l == 0) {
            for (int n = tid; n < on; n += threads) gradient[row * on + n] = slope[n] * a[n];
            return;
        }
        // The slopes are not needed again: turn them into the adjoint of the
        // layer's pre-activations.
        for (int n = tid; n < on; n += threads) slope[n] *= a[n];
        __syncthreads();
        T* target = l == 0 ? gradient + row * in0 : b;
        for (int k = tid; k < in; k += threads) {
            T acc = strided_dot<T, U>((const T*)slope, wt + k, on, in);
            if (residual)
                for (int n = k; n < on; n += in) acc += a[n];
            target[k] = acc;
        }
        __syncthreads();
        T* t = a; a = b; b = t;
    }
}

// The layers after the first for up to R rows of one network in a block.
// `gradient_network` with `partial` does one row a block and reads every
// weight of these layers from global memory for that row alone. Here a
// weight that is read serves every row of the block: thread n owns output
// neuron n going forward and input k going backward as there, with a sum
// per row. Two rows a block (the only R built) measured faster than one at
// every number of rows, 31 against 35 microseconds for 88 rows and 67
// against 136 for 704; four and eight measured no better than two, because
// a block then takes that much longer and fewer of them fit a
// multiprocessor's shared memory.
//
// chunks: {first row, rows, network} per block, rows of one network, at most
// R of them. Shared memory, for each row: the slopes of every hidden layer,
// then two work rows of the widest hidden layer (activations going forward,
// adjoints going backward); after the rows, a partial sum per warp and row,
// then a seed per row. The sums run over all R rows of shared memory
// whatever the chunk holds, so that their loops have no branches and
// unroll; what is summed for a row the chunk does not have is never stored.
template <typename T, int R>
__device__ void middle_rows(const T* partial, int tiles, const int* chunks, const int* table,
        int layers, const T* params, int net_size, T* value, T* delta) {
    extern __shared__ double shared[];
    const int* c = chunks + 3 * blockIdx.x;
    int r0 = c[0], rc = c[1];
    const T* block = params + (long long)c[2] * net_size;
    int tid = threadIdx.x, threads = blockDim.x, warps = threads >> 5;
    int hidden = layers - 1, widest = table[10 * layers];
    int slopes = 0;
    for (int l = 0; l < hidden; ++l) slopes += table[10 * l + 1];
    int per_row = slopes + 2 * widest;
    T* sh = (T*)shared;
    T* sums = sh + R * per_row;
    T* seeds = sums + warps * R;
    // The first layer from its partial sums; its slopes lead a row.
    {
        int on = table[1], act = table[2];
        const T* bias = block + table[6];
        const T* dt = block + table[7];
        for (int n = tid; n < on; n += threads) {
            T b0 = bias[n], scale = dt[n];
            for (int r = 0; r < rc; ++r) {
                const T* p = partial + (long long)(r0 + r) * tiles * on + n;
                T z = b0;
                for (int t = 0; t < tiles; ++t) z += p[(long long)t * on];
                T v, s;
                activate(act, z, &v, &s);
                T* mine = sh + r * per_row;
                mine[slopes + n] = v * scale;
                mine[n] = s * scale;
            }
        }
    }
    __syncthreads();
    // Work rows: `cur` holds a layer's inputs, `nxt` takes its outputs.
    int cur = slopes, nxt = slopes + widest, soff = table[1];
    for (int l = 1; l < hidden; ++l) {
        const int* e = table + 10 * l;
        int in = e[0], on = e[1], act = e[2], residual = e[3];
        const T* w = block + e[4];
        const T* bias = block + e[6];
        const T* dt = block + e[7];
        for (int n = tid; n < on; n += threads) {
            T acc[R];
            #pragma unroll
            for (int r = 0; r < R; ++r) acc[r] = T(0);
            const T* column = w + n;
            const T* input = sh + cur;
            for (int k = 0; k < in; ++k, column += on) {
                T v = *column;
                #pragma unroll
                for (int r = 0; r < R; ++r) acc[r] += input[r * per_row + k] * v;
            }
            T b0 = bias[n], scale = dt[n];
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                if (r >= rc) break;
                T* mine = sh + r * per_row;
                T v, s;
                activate(act, b0 + acc[r], &v, &s);
                v *= scale;
                if (residual) v += mine[cur + n % in];
                mine[nxt + n] = v;
                mine[soff + n] = s * scale;
            }
        }
        __syncthreads();
        int t = cur; cur = nxt; nxt = t;
        soff += on;
    }
    // Output layer: one value per row, reduced over the block.
    const int* last = table + 10 * (layers - 1);
    int width = last[0];
    const T* w_out = block + last[4];
    {
        T part[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) part[r] = T(0);
        const T* input = sh + cur;
        for (int k = tid; k < width; k += threads) {
            T v = w_out[k];
            #pragma unroll
            for (int r = 0; r < R; ++r) part[r] += input[r * per_row + k] * v;
        }
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            T total = warp_sum(part[r]);
            if ((tid & 31) == 0) sums[(tid >> 5) * R + r] = total;
        }
    }
    __syncthreads();
    if (tid < rc) {
        T z = (block + last[6])[0];
        for (int wi = 0; wi < warps; ++wi) z += sums[wi * R + tid];
        T v, s;
        activate(last[2], z, &v, &s);
        T scale = (block + last[7])[0];
        value[r0 + tid] = v * scale;
        seeds[tid] = s * scale;
    }
    __syncthreads();
    // The adjoint of the last hidden layer's outputs goes to the free work
    // row; from here `a` is the adjoint and `b` the row to write next.
    int a = nxt, b = cur;
    for (int k = tid; k < width; k += threads) {
        T v = w_out[k];
        for (int r = 0; r < rc; ++r) (sh + r * per_row)[a + k] = seeds[r] * v;
    }
    __syncthreads();
    for (int l = hidden - 1; l >= 1; --l) {
        const int* e = table + 10 * l;
        int in = e[0], on = e[1], residual = e[3];
        const T* wt = block + e[5];
        soff -= on;
        // The slopes are not needed again: turn them into the adjoint of
        // the layer's pre-activations.
        for (int n = tid; n < on; n += threads)
            for (int r = 0; r < rc; ++r) {
                T* mine = sh + r * per_row;
                mine[soff + n] *= mine[a + n];
            }
        __syncthreads();
        for (int k = tid; k < in; k += threads) {
            T acc[R];
            #pragma unroll
            for (int r = 0; r < R; ++r) acc[r] = T(0);
            const T* column = wt + k;
            const T* adjoint = sh + soff;
            for (int n = 0; n < on; ++n, column += in) {
                T v = *column;
                #pragma unroll
                for (int r = 0; r < R; ++r) acc[r] += adjoint[r * per_row + n] * v;
            }
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                if (r >= rc) break;
                T* mine = sh + r * per_row;
                T g = acc[r];
                if (residual)
                    for (int n = k; n < on; n += in) g += mine[a + n];
                mine[b + k] = g;
            }
        }
        __syncthreads();
        int t = a; a = b; b = t;
    }
    // The adjoint of the first layer's pre-activations, for the kernel that
    // multiplies by its weights.
    int on = table[1];
    for (int n = tid; n < on; n += threads)
        for (int r = 0; r < rc; ++r) {
            const T* mine = sh + r * per_row;
            delta[(long long)(r0 + r) * on + n] = mine[n] * mine[a + n];
        }
}

// The first layer in tiles. A dense layer that reads its weights from global
// memory is bound by those reads, a row at a time; here a tile of the weights
// is copied to shared memory once and serves every row of a chunk.
//
// chunks: 3 ints per chunk of rows {first row, rows, network}; the rows of a
// chunk use one network. A block is one chunk and one tile of inputs.

// partial[row][tile][n] = sum over the tile's inputs k of x[row][k] w[k][n].
// The tile's weights [tile_inputs][out, padded to whole warps] and the rows'
// inputs go to shared memory. A warp does two rows and a lane up to eight
// outputs at a time (n, n + 32, ...): one read of a weight serves two
// products, one read of an input sixteen.
template <typename T>
__device__ void tile_forward(const T* x, const T* params, const int* chunks, int in, int out,
        int tile_inputs, int net_size, int weights, T* partial) {
    extern __shared__ double shared[];
    int tiles = (in + tile_inputs - 1) / tile_inputs;
    int chunk = blockIdx.x / tiles, tile = blockIdx.x % tiles;
    const int* c = chunks + 3 * chunk;
    int r0 = c[0], rc = c[1], k0 = tile * tile_inputs, kc = min(tile_inputs, in - k0);
    const T* w = params + (long long)c[2] * net_size + weights;
    int padded = (out + 31) & ~31;
    T* sw = (T*)shared;
    T* sa = sw + tile_inputs * padded;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, warps = blockDim.x >> 5;
    for (int k = warp; k < kc; k += warps)
        for (int n = lane; n < padded; n += 32)
            sw[k * padded + n] = n < out ? w[(long long)(k0 + k) * out + n] : T(0);
    for (int r = warp; r < rc; r += warps)
        for (int k = lane; k < kc; k += 32)
            sa[r * tile_inputs + k] = x[(long long)(r0 + r) * in + k0 + k];
    __syncthreads();
    int first = 2 * warp;
    if (first >= rc) return;
    // A last row without a partner is done twice and stored once.
    bool pair = first + 1 < rc;
    const T* a0 = sa + first * tile_inputs;
    const T* a1 = a0 + (pair ? tile_inputs : 0);
    T* out0 = partial + ((long long)(r0 + first) * tiles + tile) * out;
    T* out1 = out0 + (long long)tiles * out;
    int columns = padded >> 5;
    for (int c0 = 0; c0 < columns; c0 += 8) {
        T acc0[8], acc1[8];
        for (int u = 0; u < 8; ++u) { acc0[u] = T(0); acc1[u] = T(0); }
        const T* column = sw + lane + 32 * c0;
        int count = min(8, columns - c0);
        if (count == 8) {
            for (int k = 0; k < kc; ++k, column += padded) {
                T p = a0[k], q = a1[k];
                for (int u = 0; u < 8; ++u) {
                    T v = column[32 * u];
                    acc0[u] += p * v;
                    acc1[u] += q * v;
                }
            }
        } else {
            for (int k = 0; k < kc; ++k, column += padded) {
                T p = a0[k], q = a1[k];
                for (int u = 0; u < count; ++u) {
                    T v = column[32 * u];
                    acc0[u] += p * v;
                    acc1[u] += q * v;
                }
            }
        }
        for (int u = 0; u < count; ++u) {
            int n = lane + 32 * (c0 + u);
            if (n >= out) continue;
            out0[n] = acc0[u];
            if (pair) out1[n] = acc1[u];
        }
    }
}

// gradient[row][k] = sum_n delta[row][n] w[k][n], the first layer backward.
// A tile is `tile_inputs` (at most 32) inputs, one lane each; its transposed
// weights [out][tile_inputs] and the rows' deltas go to shared memory. A warp
// does two rows, like the forward tile: the kernel is bound by reading the
// weights from shared memory, one read for each lane, and a read then serves
// two products.
template <typename T>
__device__ void tile_backward(const T* delta, const T* params, const int* chunks, int in,
        int out, int tile_inputs, int net_size, int transposed, T* gradient) {
    extern __shared__ double shared[];
    int tiles = (in + tile_inputs - 1) / tile_inputs;
    int chunk = blockIdx.x / tiles, tile = blockIdx.x % tiles;
    const int* c = chunks + 3 * chunk;
    int r0 = c[0], rc = c[1], k0 = tile * tile_inputs, kc = min(tile_inputs, in - k0);
    const T* wt = params + (long long)c[2] * net_size + transposed;
    T* sw = (T*)shared;
    T* sd = sw + out * tile_inputs;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, warps = blockDim.x >> 5;
    if (lane < kc)
        for (int n = warp; n < out; n += warps)
            sw[n * tile_inputs + lane] = wt[(long long)n * in + k0 + lane];
    for (int r = warp; r < rc; r += warps)
        for (int n = lane; n < out; n += 32)
            sd[r * out + n] = delta[(long long)(r0 + r) * out + n];
    __syncthreads();
    int first = 2 * warp;
    if (first >= rc || lane >= kc) return;
    // A last row without a partner is done twice and stored once.
    bool pair = first + 1 < rc;
    const T* d0 = sd + first * out;
    const T* d1 = d0 + (pair ? out : 0);
    const T* column = sw + lane;
    T acc0[4], acc1[4];
    for (int u = 0; u < 4; ++u) { acc0[u] = T(0); acc1[u] = T(0); }
    int n = 0;
    for (; n + 4 <= out; n += 4)
        for (int u = 0; u < 4; ++u) {
            T v = column[(n + u) * tile_inputs];
            acc0[u] += d0[n + u] * v;
            acc1[u] += d1[n + u] * v;
        }
    for (; n < out; ++n) {
        T v = column[n * tile_inputs];
        acc0[0] += d0[n] * v;
        acc1[0] += d1[n] * v;
    }
    T* g = gradient + (long long)(r0 + first) * in + k0 + lane;
    g[0] = (acc0[0] + acc0[1]) + (acc0[2] + acc0[3]);
    if (pair) g[in] = (acc1[0] + acc1[1]) + (acc1[2] + acc1[3]);
}

#define INSTANTIATE(T, S, U)                                                                  \
extern "C" __global__ void gradient_network_##S(const T* x, const int* ids, long long rows,   \
        const int* table, int layers, const T* params, int net_size, int adjoint, T* value,   \
        T* gradient) {                                                                        \
    gradient_network<T, U>(x, ids, rows, table, layers, params, net_size, adjoint, value,     \
                           gradient, nullptr, 0);                                             \
}                                                                                             \
extern "C" __global__ void gradient_middle_##S(const T* partial, int tiles, const int* ids,   \
        long long rows, const int* table, int layers, const T* params, int net_size,          \
        int adjoint, T* value, T* delta) {                                                    \
    gradient_network<T, U>(nullptr, ids, rows, table, layers, params, net_size, adjoint,      \
                           value, delta, partial, tiles);                                     \
}                                                                                             \
extern "C" __global__ void tile_forward_##S(const T* x, const T* params, const int* chunks,   \
        int in, int out, int tile_inputs, int net_size, int weights, T* partial) {            \
    tile_forward<T>(x, params, chunks, in, out, tile_inputs, net_size, weights, partial);     \
}                                                                                             \
extern "C" __global__ void tile_backward_##S(const T* delta, const T* params,                 \
        const int* chunks, int in, int out, int tile_inputs, int net_size, int transposed,    \
        T* gradient) {                                                                        \
    tile_backward<T>(delta, params, chunks, in, out, tile_inputs, net_size, transposed,       \
                     gradient);                                                               \
}
INSTANTIATE(float, f32, 8)
INSTANTIATE(double, f64, 1)

#define PAIRS(T, S)                                                                           \
extern "C" __global__ void gradient_pairs_##S(const T* partial, int tiles, const int* chunks, \
        const int* table, int layers, const T* params, int net_size, T* value, T* delta) {    \
    middle_rows<T, 2>(partial, tiles, chunks, table, layers, params, net_size, value, delta); \
}
PAIRS(float, f32)
PAIRS(double, f64)
"#;

/// Shared memory one block may use without opting in to more.
const SHARED_LIMIT: usize = 48 * 1024;
const MAX_WARPS: usize = 8;
/// Threads per row of [`GradientNetwork`].
const GRADIENT_BLOCK: usize = 256;

/// Host-side parameters of one dense layer; `weights` is row-major
/// `[input][output]`.
#[derive(Clone, Debug)]
pub struct LayerParameters<T> {
    pub input: usize,
    pub output: usize,
    pub weights: Vec<T>,
    pub bias: Vec<T>,
    pub timestep: Option<Vec<T>>,
    pub activation: Activation,
    pub residual: bool,
}

/// Checks that `networks` is a nonempty set of chains with identical layer
/// shapes, activations and residual flags, starting from `input` inputs when
/// given. Returns the shared layer list.
fn common_shape<T: Scalar>(
    networks: &[Vec<LayerParameters<T>>],
    input: Option<usize>,
) -> Result<&[LayerParameters<T>]> {
    let Some(first) = networks.first().filter(|n| !n.is_empty()) else {
        return Err(Error(
            "a fused network set needs at least one network with one layer".into(),
        ));
    };
    if input.is_some_and(|expected| first[0].input != expected) {
        return Err(Error(format!(
            "network takes {} inputs per row, expected {}",
            first[0].input,
            input.unwrap()
        )));
    }
    let mut previous = first[0].input;
    for layer in first {
        if layer.input != previous {
            return Err(Error(format!(
                "layer input {} does not match previous output {previous}",
                layer.input
            )));
        }
        previous = layer.output;
    }
    for network in networks {
        if network.len() != first.len() {
            return Err(Error("networks of a set must have the same depth".into()));
        }
        for (layer, reference) in network.iter().zip(first) {
            validate(
                layer.input,
                layer.output,
                layer.weights.len(),
                layer.bias.len(),
                layer.timestep.as_ref().map(Vec::len),
                layer.residual,
            )?;
            if (layer.input, layer.output, layer.activation, layer.residual)
                != (
                    reference.input,
                    reference.output,
                    reference.activation,
                    reference.residual,
                )
            {
                return Err(Error(
                    "networks of a set must have identical layer shapes, activations and residuals"
                        .into(),
                ));
            }
        }
    }
    Ok(first)
}

fn timestep_or_ones<T: Scalar>(layer: &LayerParameters<T>) -> Vec<T> {
    layer
        .timestep
        .clone()
        .unwrap_or_else(|| vec![T::ONE; layer.output])
}

fn check_ids(ids: Option<usize>, rows: usize, networks: usize) -> Result<()> {
    match ids {
        Some(len) => check_len("network ids", len, rows),
        None if networks == 1 => Ok(()),
        None => Err(Error(format!(
            "a set of {networks} networks needs one network id per row"
        ))),
    }
}

/// Same-shaped networks with one input per row, evaluated with their
/// derivative.
pub struct ScalarNetwork<T: Scalar> {
    networks: usize,
    output: usize,
    layers: i32,
    net_size: i32,
    width: usize,
    warps: usize,
    table: CudaSlice<i32>,
    params: CudaSlice<T>,
    function: CudaFunction,
}

impl<T: Scalar> ScalarNetwork<T> {
    /// A single network.
    pub fn new(device: &Device, layers: &[LayerParameters<T>]) -> Result<Self> {
        Self::new_set(device, &[layers.to_vec()])
    }

    /// Networks with identical layer shapes; rows select one by index.
    pub fn new_set(device: &Device, networks: &[Vec<LayerParameters<T>>]) -> Result<Self> {
        let shape = common_shape(networks, Some(1))?;
        let mut table = Vec::with_capacity(7 * shape.len());
        let mut offset = 0usize;
        let mut width = 1;
        for layer in shape {
            let weights = offset;
            let bias = weights + layer.input * layer.output;
            let timestep = bias + layer.output;
            offset = timestep + layer.output;
            table.extend([
                layer.input as i32,
                layer.output as i32,
                layer.activation.code(),
                i32::from(layer.residual),
                weights as i32,
                bias as i32,
                timestep as i32,
            ]);
            width = width.max(layer.output);
        }
        if i32::try_from(offset * networks.len()).is_err() {
            return Err(Error("scalar network set has too many parameters".into()));
        }
        let mut params: Vec<T> = Vec::with_capacity(offset * networks.len());
        for network in networks {
            for layer in network {
                params.extend_from_slice(&layer.weights);
                params.extend_from_slice(&layer.bias);
                params.extend(timestep_or_ones(layer));
            }
        }
        let warp_bytes = 4 * width * std::mem::size_of::<T>();
        if warp_bytes > SHARED_LIMIT {
            return Err(Error(format!(
                "scalar network width {width} needs {warp_bytes} bytes of shared memory per row; the limit is {SHARED_LIMIT}"
            )));
        }
        Ok(Self {
            networks: networks.len(),
            output: shape.last().unwrap().output,
            layers: shape.len() as i32,
            net_size: offset as i32,
            width,
            warps: (SHARED_LIMIT / warp_bytes).clamp(1, MAX_WARPS),
            table: device.upload(&table)?,
            params: device.upload(&params)?,
            function: device.kernels().function(
                &format!("{ACTIVATE}{SCALAR_SOURCE}"),
                &format!("scalar_network_{}", T::SUFFIX),
            )?,
        })
    }

    pub fn output(&self) -> usize {
        self.output
    }

    pub fn networks(&self) -> usize {
        self.networks
    }

    /// `out[rows][output]` and its derivative with respect to `x[rows]`, for
    /// a single network.
    pub fn forward_tangent_into(
        &self,
        device: &Device,
        x: &impl DevicePtr<T>,
        rows: usize,
        out: &mut impl DevicePtrMut<T>,
        dout: &mut impl DevicePtrMut<T>,
    ) -> Result<()> {
        self.launch(device, x, None::<&CudaSlice<i32>>, rows, out, dout)
    }

    /// Like [`Self::forward_tangent_into`], with row `r` evaluated by network
    /// `ids[r]` (which must be below [`Self::networks`]). A row with a
    /// negative id is skipped: its outputs are left as they were.
    pub fn forward_tangent_by_id(
        &self,
        device: &Device,
        x: &impl DevicePtr<T>,
        ids: &impl DevicePtr<i32>,
        rows: usize,
        out: &mut impl DevicePtrMut<T>,
        dout: &mut impl DevicePtrMut<T>,
    ) -> Result<()> {
        self.launch(device, x, Some(ids), rows, out, dout)
    }

    fn launch(
        &self,
        device: &Device,
        x: &impl DevicePtr<T>,
        ids: Option<&impl DevicePtr<i32>>,
        rows: usize,
        out: &mut impl DevicePtrMut<T>,
        dout: &mut impl DevicePtrMut<T>,
    ) -> Result<()> {
        check_len("scalar network input", x.len(), rows)?;
        check_len("scalar network output", out.len(), rows * self.output)?;
        check_len("scalar network tangent", dout.len(), rows * self.output)?;
        check_ids(ids.map(|i| i.len()), rows, self.networks)?;
        let blocks = rows.div_ceil(self.warps);
        if rows == 0 || blocks > u32::MAX as usize {
            return Err(Error(format!(
                "scalar network over {rows} rows is out of range"
            )));
        }
        let stream = device.stream();
        let (px, _g0) = x.device_ptr(stream);
        let ids = ids.map(|i| i.device_ptr(stream));
        let pids = ids.as_ref().map_or(0, |(pointer, _)| *pointer);
        let (pout, _g1) = out.device_ptr_mut(stream);
        let (pdout, _g2) = dout.device_ptr_mut(stream);
        let (rows_i, width_i) = (rows as i64, self.width as i32);
        let mut launch = stream.launch_builder(&self.function);
        launch
            .arg(&px)
            .arg(&pids)
            .arg(&rows_i)
            .arg(&self.table)
            .arg(&self.layers)
            .arg(&self.params)
            .arg(&self.net_size)
            .arg(&width_i)
            .arg(&pout)
            .arg(&pdout);
        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (32 * self.warps as u32, 1, 1),
            shared_mem_bytes: (self.warps * 4 * self.width * std::mem::size_of::<T>()) as u32,
        };
        unsafe { launch.launch(config) }
            .map(drop)
            .map_err(cuda_error("scalar network launch"))
    }
}

/// A first layer with at least this many weights goes through tiles (when
/// the precision is float32: double precision is bound by its arithmetic,
/// not by reading weights).
const TILED_WEIGHTS: usize = 1 << 16;
/// Most device bytes the partial sums of a tiled first layer may take; past
/// that the rows go through the kernel without tiles.
const PARTIAL_LIMIT: usize = 256 << 20;
/// Row groupings a network keeps device tables for.
const GROUPINGS: usize = 4;

/// How a network's first layer is cut into tiles (see the tile kernels).
struct Tiling {
    forward: CudaFunction,
    middle: CudaFunction,
    backward: CudaFunction,
    /// Inputs per forward tile; rows per forward chunk (even: a warp does two).
    forward_inputs: usize,
    forward_rows: usize,
    /// Inputs per backward tile (one lane each); rows per backward chunk (a
    /// warp does two).
    backward_inputs: usize,
    backward_rows: usize,
    /// Outputs of the first layer.
    width: usize,
    /// Shared memory of the kernel for the other layers, bytes.
    middle_shared: usize,
    /// That kernel with two rows a block and its shared memory in bytes,
    /// when two rows fit.
    pairs: Option<(CudaFunction, usize)>,
}

impl Tiling {
    /// The tile sizes for `shape`, or `None` when its first layer is not one
    /// for tiles: no hidden layer, a residual first layer (it needs its input
    /// row again), or outputs too wide for a tile in shared memory.
    fn plan<T: Scalar>(
        device: &Device,
        shape: &[LayerParameters<T>],
        middle_shared: usize,
    ) -> Result<Option<Self>> {
        let first = &shape[0];
        if shape.len() < 2 || first.residual {
            return Ok(None);
        }
        let size = std::mem::size_of::<T>();
        let width = first.output;
        let padded = width.next_multiple_of(32);
        // Forward: the tile's weights get three quarters of shared memory at
        // most, the rows' inputs the rest.
        let forward_inputs = (SHARED_LIMIT * 3 / 4 / (padded * size)).min(32);
        if forward_inputs < 4 {
            return Ok(None);
        }
        let room = SHARED_LIMIT - forward_inputs * padded * size;
        let forward_rows = (room / (forward_inputs * size)).min(32) & !1;
        // Backward: a whole warp of inputs if the tile leaves a third of
        // shared memory to the rows' deltas, else a half or a quarter.
        let Some(backward_inputs) = [32, 16, 8]
            .into_iter()
            .find(|lanes| width * lanes * size <= SHARED_LIMIT * 2 / 3)
        else {
            return Ok(None);
        };
        let room = SHARED_LIMIT - width * backward_inputs * size;
        let backward_rows = (room / (width * size)).min(16);
        if forward_rows < 2 || backward_rows < 1 {
            return Ok(None);
        }
        let source = format!("{ACTIVATE}{GRADIENT_SOURCE}");
        let function = |name: &str| {
            device
                .kernels()
                .function(&source, &format!("{name}_{}", T::SUFFIX))
        };
        // Two rows a block: for each the slopes of the hidden layers and two
        // work rows, a partial sum per warp, a seed.
        let hidden = &shape[..shape.len() - 1];
        let row = hidden.iter().map(|l| l.output).sum::<usize>()
            + 2 * hidden.iter().map(|l| l.output).max().unwrap_or(0);
        let pair_bytes = (2 * (row + GRADIENT_BLOCK / 32 + 1) * size).next_multiple_of(8);
        let pairs = match pair_bytes <= SHARED_LIMIT {
            true => Some((function("gradient_pairs")?, pair_bytes)),
            false => None,
        };
        Ok(Some(Self {
            pairs,
            forward: function("tile_forward")?,
            middle: function("gradient_middle")?,
            backward: function("tile_backward")?,
            forward_inputs,
            forward_rows,
            backward_inputs,
            backward_rows,
            width,
            middle_shared,
        }))
    }
}

/// Device tables for rows that come in runs of one network each.
struct Grouping<T> {
    groups: Vec<(usize, usize, usize)>,
    /// `[rows]` network of each row.
    ids: CudaSlice<i32>,
    /// Chunks `{first row, rows, network}` for the forward and the backward
    /// tile kernel, with their counts.
    forward: (CudaSlice<i32>, usize),
    backward: (CudaSlice<i32>, usize),
    /// The same in chunks of two rows, for the layers after the first with
    /// two rows a block (networks whose rows fit, see `Tiling::pairs`).
    middle: Option<(CudaSlice<i32>, usize)>,
    /// `[rows][forward tiles][width]` and `[rows][width]`; empty without
    /// tiles.
    partial: Option<CudaSlice<T>>,
    delta: Option<CudaSlice<T>>,
}

/// Runs of rows cut into chunks of at most `rows` each: `{first, rows,
/// network}` per chunk.
fn chunks(groups: &[(usize, usize, usize)], rows: usize) -> Vec<i32> {
    let mut table = Vec::new();
    for &(network, first, count) in groups {
        for start in (0..count).step_by(rows) {
            let size = rows.min(count - start);
            table.extend([(first + start) as i32, size as i32, network as i32]);
        }
    }
    table
}

/// Same-shaped networks with one output per row, evaluated with the gradient
/// of that output with respect to the row's inputs.
pub struct GradientNetwork<T: Scalar> {
    networks: usize,
    input: usize,
    layers: i32,
    net_size: i32,
    adjoint: i32,
    shared_bytes: usize,
    table: CudaSlice<i32>,
    params: CudaSlice<T>,
    function: CudaFunction,
    /// The tile kernels and sizes, when the first layer suits them.
    tiling: Option<Tiling>,
    tiled: bool,
    /// Device tables of the row groupings seen last, most recent last.
    groupings: Vec<Grouping<T>>,
}

impl<T: Scalar> GradientNetwork<T> {
    /// Networks with identical layer shapes ending in one output; rows select
    /// one by index. Fails when a row's intermediates do not fit in shared
    /// memory (then the layer-by-layer path is the one to use).
    pub fn new_set(device: &Device, networks: &[Vec<LayerParameters<T>>]) -> Result<Self> {
        let shape = common_shape(networks, None)?;
        if shape.last().unwrap().output != 1 {
            return Err(Error(format!(
                "a gradient network ends in one output, got {}",
                shape.last().unwrap().output
            )));
        }
        let input = shape[0].input;
        let hidden = &shape[..shape.len() - 1];
        let widest = hidden.iter().map(|l| l.output).max().unwrap_or(0);
        // Shared memory in elements: input row, outputs and slopes of each
        // hidden layer, two adjoint buffers, per-warp partial sums, the seed.
        let mut shared = input;
        let mut offset = 0usize;
        let mut table = Vec::with_capacity(10 * shape.len() + 1);
        for (index, layer) in shape.iter().enumerate() {
            let weights = offset;
            let transposed = weights + layer.input * layer.output;
            let bias = transposed + layer.input * layer.output;
            let timestep = bias + layer.output;
            offset = timestep + layer.output;
            let (y, slope) = if index < hidden.len() {
                let y = shared;
                shared += 2 * layer.output;
                (y, y + layer.output)
            } else {
                (0, 0)
            };
            table.extend([
                layer.input as i32,
                layer.output as i32,
                layer.activation.code(),
                i32::from(layer.residual),
                weights as i32,
                transposed as i32,
                bias as i32,
                timestep as i32,
                y as i32,
                slope as i32,
            ]);
        }
        table.push(widest as i32);
        let adjoint = shared;
        shared += 2 * widest + GRADIENT_BLOCK / 32 + 1;
        let shared_bytes = (shared * std::mem::size_of::<T>()).next_multiple_of(8);
        if shared_bytes > SHARED_LIMIT {
            return Err(Error(format!(
                "gradient network needs {shared_bytes} bytes of shared memory per row; the limit is {SHARED_LIMIT}"
            )));
        }
        if i32::try_from(offset * networks.len()).is_err() {
            return Err(Error("gradient network set has too many parameters".into()));
        }
        let mut params: Vec<T> = Vec::with_capacity(offset * networks.len());
        for network in networks {
            for layer in network {
                params.extend_from_slice(&layer.weights);
                for n in 0..layer.output {
                    params.extend((0..layer.input).map(|k| layer.weights[k * layer.output + n]));
                }
                params.extend_from_slice(&layer.bias);
                params.extend(timestep_or_ones(layer));
            }
        }
        let size = std::mem::size_of::<T>();
        let middle_shared = ((shared - input) * size).next_multiple_of(8);
        let tiling = Tiling::plan(device, shape, middle_shared)?;
        Ok(Self {
            tiled: size == 4 && input * shape[0].output >= TILED_WEIGHTS,
            tiling,
            groupings: Vec::new(),
            networks: networks.len(),
            input,
            layers: shape.len() as i32,
            net_size: offset as i32,
            adjoint: adjoint as i32,
            shared_bytes,
            table: device.upload(&table)?,
            params: device.upload(&params)?,
            function: device.kernels().function(
                &format!("{ACTIVATE}{GRADIENT_SOURCE}"),
                &format!("gradient_network_{}", T::SUFFIX),
            )?,
        })
    }

    pub fn input(&self) -> usize {
        self.input
    }

    pub fn networks(&self) -> usize {
        self.networks
    }

    /// Whether rows given in groups go through the tiled first layer.
    pub fn tiled(&self) -> bool {
        self.tiled && self.tiling.is_some()
    }

    /// Turns the tiled first layer on or off for
    /// [`Self::forward_gradient_grouped`] and returns whether it is on
    /// afterwards: a network whose first layer does not suit tiles stays
    /// without. It is on by default for float32 networks with a first layer
    /// of at least 65,536 weights.
    pub fn set_tiled(&mut self, on: bool) -> bool {
        self.tiled = on;
        self.tiled()
    }

    /// [`Self::forward_gradient_into`] for rows sorted by network: `groups`
    /// lists `(network, first row, rows)` of consecutive runs that cover all
    /// rows. Knowing the runs, a wide first layer is evaluated in tiles: its
    /// weights are read from global memory once per tile and chunk of rows
    /// instead of once per row. The results agree with the other entry to
    /// rounding (the sums are formed in another order).
    pub fn forward_gradient_grouped(
        &mut self,
        device: &Device,
        x: &impl DevicePtr<T>,
        groups: &[(usize, usize, usize)],
        rows: usize,
        value: &mut impl DevicePtrMut<T>,
        gradient: &mut impl DevicePtrMut<T>,
    ) -> Result<()> {
        let mut next = 0;
        for &(network, first, count) in groups {
            if network >= self.networks || first != next || count == 0 {
                return Err(Error(format!(
                    "row groups must be consecutive nonempty runs of networks below {}; got ({network}, {first}, {count}) at row {next}",
                    self.networks
                )));
            }
            next += count;
        }
        if next != rows || rows == 0 || rows > i32::MAX as usize {
            return Err(Error(format!(
                "row groups cover {next} rows, expected {rows} (at least one)"
            )));
        }
        check_len("gradient network input", x.len(), rows * self.input)?;
        check_len("gradient network value", value.len(), rows)?;
        check_len(
            "gradient network gradient",
            gradient.len(),
            rows * self.input,
        )?;
        let index = match self.groupings.iter().position(|g| g.groups == groups) {
            Some(index) => index,
            None => {
                if self.groupings.len() == GROUPINGS {
                    self.groupings.remove(0);
                }
                self.groupings.push(self.grouping(device, groups, rows)?);
                self.groupings.len() - 1
            }
        };
        // Taken out while in use and put back last: the least recently used
        // grouping is the one that goes when the list is full.
        let mut grouping = self.groupings.remove(index);
        let result = self.run_grouped(device, &mut grouping, x, rows, value, gradient);
        self.groupings.push(grouping);
        result
    }

    /// The device tables for `groups`.
    fn grouping(
        &self,
        device: &Device,
        groups: &[(usize, usize, usize)],
        rows: usize,
    ) -> Result<Grouping<T>> {
        let ids: Vec<i32> = groups
            .iter()
            .flat_map(|&(network, _, count)| std::iter::repeat_n(network as i32, count))
            .collect();
        let tables = |rows_per_chunk: usize| -> Result<(CudaSlice<i32>, usize)> {
            let table = chunks(groups, rows_per_chunk);
            Ok((device.upload(&table)?, table.len() / 3))
        };
        // Without tiles the chunk tables are not read: one chunk per group.
        let (forward_rows, backward_rows) = self
            .tiling
            .as_ref()
            .map_or((rows, rows), |t| (t.forward_rows, t.backward_rows));
        let buffers = match &self.tiling {
            Some(tiling) => {
                let tiles = self.input.div_ceil(tiling.forward_inputs);
                let partial = rows * tiles * tiling.width;
                // More blocks than either tile kernel launches.
                let blocks = (rows + groups.len()) * self.input;
                if partial * std::mem::size_of::<T>() > PARTIAL_LIMIT || blocks > u32::MAX as usize
                {
                    None
                } else {
                    // SAFETY: the tile kernels write every element of the rows
                    // before the next kernel reads it.
                    Some(unsafe {
                        (
                            device.alloc_uninit::<T>(partial)?,
                            device.alloc_uninit::<T>(rows * tiling.width)?,
                        )
                    })
                }
            }
            None => None,
        };
        let (partial, delta) = buffers.map_or((None, None), |(p, d)| (Some(p), Some(d)));
        let middle = match self
            .tiling
            .as_ref()
            .and_then(|tiling| tiling.pairs.as_ref())
        {
            Some(_) => Some(tables(2)?),
            None => None,
        };
        Ok(Grouping {
            groups: groups.to_vec(),
            ids: device.upload(&ids)?,
            middle,
            forward: tables(forward_rows)?,
            backward: tables(backward_rows)?,
            partial,
            delta,
        })
    }

    /// The grouped evaluation proper: three launches with tiles (first layer
    /// forward, the other layers two rows a block, first layer backward), or
    /// the one kernel without.
    fn run_grouped(
        &self,
        device: &Device,
        grouping: &mut Grouping<T>,
        x: &impl DevicePtr<T>,
        rows: usize,
        value: &mut impl DevicePtrMut<T>,
        gradient: &mut impl DevicePtrMut<T>,
    ) -> Result<()> {
        let tiling = self.tiling.as_ref().filter(|_| self.tiled);
        let (Some(tiling), Some(partial), Some(delta)) =
            (tiling, &mut grouping.partial, &mut grouping.delta)
        else {
            let ids = Some(&grouping.ids);
            return self.forward_gradient_into(device, x, ids, rows, value, gradient);
        };
        let stream = device.stream();
        let size = std::mem::size_of::<T>();
        let (input, width) = (self.input, tiling.width);
        let (input_i, width_i) = (input as i32, width as i32);
        let (px, _g0) = x.device_ptr(stream);
        let (pvalue, _g1) = value.device_ptr_mut(stream);
        let (pgradient, _g2) = gradient.device_ptr_mut(stream);
        // The first layer's weights and their transpose lead a network's block.
        let (weights, transposed) = (0i32, (input * width) as i32);

        let tiles = input.div_ceil(tiling.forward_inputs);
        let tile_inputs = tiling.forward_inputs as i32;
        let shared = (tiling.forward_inputs * width.next_multiple_of(32)
            + tiling.forward_rows * tiling.forward_inputs)
            * size;
        let mut launch = stream.launch_builder(&tiling.forward);
        launch
            .arg(&px)
            .arg(&self.params)
            .arg(&grouping.forward.0)
            .arg(&input_i)
            .arg(&width_i)
            .arg(&tile_inputs)
            .arg(&self.net_size)
            .arg(&weights)
            .arg(&mut *partial);
        let config = LaunchConfig {
            grid_dim: ((grouping.forward.1 * tiles) as u32, 1, 1),
            block_dim: (16 * tiling.forward_rows as u32, 1, 1),
            shared_mem_bytes: shared.next_multiple_of(8) as u32,
        };
        unsafe { launch.launch(config) }
            .map(drop)
            .map_err(cuda_error("first layer tile launch"))?;

        let (tiles_i, rows_i) = (tiles as i32, rows as i64);
        let mut launch;
        let config = match (&tiling.pairs, &grouping.middle) {
            // Two rows a block: a weight read once serves both.
            (Some((kernel, shared)), Some((chunks, count))) => {
                launch = stream.launch_builder(kernel);
                launch
                    .arg(&*partial)
                    .arg(&tiles_i)
                    .arg(chunks)
                    .arg(&self.table)
                    .arg(&self.layers)
                    .arg(&self.params)
                    .arg(&self.net_size)
                    .arg(&pvalue)
                    .arg(&mut *delta);
                LaunchConfig {
                    grid_dim: (*count as u32, 1, 1),
                    block_dim: (GRADIENT_BLOCK as u32, 1, 1),
                    shared_mem_bytes: *shared as u32,
                }
            }
            _ => {
                launch = stream.launch_builder(&tiling.middle);
                launch
                    .arg(&*partial)
                    .arg(&tiles_i)
                    .arg(&grouping.ids)
                    .arg(&rows_i)
                    .arg(&self.table)
                    .arg(&self.layers)
                    .arg(&self.params)
                    .arg(&self.net_size)
                    .arg(&self.adjoint)
                    .arg(&pvalue)
                    .arg(&mut *delta);
                LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (GRADIENT_BLOCK as u32, 1, 1),
                    shared_mem_bytes: tiling.middle_shared as u32,
                }
            }
        };
        unsafe { launch.launch(config) }
            .map(drop)
            .map_err(cuda_error("gradient network launch"))?;

        let tiles = input.div_ceil(tiling.backward_inputs);
        let tile_inputs = tiling.backward_inputs as i32;
        let shared = (width * tiling.backward_inputs + tiling.backward_rows * width) * size;
        let mut launch = stream.launch_builder(&tiling.backward);
        launch
            .arg(&*delta)
            .arg(&self.params)
            .arg(&grouping.backward.0)
            .arg(&input_i)
            .arg(&width_i)
            .arg(&tile_inputs)
            .arg(&self.net_size)
            .arg(&transposed)
            .arg(&pgradient);
        let config = LaunchConfig {
            grid_dim: ((grouping.backward.1 * tiles) as u32, 1, 1),
            block_dim: (32 * tiling.backward_rows.div_ceil(2) as u32, 1, 1),
            shared_mem_bytes: shared.next_multiple_of(8) as u32,
        };
        unsafe { launch.launch(config) }
            .map(drop)
            .map_err(cuda_error("first layer tile launch"))
    }

    /// `value[rows]` and `gradient[rows][input]` for `x[rows][input]`; row `r`
    /// uses network `ids[r]` (`None` only for a single network).
    pub fn forward_gradient_into(
        &self,
        device: &Device,
        x: &impl DevicePtr<T>,
        ids: Option<&impl DevicePtr<i32>>,
        rows: usize,
        value: &mut impl DevicePtrMut<T>,
        gradient: &mut impl DevicePtrMut<T>,
    ) -> Result<()> {
        check_len("gradient network input", x.len(), rows * self.input)?;
        check_len("gradient network value", value.len(), rows)?;
        check_len(
            "gradient network gradient",
            gradient.len(),
            rows * self.input,
        )?;
        check_ids(ids.map(|i| i.len()), rows, self.networks)?;
        if rows == 0 || rows > u32::MAX as usize {
            return Err(Error(format!(
                "gradient network over {rows} rows is out of range"
            )));
        }
        let stream = device.stream();
        let (px, _g0) = x.device_ptr(stream);
        let ids = ids.map(|i| i.device_ptr(stream));
        let pids = ids.as_ref().map_or(0, |(pointer, _)| *pointer);
        let (pvalue, _g1) = value.device_ptr_mut(stream);
        let (pgradient, _g2) = gradient.device_ptr_mut(stream);
        let rows_i = rows as i64;
        let mut launch = stream.launch_builder(&self.function);
        launch
            .arg(&px)
            .arg(&pids)
            .arg(&rows_i)
            .arg(&self.table)
            .arg(&self.layers)
            .arg(&self.params)
            .arg(&self.net_size)
            .arg(&self.adjoint)
            .arg(&pvalue)
            .arg(&pgradient);
        let config = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (GRADIENT_BLOCK as u32, 1, 1),
            shared_mem_bytes: self.shared_bytes as u32,
        };
        unsafe { launch.launch(config) }
            .map(drop)
            .map_err(cuda_error("gradient network launch"))
    }
}
