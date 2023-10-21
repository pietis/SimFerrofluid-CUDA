#include "MagneticCUDA.h"

#include <cstdio>
#include <utility>
#include <vector>

namespace Pivot {

class DeviceMem {
  public:
    void resize(int size, int typesize) {
        auto next_power = [](int num) {
            num--;
            num |= num >> 1;
            num |= num >> 2;
            num |= num >> 4;
            num |= num >> 8;
            num |= num >> 16;
            num++;
            return num;
        };
        if (size * typesize > _size) {
            if (_size != 0) {
                cudaFree(_data);
            }
            int size_ceil = next_power(size);
            cudaMalloc(&_data, size_ceil * typesize);
            _size = size_ceil * typesize;
        }
    }
    ~DeviceMem() { cudaFree(_data); }
    void *data() { return _data; }

  private:
    void *_data = nullptr;
    int _size = 0;
};

__global__ void init_b(double *__restrict__ u, double *__restrict__ b,
                       const double *__restrict__ normals,
                       const double *__restrict__ Hext, const double lambda,
                       const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double3 normals_ = ((double3 *)normals)[idx];
        double Hnext =
            Hext[0] * normals_.x + Hext[1] * normals_.y + Hext[2] * normals_.z;
        u[idx] = -2 * lambda * Hnext;
        b[idx] = -2 * lambda * Hnext;
    }
}

__global__ void init_u(double *__restrict__ u, double *__restrict__ b,
                       const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        u[idx] = b[idx];
    }
}

__device__ void warp_reduce32(volatile double *s_data, int idx) {
    s_data[idx] += s_data[idx + 16];
    s_data[idx] += s_data[idx + 8];
    s_data[idx] += s_data[idx + 4];
    s_data[idx] += s_data[idx + 2];
    s_data[idx] += s_data[idx + 1];
}

__device__ void warp_reduce64(volatile double *s_data, int idx) {
    s_data[idx] += s_data[idx + 32];
    s_data[idx] += s_data[idx + 16];
    s_data[idx] += s_data[idx + 8];
    s_data[idx] += s_data[idx + 4];
    s_data[idx] += s_data[idx + 2];
    s_data[idx] += s_data[idx + 1];
}

template <int BLOCK_SIZE>
__global__ void
magnetic_iter(double *__restrict__ u, const double *__restrict__ utmp,
              const double *__restrict__ positions,
              const double *__restrict__ normals,
              const double *__restrict__ areas, const double lambda,
              const double eps, const int size) {
    int iidx = blockIdx.y * blockDim.y + threadIdx.y;
    int jidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    __shared__ double sdata[BLOCK_SIZE * BLOCK_SIZE];
    sdata[tx + ty * BLOCK_SIZE] = 0;
    if (jidx < size && iidx < size && iidx != jidx) {
        double3 pi;
        double3 pj;
        double3 ni;
        double3 r;
        pi = ((double3 *)positions)[iidx];
        pj = ((double3 *)positions)[jidx];
        ni = ((double3 *)normals)[iidx];
        r.x = pj.x - pi.x;
        r.y = pj.y - pi.y;
        r.z = pj.z - pi.z;
        double rd = r.x * ni.x + r.y * ni.y + r.z * ni.z;
        double rs = r.x * r.x + r.y * r.y + r.z * r.z;
        rs = (1 / rs) * rsqrt(rs);
        rs = min(rs, 1 / eps);
        // rs = (rs > 1 / eps) ? 0 : rs;
        constexpr double C = 1.0 / (4.0 * 3.141592653589793);
        double dG = C * rd * rs;
        sdata[tx + ty * BLOCK_SIZE] = dG * areas[jidx] * utmp[jidx];
    }
    __syncthreads();
    if (tx < 16) {
        warp_reduce32(sdata + ty * BLOCK_SIZE, tx);
    }
    if (tx == 0 && iidx < size) {
        atomicAdd(u + iidx, 2 * lambda * sdata[ty * BLOCK_SIZE]);
    }
}

__global__ void compute_residual(double *sum, const double *__restrict__ utmp,
                                 const double *__restrict__ u, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    __shared__ double sdata[1024];
    sdata[tx] = 0;
    if (idx < size) {
        double dif = abs(u[idx] - utmp[idx]);
        sdata[tx] = dif;
    }
    __syncthreads();
    if (tx < 512) {
        sdata[tx] += (tx + 512 < size) ? sdata[tx + 512] : 0;
    }
    __syncthreads();
    if (tx < 256) {
        sdata[tx] += (tx + 256 < size) ? sdata[tx + 256] : 0;
    }
    __syncthreads();
    if (tx < 128) {
        sdata[tx] += (tx + 128 < size) ? sdata[tx + 128] : 0;
    }
    __syncthreads();
    if (tx < 64) {
        sdata[tx] += (tx + 64 < size) ? sdata[tx + 64] : 0;
    }
    __syncthreads();
    if (tx < 32) {
        warp_reduce64(sdata, tx);
    }
    if (tx == 0) {
        atomicAdd(sum, sdata[0]);
    }
}

__global__ void init_Ht(double *__restrict__ Ht1, double *__restrict__ Ht2,
                        double *__restrict__ tangential1,
                        double *__restrict__ tangential2,
                        const double *__restrict__ Hext,
                        const double *__restrict__ normals, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double3 itx1;
    double3 itx2;
    if (idx < size) {
        double3 normals_;
        normals_ = ((double3 *)normals)[idx];

        double x, y, z, rn;
        if (abs(normals_.x) > 0.5) {
            x = -normals_.z;
            y = 0;
            z = normals_.x;
        } else {
            x = 0;
            y = normals_.z;
            z = -normals_.y;
        }
        rn = rnorm3d(x, y, z);
        itx1.x = x * rn;
        itx1.y = y * rn;
        itx1.z = z * rn;

        x = normals_.y * itx1.z - normals_.z * itx1.y;
        y = normals_.z * itx1.x - normals_.x * itx1.z;
        z = normals_.x * itx1.y - normals_.y * itx1.x;

        rn = rnorm3d(x, y, z);
        itx2.x = x * rn;
        itx2.y = y * rn;
        itx2.z = z * rn;

        Ht1[idx] = Hext[0] * itx1.x + Hext[1] * itx1.y + Hext[2] * itx1.z;
        Ht2[idx] = Hext[0] * itx2.x + Hext[1] * itx2.y + Hext[2] * itx2.z;

        ((double3 *)tangential1)[idx] = itx1;
        ((double3 *)tangential2)[idx] = itx2;
    }
}

template <int BLOCK_SIZE>
__global__ void
compute_Ht(double *__restrict__ Ht1, double *__restrict__ Ht2,
           const double *__restrict__ u, const double *__restrict__ positions,
           const double *__restrict__ normals,
           const double *__restrict__ tangential1,
           const double *__restrict__ tangential2,
           const double *__restrict__ areas, const double eps, const int size) {
    int iidx = blockIdx.y * blockDim.y + threadIdx.y;
    int jidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ double sdata1[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ double sdata2[BLOCK_SIZE * BLOCK_SIZE];

    sdata1[tx + ty * BLOCK_SIZE] = 0;
    sdata2[tx + ty * BLOCK_SIZE] = 0;
    if (jidx < size && iidx < size && iidx != jidx) {
        double3 pi;
        double3 pj;
        double3 ni;
        double3 ti1;
        double3 ti2;
        double3 r;
        pi = ((double3 *)positions)[iidx];
        pj = ((double3 *)positions)[jidx];
        ni = ((double3 *)normals)[iidx];
        ti1 = ((double3 *)tangential1)[iidx];
        ti2 = ((double3 *)tangential2)[iidx];
        r.x = pj.x - pi.x;
        r.y = pj.y - pi.y;
        r.z = pj.z - pi.z;
        double rd = r.x * ti1.x + r.y * ti1.y + r.z * ti1.z;
        double rs = r.x * r.x + r.y * r.y + r.z * r.z;
        rs = (1 / rs) * rsqrt(rs);
        rs = min(rs, 1 / eps);
        // rs = (rs > 1 / eps) ? 0 : rs;
        constexpr double C = 1.0 / (4.0 * 3.141592653589793);
        double dG = C * rd * rs;
        sdata1[tx + ty * BLOCK_SIZE] = -dG * areas[jidx] * u[jidx];

        rd = r.x * ti2.x + r.y * ti2.y + r.z * ti2.z;
        dG = C * rd * rs;
        sdata2[tx + ty * BLOCK_SIZE] = -dG * areas[jidx] * u[jidx];
    }
    __syncthreads();
    if (tx < 16) {
        warp_reduce32(sdata1 + ty * BLOCK_SIZE, tx);
        warp_reduce32(sdata2 + ty * BLOCK_SIZE, tx);
    }
    if (tx == 0 && iidx < size) {
        atomicAdd(Ht1 + iidx, sdata1[ty * BLOCK_SIZE]);
        atomicAdd(Ht2 + iidx, sdata2[ty * BLOCK_SIZE]);
    }
}

__global__ void compute_pressure(double *__restrict__ pressures,
                                 const double *__restrict__ Ht1,
                                 const double *__restrict__ Ht2,
                                 const double *__restrict__ u, const double chi,
                                 const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const double mu = 4e-7 * 3.141592653589793;
        double Hn = 1 / chi * u[idx];
        double Hn_ = Hn * (1 + chi);
        double Ht_squared = Ht1[idx] * Ht1[idx] + Ht2[idx] * Ht2[idx];
        double pressure = 0;
        pressure += mu * (1 + chi) * 0.5 * (Hn * Hn - Ht_squared);
        pressure -= mu * 0.5 * (Hn_ * Hn_ - Ht_squared);
        pressures[idx] = pressure;
    }
}

void SolveMagneticCUDA(const void *positions, const void *normals,
                       const void *areas, const void *Hext, void *pressures,
                       const int size, const int num_iter, const double lambda,
                       const double chi, const double epsilon) {
    static DeviceMem device_positions;
    static DeviceMem device_normals;
    static DeviceMem device_tangential1;
    static DeviceMem device_tangential2;
    static DeviceMem device_areas;
    static DeviceMem device_Hext;
    static DeviceMem device_pressures;
    static DeviceMem device_buffer0;
    static DeviceMem device_buffer1;
    static DeviceMem device_buffer2;
    static DeviceMem device_buffer3;
    device_positions.resize(size * 3, sizeof(double));
    device_normals.resize(size * 3, sizeof(double));
    device_tangential1.resize(size * 3, sizeof(double));
    device_tangential2.resize(size * 3, sizeof(double));
    device_areas.resize(size, sizeof(double));
    device_Hext.resize(3, sizeof(double));
    device_pressures.resize(size, sizeof(double));
    device_buffer0.resize(size, sizeof(double));
    device_buffer1.resize(size, sizeof(double));
    device_buffer2.resize(size, sizeof(double));
    device_buffer3.resize(1, sizeof(double));
    cudaMemset(device_buffer3.data(), 0, sizeof(double));
    cudaMemcpy(device_positions.data(), positions, size * 3 * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_normals.data(), normals, size * 3 * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_areas.data(), areas, size * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_Hext.data(), Hext, 3 * sizeof(double),
               cudaMemcpyHostToDevice);
    double *b = (double *)device_buffer0.data();
    double *u = (double *)device_buffer1.data();
    double *utmp = (double *)device_buffer2.data();
    {
        int threadnum = 128;
        int blocknum = (size + threadnum - 1) / threadnum;
        init_b<<<blocknum, threadnum>>>(u, b, (double *)device_normals.data(),
                                        (double *)device_Hext.data(), lambda,
                                        size);
    }
    {
        for (int iter = 0; iter < num_iter; iter++) {
            std::swap(u, utmp);
            int threadnum = 128;
            int blocknum = (size + threadnum - 1) / threadnum;
            init_u<<<blocknum, threadnum>>>(u, b, size);

            const int BLOCK_SIZE = 32;
            dim3 grid_dim = dim3((size + BLOCK_SIZE - 1) / BLOCK_SIZE,
                                 (size + BLOCK_SIZE - 1) / BLOCK_SIZE);
            dim3 block_dim = dim3(BLOCK_SIZE, BLOCK_SIZE);
            magnetic_iter<BLOCK_SIZE><<<grid_dim, block_dim>>>(
                u, utmp, (double *)device_positions.data(),
                (double *)device_normals.data(), (double *)device_areas.data(),
                lambda, epsilon, size);
        }
    }
    {
        int threadnum = 1024;
        int blocknum = (size + threadnum - 1) / threadnum;
        compute_residual<<<blocknum, threadnum>>>(
            (double *)device_buffer3.data(), utmp, u, size);
    }
    {
        double *Ht1 = utmp;
        double *Ht2 = b;
        int threadnum = 128;
        int blocknum = (size + threadnum - 1) / threadnum;
        init_Ht<<<blocknum, threadnum>>>(
            Ht1, Ht2, (double *)device_tangential1.data(),
            (double *)device_tangential2.data(), (double *)device_Hext.data(),
            (double *)device_normals.data(), size);
        const int BLOCK_SIZE = 32;
        dim3 grid_dim = dim3((size + BLOCK_SIZE - 1) / BLOCK_SIZE,
                             (size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block_dim = dim3(BLOCK_SIZE, BLOCK_SIZE);
        compute_Ht<BLOCK_SIZE><<<grid_dim, block_dim>>>(
            Ht1, Ht2, u, (double *)device_positions.data(),
            (double *)device_normals.data(),
            (double *)device_tangential1.data(),
            (double *)device_tangential2.data(), (double *)device_areas.data(),
            epsilon, size);
        compute_pressure<<<blocknum, threadnum>>>(
            (double *)device_pressures.data(), Ht1, Ht2, u, chi, size);
    }
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }
    cudaMemcpy(pressures, device_pressures.data(), size * sizeof(double),
               cudaMemcpyDeviceToHost);
    double sum = 0;
    cudaMemcpy(&sum, device_buffer3.data(), sizeof(double),
               cudaMemcpyDeviceToHost);
    sum /= size;

    printf("residual %.3e ", sum);
};

} // namespace Pivot