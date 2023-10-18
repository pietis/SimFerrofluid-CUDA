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
        double Hnext = Hext[0] * normals[idx * 3 + 0] +
                       Hext[1] * normals[idx * 3 + 1] +
                       Hext[2] * normals[idx * 3 + 2];
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

__device__ void warp_reduce(volatile double *s_data, int idx) {
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
        double r[3];
        r[0] = positions[jidx * 3 + 0] - positions[iidx * 3 + 0];
        r[1] = positions[jidx * 3 + 1] - positions[iidx * 3 + 1];
        r[2] = positions[jidx * 3 + 2] - positions[iidx * 3 + 2];
        double rd = r[0] * normals[iidx * 3 + 0] +
                    r[1] * normals[iidx * 3 + 1] + r[2] * normals[iidx * 3 + 2];
        double rs = norm3d(r[0], r[1], r[2]);
        rs = rs * rs * rs;
        rs = max(rs, eps);
        const double C = 1.0 / (4.0 * 3.141592653589783);
        double dG = C * rd / rs;
        sdata[tx + ty * BLOCK_SIZE] = dG * areas[jidx] * utmp[jidx];
    }
    __syncthreads();
    if (tx < 16) {
        warp_reduce(sdata + ty * BLOCK_SIZE, tx);
    }
    __syncthreads();
    if (tx == 0 && iidx < size) {
        atomicAdd(u + iidx, 2 * lambda * sdata[ty * BLOCK_SIZE]);
    }
}

__global__ void init_Ht(double *__restrict__ Ht1, double *__restrict__ Ht2,
                        const double *__restrict__ Hext,
                        const double *__restrict__ normals, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double itx1[3];
    double itx2[3];
    if (idx < size) {
        double normals_[3];
        normals_[0] = normals[idx * 3 + 0];
        normals_[1] = normals[idx * 3 + 1];
        normals_[2] = normals[idx * 3 + 2];

        double x, y, z, rn;
        if (abs(normals_[0]) > 0.1) {
            x = -normals_[2];
            y = 0;
            z = normals_[0];
        } else {
            x = 0;
            y = normals_[2];
            z = -normals_[1];
        }
        rn = rnorm3d(x, y, z);
        itx1[0] = x * rn;
        itx1[1] = y * rn;
        itx1[2] = z * rn;

        x = normals_[1] * itx1[2] - normals_[2] * itx1[1];
        y = normals_[2] * itx1[0] - normals_[0] * itx1[2];
        z = normals_[0] * itx1[1] - normals_[1] * itx1[0];

        rn = rnorm3d(x, y, z);
        itx2[0] = x * rn;
        itx2[1] = y * rn;
        itx2[2] = z * rn;

        Ht1[idx] = Hext[0] * itx1[0] + Hext[1] * itx1[1] + Hext[2] * itx1[2];
        Ht2[idx] = Hext[0] * itx2[0] + Hext[1] * itx2[1] + Hext[2] * itx2[2];
    }
}

template <int BLOCK_SIZE>
__global__ void
compute_Ht(double *__restrict__ Ht1, double *__restrict__ Ht2,
           const double *__restrict__ u, const double *__restrict__ positions,
           const double *__restrict__ normals, const double *__restrict__ areas,
           const double eps, const int size) {
    int iidx = blockIdx.y * blockDim.y + threadIdx.y;
    int jidx = blockIdx.x * blockDim.x + threadIdx.x;
    int iidx_ = blockIdx.y * blockDim.y + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ double ipositions[BLOCK_SIZE * 3];
    __shared__ double jpositions[BLOCK_SIZE * 3];
    __shared__ double itx1[BLOCK_SIZE * 3];
    __shared__ double itx2[BLOCK_SIZE * 3];
    __shared__ double jareas[BLOCK_SIZE];
    __shared__ double ju[BLOCK_SIZE];
    __shared__ double sdata1[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ double sdata2[BLOCK_SIZE * BLOCK_SIZE];

    sdata1[tx + ty * BLOCK_SIZE] = 0;
    sdata2[tx + ty * BLOCK_SIZE] = 0;

    if (ty == 0 && jidx < size) {
        jpositions[tx * 3 + 0] = positions[jidx * 3 + 0];
        jpositions[tx * 3 + 1] = positions[jidx * 3 + 1];
        jpositions[tx * 3 + 2] = positions[jidx * 3 + 2];
    }

    if (ty == 1 && jidx < size) {
        jareas[tx] = areas[jidx];
    }

    if (ty == 2 && jidx < size) {
        ju[tx] = u[jidx];
    }

    if (ty == 3 && iidx_ < size) {
        ipositions[tx * 3 + 0] = positions[iidx_ * 3 + 0];
        ipositions[tx * 3 + 1] = positions[iidx_ * 3 + 1];
        ipositions[tx * 3 + 2] = positions[iidx_ * 3 + 2];
    }

    if (ty == 4 && iidx_ < size) {
        double normals_[3];
        normals_[0] = normals[iidx_ * 3 + 0];
        normals_[1] = normals[iidx_ * 3 + 1];
        normals_[2] = normals[iidx_ * 3 + 2];

        double x, y, z, rn;
        if (abs(normals_[0]) > 0.1) {
            x = -normals_[2];
            y = 0;
            z = normals_[0];
        } else {
            x = 0;
            y = normals_[2];
            z = -normals_[1];
        }
        rn = rnorm3d(x, y, z);
        itx1[tx * 3 + 0] = x * rn;
        itx1[tx * 3 + 1] = y * rn;
        itx1[tx * 3 + 2] = z * rn;

        x = normals_[1] * itx1[tx * 3 + 2] - normals_[2] * itx1[tx * 3 + 1];
        y = normals_[2] * itx1[tx * 3 + 0] - normals_[0] * itx1[tx * 3 + 2];
        z = normals_[0] * itx1[tx * 3 + 1] - normals_[1] * itx1[tx * 3 + 0];

        rn = rnorm3d(x, y, z);
        itx2[tx * 3 + 0] = x * rn;
        itx2[tx * 3 + 1] = y * rn;
        itx2[tx * 3 + 2] = z * rn;
    }
    __syncthreads();
    if (jidx < size && iidx < size && iidx != jidx) {
        double r[3];
        r[0] = jpositions[tx * 3 + 0] - ipositions[ty * 3 + 0];
        r[1] = jpositions[tx * 3 + 1] - ipositions[ty * 3 + 1];
        r[2] = jpositions[tx * 3 + 2] - ipositions[ty * 3 + 2];
        double rd = r[0] * itx1[ty * 3 + 0] + r[1] * itx1[ty * 3 + 1] +
                    r[2] * itx1[ty * 3 + 2];
        double rs = norm3d(r[0], r[1], r[2]);
        rs = rs * rs * rs;
        rs = max(rs, eps);
        const double C = 1.0 / (4.0 * 3.141592653589783);
        double dG = C * rd / rs;
        sdata1[tx + ty * BLOCK_SIZE] = -dG * jareas[tx] * ju[tx];

        rd = r[0] * itx2[ty * 3 + 0] + r[1] * itx2[ty * 3 + 1] +
             r[2] * itx2[ty * 3 + 2];
        dG = C * rd / rs;
        sdata2[tx + ty * BLOCK_SIZE] = -dG * jareas[tx] * ju[tx];
    }
    __syncthreads();
    if (tx < 16) {
        warp_reduce(sdata1 + ty * BLOCK_SIZE, tx);
        warp_reduce(sdata2 + ty * BLOCK_SIZE, tx);
    }
    __syncthreads();
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
        const double mu = 4e-7 * 3.141592653589783;
        double w = mu * (1 + chi) / (-chi) * u[idx];
        double Hn = -w / (mu * (1 + chi));
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
    static DeviceMem device_areas;
    static DeviceMem device_Hext;
    static DeviceMem device_pressures;
    static DeviceMem device_buffer0;
    static DeviceMem device_buffer1;
    static DeviceMem device_buffer2;
    device_positions.resize(size * 3, sizeof(double));
    device_normals.resize(size * 3, sizeof(double));
    device_areas.resize(size, sizeof(double));
    device_Hext.resize(3, sizeof(double));
    device_pressures.resize(size, sizeof(double));
    device_buffer0.resize(size, sizeof(double));
    device_buffer1.resize(size, sizeof(double));
    device_buffer2.resize(size, sizeof(double));
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
        double *Ht1 = utmp;
        double *Ht2 = b;
        int threadnum = 128;
        int blocknum = (size + threadnum - 1) / threadnum;
        init_Ht<<<blocknum, threadnum>>>(Ht1, Ht2, (double *)device_Hext.data(),
                                         (double *)device_normals.data(), size);
        const int BLOCK_SIZE = 32;
        dim3 grid_dim = dim3((size + BLOCK_SIZE - 1) / BLOCK_SIZE,
                             (size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block_dim = dim3(BLOCK_SIZE, BLOCK_SIZE);
        compute_Ht<BLOCK_SIZE><<<grid_dim, block_dim>>>(
            Ht1, Ht2, u, (double *)device_positions.data(),
            (double *)device_normals.data(), (double *)device_areas.data(),
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
};

} // namespace Pivot