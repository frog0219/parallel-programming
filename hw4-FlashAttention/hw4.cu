#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#define BLOCK_SIZE 32

void input(char *input_filename);
void output(char *output_filename);
void flash_attention(float *q, float *k, float *v, float *o);

__global__ void flash_attention_kenel(float *q, float *k, float *m, float *l, float *o, float *v, int d, float scalar, int grid_size);
__device__ void QKDotAndScalar(float *out, float *q, float *k, int d, float scalar);
__device__ void RowMax(float *out, float *in);
__device__ void MinusMaxAndExp(float *out, float *in, float *mx );
__device__ void RowSum(float *out, float *in);
__device__ void UpdateMiLiOi(float *shared_m, float *shared_l, float *o, float *mij, float *lij, float *pij, float *shared_v , int d);

float _max(float a, float b) { return a > b ? a : b; }
float _min(float a, float b) { return a < b ? a : b; }
double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

int B, N, d;
float *Q, *K, *V, *O;
float *device_q, *device_k, *device_v, *device_o;
float *device_l, *device_m;
int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    input(argv[1]);

    double start, end;
    start = getTimeStamp();

    cudaMalloc(&device_q, N * d * sizeof(float));
    cudaMalloc(&device_k, N * d * sizeof(float));
    cudaMalloc(&device_v, N * d * sizeof(float));
    cudaMalloc(&device_o, N * d * sizeof(float));
    cudaMalloc(&device_l, N * sizeof(float));
    cudaMalloc(&device_m, N * sizeof(float));

    for (int i = 0; i < B; i++) {
        flash_attention(
            Q + (i * N * d), 
            K + (i * N * d), 
            V + (i * N * d), 
            O + (i * N * d)
        );
    }

    end = getTimeStamp();
    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    printf("Time: %.3f seconds\n", end - start);

    output(argv[2]);
    // cudaFree(device_q);
    // cudaFree(device_k);
    // cudaFree(device_v);
    // cudaFree(device_o);
    // cudaFree(device_l);
    // cudaFree(device_m);
    return 0;
}

void input(char *input_filename) {
    FILE *file = fopen(input_filename, "rb");

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));

    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    memset(O, 0x00, B * N * d * sizeof(float));

    fclose(file);
}

void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");

    fwrite(O, sizeof(float), B * N * d, file);

    free(Q);
    free(K);
    free(V);
    free(O);

    fclose(file);
}

void flash_attention(float *q, float *k, float *v, float *o) {
    // float *device_q, *device_k, *device_v, *device_o;
    // float *device_l, *device_m;
    float *l = (float *)malloc(N * sizeof(float));
    float *m = (float *)malloc(N * sizeof(float));

    memset(l, 0x00, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        m[i] = FLT_MIN;
    }

    // cudaMalloc(&device_q, N * d * sizeof(float));
    // cudaMalloc(&device_k, N * d * sizeof(float));
    // cudaMalloc(&device_v, N * d * sizeof(float));
    // cudaMalloc(&device_o, N * d * sizeof(float));
    // cudaMalloc(&device_l, N * sizeof(float));
    // cudaMalloc(&device_m, N * sizeof(float));


    cudaMemcpy(device_q, q, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_k, k, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_v, v, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_o, o, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_l, l, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_m, m, N * sizeof(float), cudaMemcpyHostToDevice);

    int grid_size = N / BLOCK_SIZE;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(grid_size);
   
    flash_attention_kenel<<< grid, block >>>(device_q, device_k, device_m, device_l , device_o, device_v, d , 1.0 / sqrt(d) , grid_size);
    
    cudaMemcpy(o, device_o, N * d * sizeof(float), cudaMemcpyDeviceToHost);
}
__global__ void flash_attention_kenel(float *q, float *k, float *m, float *l, float *o, float *v, int d, float scalar , int grid_size){
    __shared__ float shared_q[BLOCK_SIZE * 64];
    __shared__ float shared_k[BLOCK_SIZE * 64];
    __shared__ float shared_v[BLOCK_SIZE * 64];
    __shared__ float shared_m[BLOCK_SIZE];
    __shared__ float shared_l[BLOCK_SIZE];
    __shared__ float shared_o[BLOCK_SIZE * 64]; 
    __shared__ float sij[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float mij[BLOCK_SIZE];
    __shared__ float pij[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float lij[BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int start = blockIdx.x;
    int step = d / BLOCK_SIZE;

    for(int i = 0 ; i < step; i++){
        shared_q[ty * d + (i * BLOCK_SIZE + tx)] = q[(start * BLOCK_SIZE + ty) * d + (i * BLOCK_SIZE + tx)];
        shared_o[ty * d + (i * BLOCK_SIZE + tx)] = o[(start * BLOCK_SIZE + ty) * d + (i * BLOCK_SIZE + tx)];
        // shared_k[tx * d + (i * BLOCK_SIZE + ty)] = k[tx * d + (i * BLOCK_SIZE + ty) + j * BLOCK_SIZE * d];
        // shared_v[tx * d + (i * BLOCK_SIZE + ty)] = v[tx * d + (i * BLOCK_SIZE + ty) + j * BLOCK_SIZE * d];
    }
    shared_m[ty] = m[ty + start * BLOCK_SIZE];
    shared_l[ty] = l[ty + start * BLOCK_SIZE];

    for (int j = 0; j < grid_size; j++) {

        for(int i = 0 ; i < step; i++){
            shared_k[tx * d + (i * BLOCK_SIZE + ty)] = k[tx * d + (i * BLOCK_SIZE + ty) + j * BLOCK_SIZE * d];
            shared_v[tx * d + (i * BLOCK_SIZE + ty)] = v[tx * d + (i * BLOCK_SIZE + ty) + j * BLOCK_SIZE * d];
        }
         __syncthreads();

        QKDotAndScalar(sij , shared_q , shared_k , d , scalar);

        __syncthreads();

        RowMax(mij , sij);

        __syncthreads();

        MinusMaxAndExp(pij , sij , mij);

        __syncthreads();

        RowSum(lij , pij);

        __syncthreads();
        
        UpdateMiLiOi(shared_m, shared_l, shared_o, mij, lij, pij, shared_v, d);
    }
    for(int i = 0 ; i < step; i++){
        o[(start * BLOCK_SIZE + ty) * d + (i * BLOCK_SIZE + tx)] =  shared_o[ty * d + (i * BLOCK_SIZE + tx)];
    }
    // m[ty + start * BLOCK_SIZE] = shared_m[ty];
    // l[ty + start * BLOCK_SIZE] = shared_l[ty];
}
__device__ void QKDotAndScalar(float *out, float *q, float *k, int d, float scalar) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float result = 0.0F;

    for (int t = 0; t < d ; t++) {
        result += q[ty * d + t] * k[tx * d + t];
    }
    result *= scalar;
    out[ty * BLOCK_SIZE + tx] = result;
}

__device__ void RowMax(float *out, float *in) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if(tx == 0){
        float result = in[ty  * BLOCK_SIZE];
        for (int j = 0; j < BLOCK_SIZE; j++) {
            result = max(result, in[ty * BLOCK_SIZE + j]);
        }
        out[ty] = result;
    }
}

__device__ void MinusMaxAndExp(float *out, float *in, float *mx ) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    out[ty  * BLOCK_SIZE + tx] = exp(in[ty * BLOCK_SIZE + tx] - mx[ty]);
}
__device__ void RowSum(float *out, float *in) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if(tx == 0){
        float result = 0.0F;
        for (int j = 0; j < BLOCK_SIZE; j++) {
            result += in[ty * BLOCK_SIZE + j];
        }
        out[ty] = result;
    }
}

__device__ void UpdateMiLiOi(float *shared_m, float *shared_l, float *shared_o, float *mij, float *lij, float *pij, float *shared_v , int d) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
   
    __shared__ float new_m[BLOCK_SIZE];
    __shared__ float new_l[BLOCK_SIZE];

    float thread_l = shared_l[ty];
    float thread_m = shared_m[ty];
    float thread_mij = mij[ty];

    if(tx == 0){
        new_m[ty] = max(thread_m , thread_mij);
        new_l[ty] = exp(thread_m - new_m[ty]) * thread_l + exp(thread_mij - new_m[ty]) * lij[ty];
    }

    __syncthreads();

    for (int j = 0; j < d / BLOCK_SIZE; j++) {
        float pv = 0.0F;
        for (int t = 0; t < BLOCK_SIZE; t++) {
            pv += pij[ty * BLOCK_SIZE + t] * shared_v[t * d + (j * BLOCK_SIZE + tx)];
        }
        shared_o[ty * d + (j * BLOCK_SIZE + tx)] = (thread_l * exp(thread_m - new_m[ty]) * shared_o[ty * d + (j * BLOCK_SIZE + tx)] + exp(thread_mij - new_m[ty]) * pv) / new_l[ty];
    }

    if(tx == 0){
        shared_m[ty] = new_m[ty];
        shared_l[ty] = new_l[ty];
    }

}
