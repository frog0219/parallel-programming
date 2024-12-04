#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>
//======================
#define DEV_NO 0
#define BLOCK_SIZE 32
#define block_factor 64
const int INF = (1 << 30) - 1;

int n, m , matrix_size ,*dist;

__global__ void phase1(int *Dist, int Round, int matrix_size);
__global__ void phase2(int *Dist, int Round, int matrix_size);
__global__ void phase3(int *Dist, int Round, int matrix_size);
void input(char *filename);
void output(char *filename);
int ceil(int a, int b) { return (a + b - 1) / b; }

int main(int argc, char* argv[]) {
    int *d_dist;
    input(argv[1]);

    cudaHostRegister(dist, matrix_size * matrix_size * sizeof(int), cudaHostRegisterDefault);
    cudaMalloc(&d_dist, (size_t)sizeof(int) * matrix_size * matrix_size);
    cudaMemcpyAsync(d_dist, dist, (size_t)sizeof(int) * matrix_size * matrix_size, cudaMemcpyHostToDevice, 0);

    int grid_size = matrix_size  / block_factor;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);      
    dim3 grid2(2, grid_size);
    dim3 grid3(grid_size, grid_size);

    for (int r = 0; r < grid_size; ++r) {
        phase1<<<1, block>>>(d_dist, r, matrix_size);
        phase2<<<grid2, block>>>(d_dist, r, matrix_size);
        phase3<<<grid3, block>>>(d_dist, r, matrix_size);
    }

    cudaMemcpy(dist, d_dist, (size_t)sizeof(int) * matrix_size * matrix_size, cudaMemcpyDeviceToHost);
    output(argv[2]);

    // cudaFree(d_dist);
    //Free(dist);
    return 0;
}

void input(char *filename) {
    FILE* file = fopen(filename, "rb");

    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    matrix_size = ceil(n , block_factor) * block_factor;
    dist = (int *)malloc(matrix_size * matrix_size * sizeof(int));

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < matrix_size; ++i) {
        for (int j = 0; j < matrix_size; ++j) {
            if (i == j && i < n) dist[i * matrix_size + j] = 0;
            else dist[i * matrix_size + j] = INF;
        }
    }

    int pair[3 * m];
    fread(pair, sizeof(int), m * 3, file);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < m; ++i) {
        dist[pair[3 * i] * matrix_size + pair[3 * i + 1]] = pair[3 * i + 2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        fwrite(&dist[i * matrix_size], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

__global__ void phase1(int *d_dist, int Round, int matrix_size) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    __shared__ int sharedDist[2 * BLOCK_SIZE][2 * BLOCK_SIZE];

    int base_i = Round * 2 * BLOCK_SIZE + ty;
    int base_j = Round * 2 * BLOCK_SIZE + tx;

    sharedDist[ty][tx] = d_dist[base_i * matrix_size + base_j];
    sharedDist[ty][tx + BLOCK_SIZE] = d_dist[base_i * matrix_size + base_j + BLOCK_SIZE];
    sharedDist[ty + BLOCK_SIZE][tx] = d_dist[(base_i + BLOCK_SIZE) * matrix_size + base_j];
    sharedDist[ty + BLOCK_SIZE][tx + BLOCK_SIZE] = d_dist[(base_i + BLOCK_SIZE) * matrix_size + base_j + BLOCK_SIZE];

    __syncthreads();

    #pragma unroll 32
    for (int k = 0; k < 2 * BLOCK_SIZE ; ++k) {

        sharedDist[ty][tx] = min(sharedDist[ty][tx], sharedDist[ty][k] + sharedDist[k][tx]);
        sharedDist[ty][tx + BLOCK_SIZE] = min(sharedDist[ty][tx + BLOCK_SIZE], sharedDist[ty][k] + sharedDist[k][tx + BLOCK_SIZE]);
        sharedDist[ty + BLOCK_SIZE][tx] = min(sharedDist[ty + BLOCK_SIZE][tx], sharedDist[ty + BLOCK_SIZE][k] + sharedDist[k][tx]);
        sharedDist[ty + BLOCK_SIZE][tx + BLOCK_SIZE] = min(sharedDist[ty + BLOCK_SIZE][tx + BLOCK_SIZE], sharedDist[ty + BLOCK_SIZE][k] + sharedDist[k][tx + BLOCK_SIZE]);
        
        __syncthreads();

    }

    d_dist[base_i * matrix_size + base_j] = sharedDist[ty][tx];
    d_dist[base_i * matrix_size + base_j + BLOCK_SIZE] = sharedDist[ty][tx + BLOCK_SIZE];
    d_dist[(base_i + BLOCK_SIZE) * matrix_size + base_j] = sharedDist[ty + BLOCK_SIZE][tx];
    d_dist[(base_i + BLOCK_SIZE) * matrix_size + base_j + BLOCK_SIZE] = sharedDist[ty + BLOCK_SIZE][tx + BLOCK_SIZE];

}
__global__ void phase2(int *d_dist, int Round, int matrix_size) {
  
    int tx = threadIdx.x;    
    int ty = threadIdx.y;    
    int block_type = blockIdx.x; // (0 , 1) = (row , column)
    int block_index = blockIdx.y;
    int R = Round;
    if (block_index == R) return;
    
    __shared__ int sharedPivot[2 * BLOCK_SIZE][2 * BLOCK_SIZE];
    __shared__ int sharedBlock[2 * BLOCK_SIZE][2 * BLOCK_SIZE];

    #pragma unroll 4
    for(int i = 0 ; i < 2 ; i ++){
        for(int j = 0 ; j < 2 ; j++){
            int base_i, base_j;

            if (block_type == 0) {
                base_i = (2 * R + i) * BLOCK_SIZE + ty;    
                base_j = (2 * block_index + j)* BLOCK_SIZE + tx;
            }
            else{
                base_i = (2 * block_index + i) * BLOCK_SIZE + ty;
                base_j = (2 * R + j) * BLOCK_SIZE + tx;
            }
            sharedPivot[ty + BLOCK_SIZE * i][tx + BLOCK_SIZE * j] = d_dist[((2 * R + i) * BLOCK_SIZE + ty) * matrix_size + ((2 * R + j) * BLOCK_SIZE + tx)];
            sharedBlock[ty + BLOCK_SIZE * i][tx + BLOCK_SIZE * j] = d_dist[base_i * matrix_size + base_j];
        }
    }
   
    __syncthreads();

    #pragma unroll 32
    for (int k = 0; k < 2 * BLOCK_SIZE; ++k) {
        for(int i = 0 ; i < 2 ; i ++){
            for(int j = 0 ; j < 2 ; j++){
                if (block_type == 0) sharedBlock[ty + BLOCK_SIZE * i][tx + BLOCK_SIZE * j] 
                    = min(sharedBlock[ty + BLOCK_SIZE * i][tx + BLOCK_SIZE * j], sharedPivot[ty + BLOCK_SIZE * i][k] + sharedBlock[k][tx + BLOCK_SIZE * j]);
                else sharedBlock[ty + BLOCK_SIZE * i][tx + BLOCK_SIZE * j] 
                    = min(sharedBlock[ty + BLOCK_SIZE * i][tx + BLOCK_SIZE * j], sharedBlock[ty + BLOCK_SIZE * i][k] + sharedPivot[k][tx + BLOCK_SIZE * j]);
            }
        }
        __syncthreads();
    }
    #pragma unroll 4
     for(int i = 0 ; i < 2 ; i ++){
        for(int j = 0 ; j < 2 ; j++){
            int base_i, base_j;

            if (block_type == 0) {
                base_i = (2 * R + i) * BLOCK_SIZE + ty;    
                base_j = (2 * block_index + j)* BLOCK_SIZE + tx;
            }
            else{
                base_i = (2 * block_index + i) * BLOCK_SIZE + ty;
                base_j = (2 * R + j) * BLOCK_SIZE + tx;
            }
            d_dist[((2 * R + i) * BLOCK_SIZE + ty) * matrix_size + ((2 * R + j) * BLOCK_SIZE + tx)] = sharedPivot[ty + BLOCK_SIZE * i][tx + BLOCK_SIZE * j] ;
            d_dist[base_i * matrix_size + base_j] = sharedBlock[ty + BLOCK_SIZE * i][tx + BLOCK_SIZE * j];
        }
    }
    
}
__global__ void phase3(int *d_dist, int Round, int matrix_size) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int R = Round;
    if (block_row == R || block_col == R) return;

    __shared__ int sharedRow[BLOCK_SIZE * 2][BLOCK_SIZE * 2];
    __shared__ int sharedCol[BLOCK_SIZE * 2][BLOCK_SIZE * 2];

    #pragma unroll 4
    for(int i = 0 ; i < 2 ; i ++){
        for(int j = 0 ; j < 2 ; j++){
            int base_i = 2 * block_row  * BLOCK_SIZE + ty + i * BLOCK_SIZE;
            int base_j = 2 * block_col  * BLOCK_SIZE + tx + j * BLOCK_SIZE;

            int pivot_start = R * 2 * BLOCK_SIZE;
            
            sharedRow[ty + BLOCK_SIZE * i][tx + BLOCK_SIZE * j] = d_dist[base_i * matrix_size + (pivot_start + j * BLOCK_SIZE + tx)];
            sharedCol[ty + BLOCK_SIZE * i][tx + BLOCK_SIZE * j] = d_dist[(pivot_start + i * BLOCK_SIZE + ty) * matrix_size + base_j];
        }
    }
    
    int sharedBlock_1 = d_dist[((2 * block_row) * BLOCK_SIZE + ty) * matrix_size + (2 * block_col) * BLOCK_SIZE + tx];
    int sharedBlock_2 = d_dist[((2 * block_row) * BLOCK_SIZE + ty) * matrix_size + (2 * block_col + 1) * BLOCK_SIZE + tx];
    int sharedBlock_3 = d_dist[((2 * block_row + 1) * BLOCK_SIZE + ty) * matrix_size + (2 * block_col) * BLOCK_SIZE + tx];
    int sharedBlock_4 = d_dist[((2 * block_row + 1) * BLOCK_SIZE + ty) * matrix_size + (2 * block_col + 1) * BLOCK_SIZE + tx];
    
    __syncthreads();
    #pragma unroll 32
    for (int k = 0; k < 2 * BLOCK_SIZE ; ++k) {
        sharedBlock_1 = min(sharedBlock_1, sharedRow[ty][k] + sharedCol[k][tx]);
        sharedBlock_2 = min(sharedBlock_2, sharedRow[ty][k] + sharedCol[k][tx + BLOCK_SIZE]);
        sharedBlock_3 = min(sharedBlock_3, sharedRow[ty + BLOCK_SIZE][k] + sharedCol[k][tx]);
        sharedBlock_4 = min(sharedBlock_4, sharedRow[ty + BLOCK_SIZE][k] + sharedCol[k][tx + BLOCK_SIZE]);
    }
    
    d_dist[((2 * block_row) * BLOCK_SIZE + ty )* matrix_size + (2 * block_col) * BLOCK_SIZE + tx] = sharedBlock_1;
    d_dist[((2 * block_row) * BLOCK_SIZE + ty) * matrix_size + (2 * block_col + 1) * BLOCK_SIZE + tx] = sharedBlock_2;
    d_dist[((2 * block_row + 1) * BLOCK_SIZE + ty) * matrix_size + (2 * block_col) * BLOCK_SIZE + tx] = sharedBlock_3;
    d_dist[((2 * block_row + 1) * BLOCK_SIZE + ty) * matrix_size + (2 * block_col + 1) * BLOCK_SIZE + tx] = sharedBlock_4; 
}

