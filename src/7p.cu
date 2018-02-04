/* Available optimizations (value should be used as the first parameter in the command line):
   0 - Base -> no optimization
   1 - Sham -> shared memory
   2 - ZintReg -> for iteration on Z axis (Paulius)
   3 - Zint -> for iteration on Z axis without using registers
   4 - ShamZintReg -> shared memory + for iteration on Z axis
   5 - ShamZint -> shared memory + for iteration on Z axis without registers
   6 - ShamZintTempReg -> shared memory + for iteration on Z axis + temporal blocking
   7 - Roc -> use of read only cache (__restrict__ and const modifiers)
   8 - ShamRoc -> use of shared memory + read only cache (__restrict__ and const modifiers)
   9 - RocZintReg -> for iteration on Z axis + read only cache
   10 - RocZint -> for iteration on Z axis without registers + read only cache
   11 - ShamRocZintTempReg -> shared memory + read only cache + for iteration on Z axis + temporal blocking

   Known limitations: data grid size must be multiple of BLOCK_SIZE
*/

#include <stdio.h>

//#define PRINT_GOLD
//#define PRINT_RESULT

#define BLOCK_DIMX 32
#define BLOCK_DIMY 16
#define BLOCK_DIMZ 1
#define RADIUS 1 // Half of the order
#define PADDING_SIZE 32

// Error checking function
#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            printf("ERROR: Failed to run stmt %s\n", #stmt);                       \
            printf("ERROR: Got CUDA error ...  %s\n", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

__constant__ float coeff[RADIUS*6+1];

/* 
   Optimization Base: baseline code (no optimization)
*/
__global__ void calcStencilBase(float *a, float *b, int pitchedDimx, int dimy) {

  int tx = threadIdx.x + PADDING_SIZE;
  int ty = threadIdx.y + RADIUS;
  int tz = threadIdx.z + RADIUS;
	
  int row = blockIdx.y * blockDim.y + ty;
  int col = blockIdx.x * blockDim.x + tx;
  int depth = blockIdx.z * blockDim.z + tz;
	
  int stride = pitchedDimx * (dimy + 2*RADIUS); // 2D slice
  
  int index = (depth * stride) + (row * pitchedDimx) + col;

  // Compute stencil
  b[index] = coeff[0] * a[index] +
    coeff[1] * a[index - 1] +
    coeff[2] * a[index + 1] +
    coeff[3] * a[index - pitchedDimx] +
    coeff[4] * a[index + pitchedDimx] +
    coeff[5] * a[index - stride] +
    coeff[6] * a[index + stride];
}

/* 
   Optimization BaseCoeff: baseline code (no optimization) without using constant coeff
*/
__global__ void calcStencilBaseoeff(float *a, float *b, int pitchedDimx, int dimy, float c0, float c1, float c2, float c3, float c4, float c5, float c6) {

  int tx = threadIdx.x + PADDING_SIZE;
  int ty = threadIdx.y + RADIUS;
  int tz = threadIdx.z + RADIUS;
	
  int row = blockIdx.y * blockDim.y + ty;
  int col = blockIdx.x * blockDim.x + tx;
  int depth = blockIdx.z * blockDim.z + tz;
	
  int stride = pitchedDimx * (dimy + 2*RADIUS); // 2D slice
  
  int index = (depth * stride) + (row * pitchedDimx) + col;
  
  // Compute stencil
  b[index] = c0 * a[index] +
    c1 * a[index - 1] +
    c2 * a[index + 1] +
    c3 * a[index - pitchedDimx] +
    c4 * a[index + pitchedDimx] +
    c5 * a[index - stride] +
    c6 * a[index + stride];
}

/* 
   Optimization Sham: shared memory
*/
__global__ void calcStencilSham(float *a, float *b, int pitchedDimx, int dimy) {

  // Shared Memory Declaration
  __shared__ float ds_a[BLOCK_DIMY+2*RADIUS][BLOCK_DIMX+2*RADIUS];

  int tx = threadIdx.x + PADDING_SIZE;
  int sharedTx = threadIdx.x + RADIUS; // Index for shared memory (no padding)
  int ty = threadIdx.y + RADIUS;
  int tz = threadIdx.z + RADIUS;
	
  int row = blockIdx.y * blockDim.y + ty;
  int col = blockIdx.x * blockDim.x + tx;
  int depth = blockIdx.z * blockDim.z + tz;
  
  int stride = pitchedDimx * (dimy + 2*RADIUS); // 2D slice

  int index = (depth * stride) + (row * pitchedDimx) + col;

  // Load above/below halo data to shared memory
  if (threadIdx.y < RADIUS) {
    ds_a[threadIdx.y][sharedTx] = a[index - (RADIUS*pitchedDimx)];
    ds_a[threadIdx.y + BLOCK_DIMY + RADIUS][sharedTx] = a[index + (BLOCK_DIMY*pitchedDimx)];
  }

  // Load left/right halo data to shared memory
  if (threadIdx.x < RADIUS) {
    ds_a[ty][threadIdx.x] = a[index - RADIUS];
    ds_a[ty][threadIdx.x + BLOCK_DIMX + RADIUS] = a[index + BLOCK_DIMX];
  }

  // Load current position to shared memory
  ds_a[ty][sharedTx] = a[index];

  __syncthreads();

  // Compute stencil
  b[index] = coeff[0] * ds_a[ty][sharedTx] +
    coeff[1] * ds_a[ty][sharedTx - 1] +
    coeff[2] * ds_a[ty][sharedTx + 1] +
    coeff[3] * ds_a[ty - 1][sharedTx] +
    coeff[4] * ds_a[ty + 1][sharedTx] +
    coeff[5] * a[index - stride] +
    coeff[6] * a[index + stride];
}

/* 
   Optimization ZintReg: for iteration on Z axis with registers
*/
__global__ void calcStencilZintReg(float *a, float *b, int pitchedDimx, int dimy, int dimz) {

  int tx = threadIdx.x + PADDING_SIZE;
  int ty = threadIdx.y + RADIUS;

  int row = blockIdx.y * blockDim.y + ty;
  int col = blockIdx.x * blockDim.x + tx;
	
  int stride = pitchedDimx * (dimy + 2*RADIUS); // 2D slice

  int in_index = (row * pitchedDimx) + col; // Index for reading Z values

  int out_index = 0; // Index for writing output
    
  register float infront; // Variable to store the value in front (in the Z axis) of the current slice
  register float behind; // Variable to store the value behind (in the Z axis) the current slice
  register float current; // Input value in the current slice

  // Load initial values (behind will be loaded inside the next 'for')
  current = a[in_index];
  out_index = in_index;
  in_index += stride;
  
  infront = a[in_index];
  in_index += stride;

  // Iterate over the Z axis
  for (int i = 0; i < dimz; i++) {

    // Load the new values in Z axis
    behind = current;
    current = infront;
    infront = a[in_index];

    in_index += stride;
    out_index += stride;

    // Compute stencil
    b[out_index] = coeff[0] * current +
      coeff[1] * a[out_index - 1] +
      coeff[2] * a[out_index + 1] +
      coeff[3] * a[out_index - pitchedDimx] +
      coeff[4] * a[out_index + pitchedDimx] +
      coeff[5] * behind +
      coeff[6] * infront;
  }
}

/* 
   Optimization Zint: for iteration on Z axis without using registers
*/
__global__ void calcStencilZint(float *a, float *b, int pitchedDimx, int dimy, int dimz) {

  int tx = threadIdx.x + PADDING_SIZE;
  int ty = threadIdx.y + RADIUS;

  int row = blockIdx.y * blockDim.y + ty;
  int col = blockIdx.x * blockDim.x + tx;
	
  int stride = pitchedDimx * (dimy + 2*RADIUS); // 2D slice

  int out_index = (row * pitchedDimx) + col; // Index for writing output

  // Iterate over the Z axis
  for (int i = 0; i < dimz; i++) {

    out_index += stride;

    // Compute stencil
    b[out_index] = coeff[0] * a[out_index] +
      coeff[1] * a[out_index - 1] +
      coeff[2] * a[out_index + 1] +
      coeff[3] * a[out_index - pitchedDimx] +
      coeff[4] * a[out_index + pitchedDimx] +
      coeff[5] * a[out_index - stride] +
      coeff[6] * a[out_index + stride];
  }
}

/* 
   Optimization ShamZintReg: for iteration on Z axis + use of shared memory
*/
__global__ void calcStencilShamZintReg(float *a, float *b, int pitchedDimx, int dimy, int dimz) {

  // Shared memory declaration
  __shared__ float ds_a[BLOCK_DIMY+2*RADIUS][BLOCK_DIMX+2*RADIUS];

  int tx = threadIdx.x + PADDING_SIZE;
  int sharedTx = threadIdx.x + RADIUS; // Index for shared memory (no padding)
  int ty = threadIdx.y + RADIUS;

  int row = blockIdx.y * blockDim.y + ty;
  int col = blockIdx.x * blockDim.x + tx;
	
  int stride = pitchedDimx * (dimy + 2*RADIUS); // 2D slice

  int in_index = (row * pitchedDimx) + col; // Index for reading Z values
  int out_index = 0; // Index for writing output
    
  register float infront; // Variable to store the value in front (in the Z axis) of the current slice
  register float behind; // Variable to store the value behind (in the Z axis) the current slice
  register float current; // Input value in the current slice

  // Load initial values (behind will be loaded inside the next 'for')
  current = a[in_index];
  out_index = in_index;
  in_index += stride;
  
  infront = a[in_index];
  in_index += stride;

  // Iterate over the Z axis
  for (int i = 0; i < dimz; i++) {

    // Load the new values in Z axis
    behind = current;
    current = infront;
    infront = a[in_index];

    in_index += stride;
    out_index += stride;

    // Load above/below halo data to shared memory
    if (threadIdx.y < RADIUS) {
      ds_a[threadIdx.y][sharedTx] = a[out_index - (RADIUS * pitchedDimx)];
      ds_a[threadIdx.y + BLOCK_DIMY + RADIUS][sharedTx] = a[out_index + (pitchedDimx * BLOCK_DIMY)];
    }

    // Load left/right halo data to shared memory
    if (threadIdx.x < RADIUS) {
      ds_a[ty][threadIdx.x] = a[out_index - RADIUS];
      ds_a[ty][threadIdx.x + BLOCK_DIMX + RADIUS] = a[out_index + BLOCK_DIMX];
    }

    // Load current position to shared memory
    ds_a[ty][sharedTx] = current;

    __syncthreads();

    // Compute stencil (7 single precision mul + 6 single precision add)
    b[out_index] = coeff[0] * current +
      coeff[1] * ds_a[ty][sharedTx - 1] +
      coeff[2] * ds_a[ty][sharedTx + 1] +
      coeff[3] * ds_a[ty - 1][sharedTx] +
      coeff[4] * ds_a[ty + 1][sharedTx] +
      coeff[5] * behind +
      coeff[6] * infront;

    __syncthreads();
  
  }
}

/* 
   Optimization ShamZint: for iteration on Z axis without registers + use of shared memory
*/
__global__ void calcStencilShamZint(float *a, float *b, int pitchedDimx, int dimy, int dimz) {

  // Shared memory declaration
  __shared__ float ds_a[BLOCK_DIMY+2*RADIUS][BLOCK_DIMX+2*RADIUS];

  int tx = threadIdx.x + PADDING_SIZE;
  int sharedTx = threadIdx.x + RADIUS; // Index for shared memory (no padding)
  int ty = threadIdx.y + RADIUS;

  int row = blockIdx.y * blockDim.y + ty;
  int col = blockIdx.x * blockDim.x + tx;
	
  int stride = pitchedDimx * (dimy + 2*RADIUS); // 2D slice

  int out_index = (row * pitchedDimx) + col; // Index for writing output
    
  // Iterate over the Z axis
  for (int i = 0; i < dimz; i++) {

    out_index += stride;

    // Load above/below halo data to shared memory
    if (threadIdx.y < RADIUS) {
      ds_a[threadIdx.y][sharedTx] = a[out_index - (RADIUS * pitchedDimx)];
      ds_a[threadIdx.y + BLOCK_DIMY + RADIUS][sharedTx] = a[out_index + (pitchedDimx * BLOCK_DIMY)];
    }

    // Load left/right halo data to shared memory
    if (threadIdx.x < RADIUS) {
      ds_a[ty][threadIdx.x] = a[out_index - RADIUS];
      ds_a[ty][threadIdx.x + BLOCK_DIMX + RADIUS] = a[out_index + BLOCK_DIMX];
    }

    // Load current position to shared memory
    ds_a[ty][sharedTx] = a[out_index];

    __syncthreads();

    // Compute stencil (7 single precision mul + 6 single precision add)
    b[out_index] = coeff[0] * ds_a[ty][sharedTx] +
      coeff[1] * ds_a[ty][sharedTx - 1] +
      coeff[2] * ds_a[ty][sharedTx + 1] +
      coeff[3] * ds_a[ty - 1][sharedTx] +
      coeff[4] * ds_a[ty + 1][sharedTx] +
      coeff[5] * a[out_index - stride] +
      coeff[6] * a[out_index + stride]; 

    __syncthreads();
  }
}

/* 
   Optimization ShamZintTempReg: shared memory + for iteration on Z axis + temporal blocking (will always compute 2 time iterations)
*/
__global__ void calcStencilShamZintTempReg(float *a, float *b, int pitchedDimx, int dimy, int dimz) {

  // Shared memory declaration
  __shared__ float ds_a[BLOCK_DIMY+2*RADIUS][BLOCK_DIMX+2*RADIUS][2];

  int tx = threadIdx.x + PADDING_SIZE - RADIUS;
  int sharedTx = threadIdx.x + RADIUS; // Index for shared memory (no padding)
  int ty = threadIdx.y + RADIUS;

  int row = blockIdx.y * (BLOCK_DIMY-2*RADIUS) + ty;
  int col = blockIdx.x * (BLOCK_DIMX-2*RADIUS) + tx;

  int stride = pitchedDimx * (dimy + 4*RADIUS); // 2D slice
  
  int in_index = (row * pitchedDimx) + col; // Index for reading Z values
  int out_index = 0; // Index for writing output
  int next_index = 0; // Index for plane Z = output + RADIUS

  // t0 = t + 0
  register float t0_infront1; // Variable to store the value ahead (in the Z axis) of the current slice
  register float t0_behind1; // Variable to store the value behind (in the Z axis) the current slice
  register float t0_current; // Input value in the current slice

  // t1 = t + 1
  register float t1_infront1; // Variable to store the value ahead (in the Z axis) of the current slice
  register float t1_behind1; // Variable to store the value behind (in the Z axis) the current slice
  register float t1_current; // Value in current slice for t+1

  // Load ghost zones
  in_index += stride;
  t0_behind1 = a[in_index]; // Z = -R = -1
  in_index += stride;
  next_index = in_index; // Z = R-1 = 0
  
  out_index = in_index; // Index for writing output, Z = 0
  
  t0_current = a[in_index]; // Z = 0
  in_index += stride;
  t0_infront1 = a[in_index]; // Z = (2R-1) = 1
  in_index += stride;

  // Load Z = 0 to shared memory
  // Load above/below halo data
  if (threadIdx.y < RADIUS) {
    ds_a[threadIdx.y][sharedTx][1] = a[out_index - (RADIUS * pitchedDimx)];
    ds_a[threadIdx.y + BLOCK_DIMY + RADIUS][sharedTx][1] = a[out_index + (pitchedDimx * BLOCK_DIMY)];
  }
  
  // Load left/right halo data
  if (threadIdx.x < RADIUS) {
    ds_a[ty][threadIdx.x][1] = a[out_index - RADIUS];
    ds_a[ty][threadIdx.x + BLOCK_DIMX + RADIUS][1] = a[out_index + BLOCK_DIMX];
  }
  ds_a[ty][sharedTx][1] = t0_current;

  __syncthreads();

  // Compute stencil for Z = 0 (t + 1) but exclude ghost zones 
  if ( (row >= 2*RADIUS) && (row < (dimy+2*RADIUS)) && (col >= PADDING_SIZE) && (col < (pitchedDimx - PADDING_SIZE)) ) {
    t1_current = coeff[0] * t0_current +
     coeff[1] * ds_a[ty][sharedTx - 1][1] +
      coeff[2] * ds_a[ty][sharedTx + 1][1] +
      coeff[3] * ds_a[ty - 1][sharedTx][1] +
      coeff[4] * ds_a[ty + 1][sharedTx][1] +
      coeff[5] * t0_behind1 +
      coeff[6] * t0_infront1;
   } else {
     t1_current = t0_current;
   }
  
  // Copy planes Z = -1 to -R to registers in t+1 (ghost zones, keep values in 0.0)
  t1_behind1 = t0_behind1;
  
  __syncthreads();

  for (int i = 0; i < dimz; i++) {

    // Load Z = (2R+i) to registers
    t0_behind1 = t0_current;
    t0_current = t0_infront1;
    t0_infront1 = a[in_index]; // Z = 2R+i   

    in_index += stride;
    next_index += stride;
    
    // Load Z = R+i to shared memory
    if (threadIdx.y < RADIUS) {
      ds_a[threadIdx.y][sharedTx][1] = a[next_index - (RADIUS * pitchedDimx)];
      ds_a[threadIdx.y + BLOCK_DIMY + RADIUS][sharedTx][1] = a[next_index + (pitchedDimx * BLOCK_DIMY)];
    }
  
    // Load left/right halo data
    if (threadIdx.x < RADIUS) {
      ds_a[ty][threadIdx.x][1] = a[next_index - RADIUS];
      ds_a[ty][threadIdx.x + BLOCK_DIMX + RADIUS][1] = a[next_index + BLOCK_DIMX];
    }
    ds_a[ty][sharedTx][1] = t0_current;

    __syncthreads();

    // Compute stencil for Z = R+i (t + 1) but exclude ghost zones
    if ( (row >= 2*RADIUS) && (row < (dimy+2*RADIUS)) && (col >= PADDING_SIZE) && (col < (pitchedDimx - PADDING_SIZE)) && (i < dimz-RADIUS) ) {
      t1_infront1 = coeff[0] * t0_current +
	coeff[1] * ds_a[ty][sharedTx - 1][1] +
	coeff[2] * ds_a[ty][sharedTx + 1][1] +
	coeff[3] * ds_a[ty - 1][sharedTx][1] +
	coeff[4] * ds_a[ty + 1][sharedTx][1] +
	coeff[5] * t0_behind1 +
	coeff[6] * t0_infront1;
    } else {
      t1_infront1 = t0_current;
    }

    __syncthreads();

    // Load Z = k (t + 1) to shared memory
    ds_a[ty][sharedTx][0] = t1_current;

    __syncthreads();
    
    // Compute stencil for Z = k (t + 2) but exclude halo zones
    if ( (threadIdx.y >= RADIUS) && (threadIdx.y < (BLOCK_DIMY-RADIUS)) && (threadIdx.x >= RADIUS) && (threadIdx.x < (BLOCK_DIMX-RADIUS)) ) {    
      b[out_index] = coeff[0] * t1_current +
	coeff[1] * ds_a[ty][sharedTx - 1][0] +
	coeff[2] * ds_a[ty][sharedTx + 1][0] +
	coeff[3] * ds_a[ty - 1][sharedTx][0] +
	coeff[4] * ds_a[ty + 1][sharedTx][0] +
	coeff[5] * t1_behind1 +
	coeff[6] * t1_infront1;
    }

    out_index += stride;
    t1_behind1 = t1_current;
    t1_current = t1_infront1;
  }
}

/* 
   Optimization Roc: use of read only cache (texture memory)
*/
__global__ void calcStencilRoc(const float* __restrict__ a, float* __restrict__ b, int pitchedDimx, int dimy) {

  int tx = threadIdx.x + PADDING_SIZE;
  int ty = threadIdx.y + RADIUS;
  int tz = threadIdx.z + RADIUS;
	
  int row = blockIdx.y * blockDim.y + ty;
  int col = blockIdx.x * blockDim.x + tx;
  int depth = blockIdx.z * blockDim.z + tz;

  int stride = pitchedDimx * (dimy + 2*RADIUS); // 2D slice
  
  int index = (depth * stride) + (row * pitchedDimx) + col;
  
  // Compute stencil
  b[index] = coeff[0] * __ldg(&a[index]) +
    coeff[1] * __ldg(&a[index - 1]) +
    coeff[2] * __ldg(&a[index + 1]) +
    coeff[3] * __ldg(&a[index - pitchedDimx]) +
    coeff[4] * __ldg(&a[index + pitchedDimx]) +
    coeff[5] * __ldg(&a[index - stride]) +
    coeff[6] * __ldg(&a[index + stride]);
}


/* 
   Optimization ShamRoc: use of shared memory + read only cache (texture memory)
*/
__global__ void calcStencilShamRoc(const float* __restrict__ a, float* __restrict__ b, int pitchedDimx, int dimy) {

  // Shared Memory Declaration
  __shared__ float ds_a[BLOCK_DIMY+2*RADIUS][BLOCK_DIMX+2*RADIUS];

  int tx = threadIdx.x + PADDING_SIZE;
  int sharedTx = threadIdx.x + RADIUS; // Index for shared memory (no padding)
  int ty = threadIdx.y + RADIUS;
  int tz = threadIdx.z + RADIUS;
	
  int row = blockIdx.y * blockDim.y + ty;
  int col = blockIdx.x * blockDim.x + tx;
  int depth = blockIdx.z * blockDim.z + tz;

  int stride = pitchedDimx * (dimy + 2*RADIUS); // 2D slice

  int index = (depth * stride) + (row * pitchedDimx) + col;

  // Load above/below halo data to shared memory
  if (threadIdx.y < RADIUS) {
    ds_a[threadIdx.y][sharedTx] = __ldg(&a[index-(RADIUS*pitchedDimx)]);
    ds_a[threadIdx.y + BLOCK_DIMY + RADIUS][sharedTx] = __ldg(&a[index+(BLOCK_DIMY*pitchedDimx)]);
  }

  // Load left/right halo data to shared memory
  if (threadIdx.x < RADIUS) {
    ds_a[ty][threadIdx.x] = __ldg(&a[index-RADIUS]);
    ds_a[ty][threadIdx.x + BLOCK_DIMX + RADIUS] = __ldg(&a[index+BLOCK_DIMX]);
  }

  // Load current position to shared memory
  ds_a[ty][sharedTx] = __ldg(&a[index]);

  __syncthreads();

  // Compute stencil
  b[index] = coeff[0] * ds_a[ty][sharedTx] +
    coeff[1] * ds_a[ty][sharedTx - 1] +
    coeff[2] * ds_a[ty][sharedTx + 1] +
    coeff[3] * ds_a[ty - 1][sharedTx] +
    coeff[4] * ds_a[ty + 1][sharedTx] +
    coeff[5] * __ldg(&a[index - stride]) +
    coeff[6] * __ldg(&a[index + stride]);
}


/* 
   Optimization RocZintReg: use of iteration on Z axis + read only cache (texture memory)
*/
__global__ void calcStencilRocZintReg(const float* __restrict__ a, float* __restrict__ b, int pitchedDimx, int dimy, int dimz) {

  int tx = threadIdx.x + PADDING_SIZE;
  int ty = threadIdx.y + RADIUS;

  int row = blockIdx.y * blockDim.y + ty;
  int col = blockIdx.x * blockDim.x + tx;
		
  int stride = pitchedDimx * (dimy + 2*RADIUS); // 2D slice

  int in_index = (row * pitchedDimx) + col; // Index for reading Z values
  int out_index = 0; // Index for writing output
  
  register float infront; // Variable to store the value in front (in the Z axis) of the current slice
  register float behind; // Variable to store the value behind (in the Z axis) the current slice
  register float current; // Input value in the current slice

  // Load initial values (behind will be loaded inside the next 'for')
  current = __ldg(&a[in_index]);
  out_index = in_index;
  in_index += stride;
  
  infront = __ldg(&a[in_index]);
  in_index += stride;

  // Iterate over the Z axis
  for (int i = 0; i < dimz; i++) {

    // Load the new values in Z axis
    behind = current;
    current = infront;
    infront = __ldg(&a[in_index]);

    in_index += stride;
    out_index += stride;

    // Compute stencil
    b[out_index] = coeff[0] * current +
      coeff[1] * __ldg(&a[out_index - 1]) +
      coeff[2] * __ldg(&a[out_index + 1]) +
      coeff[3] * __ldg(&a[out_index - pitchedDimx]) +
      coeff[4] * __ldg(&a[out_index + pitchedDimx]) +
      coeff[5] * behind +
      coeff[6] * infront;
  }
}


/* 
   Optimization RocZint: use of iteration on Z axis without registers + read only cache (texture memory)
*/
__global__ void calcStencilRocZint(const float* __restrict__ a, float* __restrict__ b, int pitchedDimx, int dimy, int dimz) {

  int tx = threadIdx.x + PADDING_SIZE;
  int ty = threadIdx.y + RADIUS;

  int row = blockIdx.y * blockDim.y + ty;
  int col = blockIdx.x * blockDim.x + tx;
		
  int stride = pitchedDimx * (dimy + 2*RADIUS); // 2D slice

  int out_index = (row * pitchedDimx) + col; // Index for reading Z values
	  
  // Iterate over the Z axis
  for (int i = 0; i < dimz; i++) {

    out_index += stride;

    // Compute stencil
    b[out_index] = coeff[0] * __ldg(&a[out_index]) +
      coeff[1] * __ldg(&a[out_index - 1]) +
      coeff[2] * __ldg(&a[out_index + 1]) +
      coeff[3] * __ldg(&a[out_index - pitchedDimx]) +
      coeff[4] * __ldg(&a[out_index + pitchedDimx]) +
      coeff[5] * __ldg(&a[out_index - stride]) +
      coeff[6] * __ldg(&a[out_index + stride]);
  }
}


/* 
   Optimization ShamRocZintTempReg: shared memory + for iteration on Z axis + temporal blocking (will always compute 2 time iterations)
*/
__global__ void calcStencilShamRocZintTempReg(const float* __restrict__ a, float* __restrict__ b, int pitchedDimx, int dimy, int dimz) {

  // Shared memory declaration
  __shared__ float ds_a[BLOCK_DIMY][BLOCK_DIMX];

  int tx = threadIdx.x + PADDING_SIZE;
  int ty = threadIdx.y + RADIUS;

  int row = blockIdx.y * (BLOCK_DIMY-2*RADIUS) + ty;
  int col = blockIdx.x * (BLOCK_DIMX-2*RADIUS) + tx;
	
  int stride = pitchedDimx * (dimy + 4*RADIUS); // 2D slice
  
  int in_index = (row * pitchedDimx) + col; // Index for reading Z values
  int out_index = 0; // Index for writing output
  int next_index = 0; // Index for plane Z = output + RADIUS
  
  // t0 = t + 0
  register float t0_infront1; // Variable to store the value ahead (in the Z axis) of the current slice
  register float t0_behind1; // Variable to store the value behind (in the Z axis) the current slice
  register float t0_current; // Input value in the current slice

  // t1 = t + 1
  register float t1_infront1; // Variable to store the value ahead (in the Z axis) of the current slice
  register float t1_behind1; // Variable to store the value behind (in the Z axis) the current slice
  register float t1_current; // Value in current slice for t+1

  // Load ghost zones
  in_index += stride;
  t0_behind1 = __ldg(&a[in_index]); // Z = -R
  in_index += stride;
  next_index = in_index + ((RADIUS-1)*stride); // Z = R-1
  
  out_index = in_index; // Index for writing output, Z = 0
  
  // Load current to (2RADIUS-1) planes
  t0_current = __ldg(&a[in_index]); // Z = 0
  in_index += stride;
  t0_infront1 = __ldg(&a[in_index]); // Z = (2R-1) = 1
  in_index += stride;

  // Compute stencil for Z = 0 (t + 1) but exclude ghost zones 
  if ( (row >= 2*RADIUS) && (row < (dimy+2*RADIUS)) && (col >= (PADDING_SIZE)) && (col < (pitchedDimx - PADDING_SIZE)) ) {
    t1_current = coeff[0] * t0_current +
      coeff[1] * __ldg(&a[out_index - 1]) +
      coeff[2] * __ldg(&a[out_index + 1]) +
      coeff[3] * __ldg(&a[out_index - pitchedDimx]) +
      coeff[4] * __ldg(&a[out_index + pitchedDimx]) +
      coeff[5] * t0_behind1 +
      coeff[6] * t0_infront1;
   } else {
     t1_current = t0_current;
   }
  
  // Copy planes Z = -1 to -R to registers in t+1 (ghost zones, keep values in 0.0)
  t1_behind1 = t0_behind1;
  
  for (int i = 0; i < dimz; i++) {
    // Load Z = (2R+i) to registers
    t0_behind1 = t0_current;
    t0_current = t0_infront1;
    t0_infront1 = __ldg(&a[in_index]); // Z = 2R+i   

    in_index += stride;
    next_index += stride;
    
    // Compute stencil for Z = R+i (t + 1) but exclude ghost zones
    if ( (row >= 2*RADIUS) && (row < (dimy+2*RADIUS)) && (col >= PADDING_SIZE) && (col < (pitchedDimx - PADDING_SIZE)) && (i < dimz-RADIUS) ) {
      t1_infront1 = coeff[0] * t0_current +
	coeff[1] * __ldg(&a[next_index - 1]) +
	coeff[2] * __ldg(&a[next_index + 1]) +
	coeff[3] * __ldg(&a[next_index - pitchedDimx]) +
	coeff[4] * __ldg(&a[next_index + pitchedDimx]) +
	coeff[5] * t0_behind1 +
	coeff[6] * t0_infront1;
    } else {
      t1_infront1 = t0_current;
    }

    __syncthreads();

    // Load Z = k (t + 1) to shared memory
    ds_a[threadIdx.y][threadIdx.x] = t1_current;
    
    __syncthreads();

    // Compute stencil for Z = k (t + 2) but exclude halo zones
    if ( (threadIdx.y >= RADIUS) && (threadIdx.y < (BLOCK_DIMY-RADIUS)) && (threadIdx.x >= RADIUS) && (threadIdx.x < (BLOCK_DIMX-RADIUS)) ) {    
      b[out_index] = coeff[0] * t1_current +
	coeff[1] * ds_a[threadIdx.y][threadIdx.x - 1] +
	coeff[2] * ds_a[threadIdx.y][threadIdx.x + 1] +
	coeff[3] * ds_a[threadIdx.y - 1][threadIdx.x] +
	coeff[4] * ds_a[threadIdx.y + 1][threadIdx.x] +
	coeff[5] * t1_behind1 +
	coeff[6] * t1_infront1;
    }

    out_index += stride;
    t1_behind1 = t1_current;
    t1_current = t1_infront1;
  }
  
}


void initGold(float *a, int dimx, int dimy, int dimz, int pitchedDimx) {

  int stride = pitchedDimx * (dimy+2*RADIUS);
  int index = 0;
  
  for (int i = 0; i < (dimz+2*RADIUS); i++) {
    for (int j = 0; j < (dimy+2*RADIUS); j++) {
      for (int k = 0; k < pitchedDimx; k++) {
	index = i*stride + j*pitchedDimx + k;
	if (i<RADIUS || j<RADIUS || i>=dimz+RADIUS || j>=dimy+RADIUS || k<PADDING_SIZE || k>=dimx+PADDING_SIZE) {
	  a[index] = 0.0;
        } else {
	  a[index] = 1.0;
	}
      }
    }
  }
}

void initGoldTemporal(float *a, int dimx, int dimy, int dimz, int pitchedDimx) {

  int stride = pitchedDimx * (dimy+4*RADIUS);
  int index = 0;

  for (int i = 0; i < (dimz+4*RADIUS); i++) {
    for (int j = 0; j < (dimy+4*RADIUS); j++) {
      for (int k = 0; k < pitchedDimx; k++) {
	index = i*stride + j*pitchedDimx + k;
	if ( i<2*RADIUS || j<2*RADIUS || i>=dimz+2*RADIUS || j>=dimy+2*RADIUS || k<PADDING_SIZE || k>=dimx+PADDING_SIZE )  {
	  a[index] = 0.0;
	} else {
	  a[index] = 1.0;
	}
      }
    }
  }
}

void hostStencil(float *a, int t_end, int dimx, int dimy, int dimz, float *hcoeff, int pitchedDimx) {

  float *b;
  
  int stride = pitchedDimx * (dimy+2*RADIUS);

  b = (float *)malloc((dimz+2*RADIUS) * stride * sizeof(float));
  initGold(b, dimx, dimy, dimz, pitchedDimx);

  int index = 0;
 
  for (int t = 0; t < t_end; t++) {
    for (int i = RADIUS; i < dimz+RADIUS; i++) {
      for (int j = RADIUS; j < dimy+RADIUS; j++) {
	for (int k = PADDING_SIZE; k < dimx+PADDING_SIZE; k++) {
	  index = i*stride + j*pitchedDimx + k;
	  if (t%2) {
	    a[index] = hcoeff[0] * b[index] +
	      hcoeff[1] * b[index - 1] +
	      hcoeff[2] * b[index + 1] +
	      hcoeff[3] * b[index - pitchedDimx] +
	      hcoeff[4] * b[index + pitchedDimx] +
	      hcoeff[5] * b[index - stride] +
	      hcoeff[6] * b[index + stride];
	  } else {
	    b[index] = hcoeff[0] * a[index] +
	      hcoeff[1] * a[index - 1] +
	      hcoeff[2] * a[index + 1] +
	      hcoeff[3] * a[index - pitchedDimx] +
	      hcoeff[4] * a[index + pitchedDimx] +
	      hcoeff[5] * a[index - stride] +
	      hcoeff[6] * a[index + stride];
	  }
	}
      }
    }
  }  

  if (t_end%2) {
    for (int i = RADIUS; i < dimz+RADIUS; i++) {
      for (int j = RADIUS; j < dimy+RADIUS; j++) {
	for (int k = PADDING_SIZE; k < dimx+PADDING_SIZE; k++) {
	  index = i*stride + j*pitchedDimx + k;
	  a[index] = b[index];
	}
      }
    }    
  } 
  free(b);
}

void hostStencilTemporal(float *a, int t_end, int dimx, int dimy, int dimz, float *hcoeff, int pitchedDimx) {

  float *b;

  int stride = pitchedDimx * (dimy+4*RADIUS);
  
  b = (float *)malloc((dimz+4*RADIUS) * stride * sizeof(float));
  initGoldTemporal(b, dimx, dimy, dimz, pitchedDimx);

  int index = 0;
  
  for (int t = 0; t < t_end; t++) {
    for (int i = 2*RADIUS; i < dimz+2*RADIUS; i++) {
      for (int j = 2*RADIUS; j < dimy+2*RADIUS; j++) {
	for (int k = PADDING_SIZE; k < pitchedDimx-PADDING_SIZE; k++) {
	  index = i*stride + j*pitchedDimx + k;
	  if (t%2) {
	    a[index] = hcoeff[0] * b[index] +
	      hcoeff[1] * b[index - 1] +
	      hcoeff[2] * b[index + 1] +
	      hcoeff[3] * b[index - pitchedDimx] +
	      hcoeff[4] * b[index + pitchedDimx] +
	      hcoeff[5] * b[index - stride] +
	      hcoeff[6] * b[index + stride];
	  } else {
	    b[index] = hcoeff[0] * a[index] +
	      hcoeff[1] * a[index - 1] +
	      hcoeff[2] * a[index + 1] +
	      hcoeff[3] * a[index - pitchedDimx] +
	      hcoeff[4] * a[index + pitchedDimx] +
	      hcoeff[5] * a[index - stride] +
	      hcoeff[6] * a[index + stride];
	  }
	}
      }
    }
  }  

  if (t_end%2) {
    for (int i = 2*RADIUS; i < dimz+2*RADIUS; i++) {
      for (int j = 2*RADIUS; j < dimy+2*RADIUS; j++) {
	for (int k = PADDING_SIZE; k < pitchedDimx-PADDING_SIZE; k++) {
	  index = i*stride + j*pitchedDimx + k;
	  a[index] = b[index];
	}
      }
    }    
  } 
  free(b);
}

void printMatrix(float *a, int pitchedDimx, int dimy, int dimz) {

  int index;
  int stride = pitchedDimx * (dimy+2*RADIUS);
  
  for (int i=0; i < dimz+2*RADIUS; i++) {    
    for (int j=0; j < dimy+2*RADIUS; j++) {
      for (int k=0; k < pitchedDimx; k++) {
	index = i*stride + j*pitchedDimx + k;
	printf("%f, ",a[index]);
      }
      printf("\n");
    }
    printf("\n");
  }
}

void printMatrixTemporal(float *a, int pitchedDimx, int dimy, int dimz) {

  int index;
  int stride = pitchedDimx * (dimy+4*RADIUS);
  
  for (int i=0; i < dimz+4*RADIUS; i++) {    
    for (int j=0; j < dimy+4*RADIUS; j++) {
      for (int k=0; k < pitchedDimx; k++) {
	index = i*stride + j*pitchedDimx + k;
	printf("%f, ",a[index]);
      }
      printf("\n");
    }
    printf("\n");
  }
}

bool checkResult(float *a, float *ref, int pitchedDimx, int dimy, int dimz) {

  int index;
  int stride = pitchedDimx * (dimy+2*RADIUS);
  
  for (int i = 0; i < dimz+2*RADIUS; i++) {
    for (int j = 0; j < dimy+2*RADIUS; j++) {
      for (int k = 0; k < pitchedDimx; k++) {
	index = i*stride + j*pitchedDimx + k;
	if (a[index] != ref[index]) {
	  printf("Expected: %f, received: %f at position [z=%d,y=%d,x=%d]\n",ref[index],a[index],i,j,k);
	  return 0;
	}
      }
    }
  }    
  return 1;
}

bool checkResultTemporal(float *a, float *ref, int pitchedDimx, int dimy, int dimz) {

  int index;
  int stride = pitchedDimx * (dimy+4*RADIUS);
  
  for (int i = 0; i < dimz+4*RADIUS; i++) {
    for (int j = 0; j < dimy+4*RADIUS; j++) {
      for (int k = 0; k < pitchedDimx; k++) {
	index = i*stride + j*pitchedDimx + k;
	if (a[index] != ref[index]) {
	  printf("Expected: %f, received: %f at position [z=%d,y=%d,x=%d]\n",ref[index],a[index],i,j,k);
	  return 0;
	}
      }
    }
  }    
  return 1;
}

int main(int argc, char* argv[]) {

  float *h_a, *h_gold_a;
  float *d_a, *d_b;
  float hcoeff[7] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0}; //cc, cw, ce, cn, cs, ct, cb

  cudaEvent_t t0, t1, t2, t3, t4, t5;
  float init, host_comp, host2gpu, gpu2host, gpu_comp, tot;
  int dimx, dimy, dimz, t_end;
  long points, flop;
  float gFlops;
  int opt; // Variable to select the optimization
  char vbs;
  
  if (argc == 7) {
    vbs = 1;
  } else {
    vbs = 0;
    if (argc != 6) {    
      printf("use: <exec> <OPT> <DIMX> <DIMY> <DIMZ> <T_END> <VBS(1)>\n"
	   "Available optimizations (value should be used as the first parameter in the command line):\n"
	   "0 - Base -> no optimization\n"
	   "1 - Sham -> shared memory\n"
	   "2 - ZintReg -> for iteration on Z axis (Paulius)\n"
	   "3 - Zint -> for iteration on Z axis without using registers\n"
	   "4 - ShamZintReg -> shared memory + for iteration on Z axis\n"
	   "5 - ShamZint -> shared memory + for iteration on Z axis without registers\n"
	   "6 - ShamZintTempReg -> shared memory + for iteration on Z axis + temporal blocking\n"
	   "7 - Roc -> use of read only cache (__restrict__ and const modifiers)\n"
	   "8 - ShamRoc -> use of shared memory + read only cache (__restrict__ and const modifiers)\n"
	   "9 - RocZintReg -> for iteration on Z axis + read only cache\n"
	   "10 - RocZint -> for iteration on Z axis without registers + read only cache\n"
	   "11 - ShamRocZintTempReg -> shared memory + read only cache + for iteration on Z axis + temporal blocking\n"
	   );
      exit(-1);
    }
  }
  
  opt = atoi(argv[1]);
  dimx = atoi(argv[2]);
  dimy = atoi(argv[3]);
  dimz = atoi(argv[4]);
  t_end = atoi(argv[5]);
  vbs = atoi(argv[6]);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventCreate(&t2);
  cudaEventCreate(&t3);
  cudaEventCreate(&t4);
  cudaEventCreate(&t5);

  int pitchedDimx = dimx + 2*PADDING_SIZE;
  
  int gold_size;
  
  // If temporal blocking is requested, allocate more device memory
  if ( (opt == 6) || (opt == 11) ) {
    gold_size = pitchedDimx * (dimy+4*RADIUS) * (dimz+4*RADIUS) * sizeof(float);
    // Check if the number of iterations is even
    if ( (t_end%2) != 0) {
      if (vbs == 0) printf("Number of time iterations is odd, adding one iteration!\n");
      t_end++;
    }
  } else {
    gold_size = pitchedDimx * (dimy+2*RADIUS) * (dimz+2*RADIUS) * sizeof(float);
  }
  points = (long)dimx * (long)dimy * (long)dimz * (long)t_end;
  flop = (long)(6 + 7) * points; // 6 adds, 7 multiplies

  cudaEventRecord(t0);

  /* allocate device variables */
  wbCheck(cudaMalloc((void**) &d_a, gold_size));
  wbCheck(cudaMalloc((void**) &d_b, gold_size));

  /* allocate host variables */
  h_a = (float *)malloc(gold_size);
  h_gold_a = (float *)malloc(gold_size);
 
  if ( (opt == 6) || (opt == 11) ) {
    initGoldTemporal(h_a, dimx, dimy, dimz, pitchedDimx);
    initGoldTemporal(h_gold_a, dimx, dimy, dimz, pitchedDimx);
  } else {
    initGold(h_a, dimx, dimy, dimz, pitchedDimx);
    initGold(h_gold_a, dimx, dimy, dimz, pitchedDimx);
  }
  
  cudaEventRecord(t1);

  if (vbs == 0) {
    if ( (opt == 6) || (opt == 11) ) {
      hostStencilTemporal(h_gold_a, t_end, dimx, dimy, dimz, hcoeff, pitchedDimx);
    } else {
      hostStencil(h_gold_a, t_end, dimx, dimy, dimz, hcoeff, pitchedDimx);
    }
  }

#ifdef PRINT_GOLD
  if ( (opt == 6) || (opt == 11) ) {
    printMatrixTemporal(h_gold_a, pitchedDimx, dimy, dimz);    
  } else {  
    printMatrix(h_gold_a, pitchedDimx, dimy, dimz);
  }
#endif

  cudaEventRecord(t2);

  wbCheck(cudaMemcpyToSymbol(coeff, hcoeff, sizeof(hcoeff)));
  wbCheck(cudaMemcpy(d_a, h_a, gold_size, cudaMemcpyHostToDevice)); // Initialize device values
  wbCheck(cudaMemcpy(d_b, d_a, gold_size, cudaMemcpyDeviceToDevice)); // Copy contents from d_a to d_b
 
  cudaEventRecord(t3);

  dim3 dimBlock;
  dim3 dimGrid;

  switch (opt) {
  case 0:
    if (vbs == 0) printf("Optimization level: 0 - Base\n");
    dimBlock.x = BLOCK_DIMX;
    dimBlock.y = BLOCK_DIMY;
    dimBlock.z = BLOCK_DIMZ;
    dimGrid.x = (int)ceil(dimx/BLOCK_DIMX);
    dimGrid.y = (int)ceil(dimy/BLOCK_DIMY);
    dimGrid.z = (int)ceil(dimz/BLOCK_DIMZ);

    for (int i = 0; i < t_end; i++) {
      if (i%2) {
	calcStencilBase <<< dimGrid,dimBlock >>> (d_b, d_a, pitchedDimx, dimy);
      } else {
	calcStencilBase <<< dimGrid,dimBlock >>> (d_a, d_b, pitchedDimx, dimy);
      }
      wbCheck(cudaGetLastError());
    }
    break;

  case 1:
    if (vbs == 0) printf("Optimization level: 1 - Sham\n");
    dimBlock.x = BLOCK_DIMX;
    dimBlock.y = BLOCK_DIMY;
    dimBlock.z = BLOCK_DIMZ;
    dimGrid.x = (int)ceil(dimx/BLOCK_DIMX);
    dimGrid.y = (int)ceil(dimy/BLOCK_DIMY);
    dimGrid.z = (int)ceil(dimz/BLOCK_DIMZ);

    for (int i = 0; i < t_end; i++) {
      if (i%2) {
	calcStencilSham <<< dimGrid,dimBlock >>> (d_b, d_a, pitchedDimx, dimy);
      } else {
	calcStencilSham <<< dimGrid,dimBlock >>> (d_a, d_b, pitchedDimx, dimy);
      }
      wbCheck(cudaGetLastError());
    }
    break;

  case 2:
    if (vbs == 0) printf("Optimization level: 2 - ZintReg\n");
    dimBlock.x = BLOCK_DIMX;
    dimBlock.y = BLOCK_DIMY;
    dimBlock.z = 1;
    dimGrid.x = (int)ceil(dimx/BLOCK_DIMX);
    dimGrid.y = (int)ceil(dimy/BLOCK_DIMY);
    dimGrid.z = 1;

    for (int i = 0; i < t_end; i++) {
      if (i%2) {
	calcStencilZintReg <<< dimGrid,dimBlock >>> (d_b, d_a, pitchedDimx, dimy, dimz);
      } else {
	calcStencilZintReg <<< dimGrid,dimBlock >>> (d_a, d_b, pitchedDimx, dimy, dimz);
      }
      wbCheck(cudaGetLastError());
    }
    break;

  case 3:
    if (vbs == 0) printf("Optimization level: 3 - Zint\n");
    dimBlock.x = BLOCK_DIMX;
    dimBlock.y = BLOCK_DIMY;
    dimBlock.z = 1;
    dimGrid.x = (int)ceil(dimx/BLOCK_DIMX);
    dimGrid.y = (int)ceil(dimy/BLOCK_DIMY);
    dimGrid.z = 1;

    for (int i = 0; i < t_end; i++) {
      if (i%2) {
	calcStencilZint <<< dimGrid,dimBlock >>> (d_b, d_a, pitchedDimx, dimy, dimz);
      } else {
	calcStencilZint <<< dimGrid,dimBlock >>> (d_a, d_b, pitchedDimx, dimy, dimz);
      }
      wbCheck(cudaGetLastError());
    }
    break;

  case 4:
    if (vbs == 0) printf("Optimization level: 4 - ShamZintReg\n");
    dimBlock.x = BLOCK_DIMX;
    dimBlock.y = BLOCK_DIMY;
    dimBlock.z = 1;
    dimGrid.x = (int)ceil(dimx/BLOCK_DIMX);
    dimGrid.y = (int)ceil(dimy/BLOCK_DIMY);
    dimGrid.z = 1;

    for (int i = 0; i < t_end; i++) {
      if (i%2) {
	calcStencilShamZintReg <<< dimGrid,dimBlock >>> (d_b, d_a, pitchedDimx, dimy, dimz);
      } else {
	calcStencilShamZintReg <<< dimGrid,dimBlock >>> (d_a, d_b, pitchedDimx, dimy, dimz);
      }
      wbCheck(cudaGetLastError());
    }
    break;

  case 5:
    if (vbs == 0) printf("Optimization level: 5 - ShamZint\n");
    dimBlock.x = BLOCK_DIMX;
    dimBlock.y = BLOCK_DIMY;
    dimBlock.z = 1;
    dimGrid.x = (int)ceil(dimx/BLOCK_DIMX);
    dimGrid.y = (int)ceil(dimy/BLOCK_DIMY);
    dimGrid.z = 1;

    for (int i = 0; i < t_end; i++) {
      if (i%2) {
	calcStencilShamZint <<< dimGrid,dimBlock >>> (d_b, d_a, pitchedDimx, dimy, dimz);
      } else {
	calcStencilShamZint <<< dimGrid,dimBlock >>> (d_a, d_b, pitchedDimx, dimy, dimz);
      }
      wbCheck(cudaGetLastError());
    }
    break;

  case 6:
    if (vbs == 0) printf("Optimization level: 6 - ShamZintTempReg\n");
    dimBlock.x = BLOCK_DIMX;
    dimBlock.y = BLOCK_DIMY;
    dimBlock.z = 1;
    dimGrid.x = (int)ceil(dimx/(BLOCK_DIMX-2*RADIUS));
    dimGrid.y = (int)ceil(dimy/(BLOCK_DIMY-2*RADIUS));
    dimGrid.z = 1;

    for (int i = 0; i < t_end/2; i++) {
      if (i%2) {
	calcStencilShamZintTempReg <<< dimGrid,dimBlock >>> (d_b, d_a, pitchedDimx, dimy, dimz);
      } else {
	calcStencilShamZintTempReg <<< dimGrid,dimBlock >>> (d_a, d_b, pitchedDimx, dimy, dimz);
      }
      wbCheck(cudaGetLastError());
    }
    break;


  case 7:
    if (vbs == 0) printf("Optimization level: 7 - Roc\n");
    dimBlock.x = BLOCK_DIMX;
    dimBlock.y = BLOCK_DIMY;
    dimBlock.z = BLOCK_DIMZ;
    dimGrid.x = (int)ceil(dimx/BLOCK_DIMX);
    dimGrid.y = (int)ceil(dimy/BLOCK_DIMY);
    dimGrid.z = (int)ceil(dimz/BLOCK_DIMZ);

    for (int i = 0; i < t_end; i++) {
      if (i%2) {
	calcStencilRoc <<< dimGrid,dimBlock >>> (d_b, d_a, pitchedDimx, dimy);
      } else {
	calcStencilRoc <<< dimGrid,dimBlock >>> (d_a, d_b, pitchedDimx, dimy);
      }
      wbCheck(cudaGetLastError());
    }
    break;

  case 8:
    if (vbs == 0) printf("Optimization level: 8 - ShamRoc\n");
    dimBlock.x = BLOCK_DIMX;
    dimBlock.y = BLOCK_DIMY;
    dimBlock.z = BLOCK_DIMZ;
    dimGrid.x = (int)ceil(dimx/BLOCK_DIMX);
    dimGrid.y = (int)ceil(dimy/BLOCK_DIMY);
    dimGrid.z = (int)ceil(dimz/BLOCK_DIMZ);

    for (int i = 0; i < t_end; i++) {
      if (i%2) {
	calcStencilShamRoc <<< dimGrid,dimBlock >>> (d_b, d_a, pitchedDimx, dimy);
      } else {
	calcStencilShamRoc <<< dimGrid,dimBlock >>> (d_a, d_b, pitchedDimx, dimy);
      }
      wbCheck(cudaGetLastError());
    }
    break;

  case 9:
    if (vbs == 0) printf("Optimization level: 9 - RocZintReg\n");
    dimBlock.x = BLOCK_DIMX;
    dimBlock.y = BLOCK_DIMY;
    dimBlock.z = 1;
    dimGrid.x = (int)ceil(dimx/BLOCK_DIMX);
    dimGrid.y = (int)ceil(dimy/BLOCK_DIMY);
    dimGrid.z = 1;

    for (int i = 0; i < t_end; i++) {
      if (i%2) {
	calcStencilRocZintReg <<< dimGrid,dimBlock >>> (d_b, d_a, pitchedDimx, dimy, dimz);
      } else {
	calcStencilRocZintReg <<< dimGrid,dimBlock >>> (d_a, d_b, pitchedDimx, dimy, dimz);
      }
      wbCheck(cudaGetLastError());
    }
    break;

  case 10:
    if (vbs == 0) printf("Optimization level: 10 - RocZint\n");
    dimBlock.x = BLOCK_DIMX;
    dimBlock.y = BLOCK_DIMY;
    dimBlock.z = 1;
    dimGrid.x = (int)ceil(dimx/BLOCK_DIMX);
    dimGrid.y = (int)ceil(dimy/BLOCK_DIMY);
    dimGrid.z = 1;

    for (int i = 0; i < t_end; i++) {
      if (i%2) {
	calcStencilRocZint <<< dimGrid,dimBlock >>> (d_b, d_a, pitchedDimx, dimy, dimz);
      } else {
	calcStencilRocZint <<< dimGrid,dimBlock >>> (d_a, d_b, pitchedDimx, dimy, dimz);
      }
      wbCheck(cudaGetLastError());
    }
    break;    

  case 11:
    if (vbs == 0) printf("Optimization level: 11 - ShamRocZintTempReg\n");
    dimBlock.x = BLOCK_DIMX;
    dimBlock.y = BLOCK_DIMY;
    dimBlock.z = 1;
    dimGrid.x = (int)ceil(dimx/(BLOCK_DIMX-2*RADIUS));
    dimGrid.y = (int)ceil(dimy/(BLOCK_DIMY-2*RADIUS));
    dimGrid.z = 1;

    for (int i = 0; i < t_end/2; i++) {
      if (i%2) {
	calcStencilShamRocZintTempReg <<< dimGrid,dimBlock >>> (d_b, d_a, pitchedDimx, dimy, dimz);
      } else {
	calcStencilShamRocZintTempReg <<< dimGrid,dimBlock >>> (d_a, d_b, pitchedDimx, dimy, dimz);
      }
      wbCheck(cudaGetLastError());
    }
    break;

  default:
    printf("Invalid optimization selected\n");
    break;
  }

  cudaEventRecord(t4);
  cudaDeviceSynchronize();

  if ( (opt == 6) || (opt == 11) ) {
    if ((t_end/2)%2) {
      wbCheck(cudaMemcpy(h_a, d_b, gold_size, cudaMemcpyDeviceToHost));
    } else {
      wbCheck(cudaMemcpy(h_a, d_a, gold_size, cudaMemcpyDeviceToHost));
    }
  } else {
    if (t_end%2) {
      wbCheck(cudaMemcpy(h_a, d_b, gold_size, cudaMemcpyDeviceToHost));
    } else {
      wbCheck(cudaMemcpy(h_a, d_a, gold_size, cudaMemcpyDeviceToHost));
    }
  }
  
  cudaEventRecord(t5);

  cudaFree(d_a);
  cudaFree(d_b);
 
#ifdef PRINT_RESULT
  if ( (opt == 6) || (opt == 11) ) {
    printMatrixTemporal(h_a,pitchedDimx,dimy,dimz);
  } else {
    printMatrix(h_a,pitchedDimx,dimy,dimz);
  }
#endif

  if (vbs == 0) {
    if ( (opt == 6) || (opt == 11) ) {
      if (checkResultTemporal(h_a,h_gold_a,pitchedDimx,dimy,dimz)) {
	printf("Correct results!\n");
      } else {
	printf("Wrong results!!!!!!\n");
      }
    } else {
      if (checkResult(h_a,h_gold_a,pitchedDimx,dimy,dimz)) {
	printf("Correct results!\n");
      } else {
	printf("Wrong results!!!!!!\n");
      }
    }
  }
  cudaEventSynchronize(t5);

  cudaEventElapsedTime(&init, t0, t1);
  cudaEventElapsedTime(&host_comp, t1, t2);
  cudaEventElapsedTime(&host2gpu, t2, t3);
  cudaEventElapsedTime(&gpu_comp, t3, t4);
  cudaEventElapsedTime(&gpu2host, t4, t5);
  cudaEventElapsedTime(&tot, t0, t5);

  gFlops = (1.0e-6)*flop/gpu_comp;
  
  free(h_a);
  free(h_gold_a);

  if (vbs == 0) {
    printf("GPU Clock: %d MHz\n",prop.clockRate/1000);
    printf("DIM = %dx%dx%d; T_END = %d; BLOCK_WIDTH = %dx%dx%d\n", dimx,dimy,dimz,t_end,BLOCK_DIMX,BLOCK_DIMY,BLOCK_DIMZ);
    printf("init=%f, host_comp=%f, host2gpu=%f, gpu_comp=%f, gpu2host=%f, tot=%f \n", 
	      init, host_comp, host2gpu, gpu_comp, gpu2host, tot);
    printf("Stencil Throughput: %f Gpts/s\n", (1.0e-6*points)/gpu_comp); // gpu_comp is measured in ms
    printf("gFlops = %f GFLOPs\n", gFlops);
    printf("\n");
  } else {
    printf("%d,%d,%d,%f,%f\n", dimx,dimy,dimz,gFlops,gpu_comp);
  }
  
  return 0;
}
