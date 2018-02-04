#include <stdio.h>

int main() {

  int c;
  cudaGetDeviceCount(&c);
  printf("Total device %d\n",c);

  int i;
  cudaDeviceProp deviceProp;
  for(i=0; i<c; i++){
    cudaGetDeviceProperties(&deviceProp, i);
    printf("Device %d has compute capability %d.%d.\n",
      i, deviceProp.major, deviceProp.minor);
  }
}
