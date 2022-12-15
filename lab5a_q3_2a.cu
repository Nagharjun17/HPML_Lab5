%%cu
#include<iostream>
#include<math.h>
#include<ctime>

__global__ 
void add(int n, float *x, float*y)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    for(int i=index; i<n; i+=stride)
    {
        y[i] = x[i] + y[i];
    }
}

int main(void)
{
    int K = 1<<20;
    
    float *x, *y;
    int count = 1;
    int arr[] = {1,5,2,5,2};
    
    for(int i=0;i<5;i++)
    {
    
    K = K*arr[i];
    count = count*arr[i];
     
    size_t size = K*sizeof(float);
    float *hx = (float*)malloc(size);
    float *hy = (float*)malloc(size);
 
    for(int i=0;i<K;i++)
    {
        hx[i] = 2.0f;
        hy[i] = 3.0f;
    }
 
    cudaMalloc(&x, size);   
    cudaMalloc(&y, size);
 
    clock_t start = clock();

    cudaMemcpy(x, hx, size, cudaMemcpyHostToDevice);
    cudaMemcpy(y, hy, size, cudaMemcpyHostToDevice);
     
    add<<<1,1>>>(K,x,y);

    cudaMemcpy(hy, y, size, cudaMemcpyDeviceToHost);
 
    clock_t end = clock();
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
 
    std::cout<<"Time taken for "<<count<<"M elements is : "<<time_taken<<std::endl;
     
    free(hx);
    free(hy);
    }
    cudaFree(x);
    cudaFree(y);
    
    return 0;

}
