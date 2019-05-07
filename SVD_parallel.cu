#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "fileRead.h"
#include<iostream>
#include<math.h>
#include<float.h>
#include <ctime>
#include <ratio>
#include <chrono>

#define BLOCK_SIZE 1

using namespace std;
	using namespace std::chrono;
	int total_time =0;
void debug(int n){
  //  cout<<"asd "<<n<<endl;
}

__global__ void VectorSub(double *a, double *b, double *c, int n)
{
    int i = threadIdx.x;

    if(i < n)
        c[i] = a[i] + b[i];
}

void printMatrix(int m, int n, double*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}

// Matrix Matrix Multiplication
__global__ void gpu_matrix_matrix(double *a,double *b, double *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[i * m + row] * b[n * col + i];
        }
        c[col * k + row] = sum;
    }
} 

// Matrix transpose
__global__ void gpu_matrix_transpose(double* mat_in, double* mat_out, unsigned int rows, unsigned int cols) 
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < cols && row < rows) 
    {
        unsigned int pos = col * rows + row;
        unsigned int trans_pos = row * cols + col;
        mat_out[trans_pos] = mat_in[pos];
    }
}

int main(int argc, char*argv[])
{
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

    testAndTrainData tt = calcData();
    vector<vector<double >> svd(10);
    
    cout<<"Training Size of each Set"<<endl;
   for(int i=0;i<10;i++){
        cout<<tt.train[i].size()<<"  "<<tt.train[i][0].size()<<endl;

    }

    for(int set=0;set<10;set++){
        //------------------------
        int m = tt.train[set][0].size();
        int n = tt.train[set].size();
         int lda = m;
         double A[lda*n];
       for(int j=0;j<m;j++){
            for(int k=0;k<n;k++){
                A[k*m + j] = tt.train[set][k][j];
            }
        }
        
        double U[lda*m]; // m-by-m unitary matrix 
        double VT[lda*n];  // n-by-n unitary matrix
        double S[n]; // singular value
        
        double *d_A = NULL;
        double *d_S = NULL;
        double *d_U = NULL;
        double *d_VT = NULL;
        int *devInfo = NULL;
        double *d_work = NULL;
        double *d_rwork = NULL;
        double *d_W = NULL;  // W = S*VT

        int lwork = 0;
        int info_gpu = 0;
    //    printMatrix(m, n, A, lda, "A");
    //    printf("=====\n");

        cusolver_status = cusolverDnCreate(&cusolverH);
        cublas_status = cublasCreate(&cublasH);
        
        cudaMalloc ((void**)&d_A  , sizeof(double)*lda*n);
        cudaMalloc ((void**)&d_S  , sizeof(double)*n);
        cudaMalloc ((void**)&d_U  , sizeof(double)*lda*m);
        cudaMalloc ((void**)&d_VT , sizeof(double)*lda*n);
        cudaMalloc ((void**)&devInfo, sizeof(int));
        cudaMalloc ((void**)&d_W  , sizeof(double)*lda*n);
        cudaMemcpy(d_A, A, sizeof(double)*lda*n, cudaMemcpyHostToDevice);
        
        cusolver_status = cusolverDnDgesvd_bufferSize(cusolverH, m, n, &lwork );
        cudaMalloc((void**)&d_work , sizeof(double)*lwork);
        
        signed char jobu = 'A'; // all m columns of U
        signed char jobvt = 'A'; // all n columns of VT
        cusolver_status = cusolverDnDgesvd (
            cusolverH,
            jobu,
            jobvt,
            m,
            n,
            d_A,
            lda,
            d_S,
            d_U,
            lda,  // ldu
            d_VT,
            lda, // ldvt,
            d_work,
            lwork,
            d_rwork,
            devInfo);
        cudaDeviceSynchronize();
        cudaMemcpy(U , d_U , sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
        cudaMemcpy(VT, d_VT, sizeof(double)*lda*n, cudaMemcpyDeviceToHost);
        cudaMemcpy(S , d_S , sizeof(double)*n    , cudaMemcpyDeviceToHost);
        cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
        
      //  printf("after gesvd: info_gpu = %d\n", info_gpu);
      //  printf("=====\n");

    /*
        printf("S = (matlab base-1)\n");
        printMatrix(n, 1, S, lda, "S");
        printf("=====\n");

        printf("U = (matlab base-1)\n");
        printMatrix(m, m, U, lda, "U");
        printf("=====\n");

        printf("VT = (matlab base-1)\n");
        printMatrix(n, n, VT, lda, "VT");
        printf("=====\n");
    */
        double Ui[20*lda];

        for(int i=0;i<20*lda;i++){
                Ui[i] = U[i];
        } 

        /* printf("U = (matlab base-1)\n");
        printMatrix(20, m, Ui, lda, "Ui");
        printf("=====\n");
        */
        
        int d1 = 20;
        int d2 = lda;

        double *tUi;
        cudaMallocHost((void **) &tUi, sizeof(double)*d2*d1);

        double *d_tUi,*d_Ui;
        cudaMalloc((void **) &d_tUi, sizeof(double)*d2*d1);
        cudaMalloc((void **) &d_Ui, sizeof(double)*d1*d2);
        cudaMemcpy(d_Ui, Ui, sizeof(double)*d1*d2, cudaMemcpyHostToDevice);

        dim3 dimGrid((d2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (d1 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	
auto start = high_resolution_clock::now();
        gpu_matrix_transpose<<<dimGrid,dimBlock>>>(d_Ui,d_tUi,d1,d2);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop-start);
    //cout<<duration.count()<<endl;
    total_time += duration.count();

        
        cudaMemcpy(tUi, d_tUi, sizeof(double)*d2*d1, cudaMemcpyDeviceToHost);
         /*
        printf("U = (matlab base-1)\n");
        printMatrix(m, 20, tUi, lda, "Ut");
        printf("=====\n");
        */
        
        double *utu;
        cudaMallocHost((void **) &utu, sizeof(double)*d2*d2);
        
        double  *d_utu;
        cudaMalloc((void **) &d_tUi, sizeof(double)*d2*d1);
        cudaMalloc((void **) &d_utu, sizeof(double)*d2*d2);

        // copy matrix A and B from host to device memory
        cudaMemcpy(d_Ui, Ui, sizeof(double)*d1*d2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_tUi, tUi, sizeof(double)*d2*d1, cudaMemcpyHostToDevice);

        dim3 dimGrid1((d2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (d2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 dimBlock1(BLOCK_SIZE, BLOCK_SIZE);

 start = high_resolution_clock::now();
        gpu_matrix_matrix<<<dimGrid1, dimBlock1>>>(d_tUi, d_Ui, d_utu, d2, d1, d2);
    stop = high_resolution_clock::now();
     duration = duration_cast<microseconds>(stop-start);
    //cout<<duration.count()<<endl;
    total_time += duration.count();


        cudaMemcpy(utu, d_utu, sizeof(double)*d2*d2, cudaMemcpyDeviceToHost);

        //printf("U = (matlab base-1)\n");
        //printMatrix(m, m, utu, lda, "U");
        //printf("=====\n");
        //------------------------

        //adding utu to svd[set]
       // cout<<"SVD"<<endl;
        for(int i=0;i<d2*d2;i++){
            svd[set].push_back(utu[i]);
         //   cout<<set<<"----"<<utu[i]<<endl;
        }
        cudaDeviceReset();
    }
    
cout<<"Training Completed"<<endl;
    //prediction

    vector<double> accuracy(10);
    int total_img=0, total_acc=0;
    
    //host;
    double *result , *svdU;
    cudaMallocHost((void **) &result , 784 * sizeof(double));
    cudaMallocHost((void **) &svdU , 784*784 * sizeof(double));
    double *img;
    cudaMallocHost((void **) &img , 784 * sizeof(double));
    double *sub;
    cudaMallocHost((void **) &sub , 784 * sizeof(double));
    
    //device;
    double *d_result , *d_svdU;
    cudaMalloc((void **) &d_result, sizeof(double)* 784);
    cudaMalloc((void **) &d_svdU , 784*784 * sizeof(double));
    double *d_img;
    cudaMalloc((void **) &d_img , 784 * sizeof(double));
    double *d_sub;
    cudaMalloc((void **) &d_sub , 784 * sizeof(double));

    for(int set=0;set<10;set++){
        int acc = 0;
        dim3 dimGrid2((784 + BLOCK_SIZE - 1) / BLOCK_SIZE, (784 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE);

        for(int row=0;row<tt.test[set].size();row++){
            for(int i=0;i<784;i++){
                img[i] = tt.test[set][row][i];
            }
            cudaMemcpy(d_img, img, 784 * sizeof(double), cudaMemcpyDeviceToHost);

            double min = DBL_MAX;
	    //long double min = 999999999999999999999;
            int index = 0;

            //check for every svd;
            for(int k=0;k<10;k++){
                for(int i=0;i<784*784;i++){
                    svdU[i] = svd[k][i];
                }
                cudaMemcpy(d_svdU, svdU , 784 * 784 * sizeof(double) , cudaMemcpyDeviceToHost);                

                //UUT * img

auto start = high_resolution_clock::now();
                gpu_matrix_matrix<<<dimGrid2, dimBlock2>>>(d_svdU, d_img, d_result, 784, 784, 1);
    auto stop = high_resolution_clock::now();
     auto duration = duration_cast<microseconds>(stop-start);
    //cout<<duration.count()<<endl;
    total_time += duration.count();





                // result -img
start = high_resolution_clock::now();
                VectorSub<<<1,784>>>(d_result,d_img,d_sub,784);
     stop = high_resolution_clock::now();
     duration = duration_cast<microseconds>(stop-start);
    //cout<<duration.count()<<endl;
    total_time += duration.count();



                cudaMemcpy(sub, d_sub, 784 * sizeof(double), cudaMemcpyHostToDevice);
                //sub * sub 
                double inner_product = 0.0;
                for(int i=0;i<784;i++){
                    inner_product += sub[i]*sub[i];
                }

               /* printf("Result = (matlab base-1)\n");
                printMatrix(784, 1, result, m, "Result");
                printf("=====\n"); 
                cout<<"InnerProduct SQRT - "<<sqrt(inner_product); */

                if(min > sqrt(inner_product)){
                    min = sqrt(inner_product);
                    index = k;
                }

            }
            if(index == set){
                total_acc++;
                acc++;
                cout<<set<<"----"<<index<<endl;
            }

        }
        accuracy[set] = (double)acc/(double)tt.test[set].size();
        total_img += tt.test[set].size();
    }

    cudaDeviceReset();

    for(int set=0;set<10;set++){
        cout<<"Accuracy of Set "<<set<<" is "<<accuracy[set]<<endl;
    }
    cout<<total_img<<"---"<<total_acc<<endl;
    cout<<"Total Accurayc is "<<(double)total_acc/(double)total_img<<endl; 
	cout<<"Total Time - "<<total_time<<"---- No of threads - "<<BLOCK_SIZE<<endl;
    return 0;
}
