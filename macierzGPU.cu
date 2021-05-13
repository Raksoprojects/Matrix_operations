#include <iostream>
#include <random>
#include <ctime>
#include <chrono>
#include <omp.h>

#define THREADS 32

//========================================Kernele============================================
__global__ void AddKernel(const float *A, const float *B,float *C, const int n){
     
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    __syncthreads();
    
    if (row < n && col < n) C[(row*n) + col] = A[(row*n) + col] + B[(row*n) + col];

}

__global__ void MulKernel(const float *A, const float *B,float *C, const int n){
    
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    __syncthreads();

    float suma = 0.0f; 
    if (row < n && col < n){
         for(int k=0;k<n;++k){
            suma += A[(row*n) + k] * B[(k*n) + col];
        }
        C[row*n + col] = suma;
    }
}

__global__ void TransposeKernel(const float *A, float* A_T, const int n){
    
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    __syncthreads();
    
    if (row < n && col < n){
        A_T[col*n + row] = A[(row*n) + col];
    }
    
}

__global__ void MulValKernel(const float* A, const float* val, float *C, const int n){
    
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    __syncthreads();

    if (row < n && col < n){
       C[(row*n) + col] = A[(row*n) + col] * (*val);
    }
   
}

//====================================================Kod GPU=================================================
float* AddingMatrixes_GPU(const float *macierzA, const float *macierzB,const int n){

    float *macierzC;
    macierzC = new float[n*n];
    float *d_A = new float[n*n];
    float *d_B = new float[n*n];
    float *d_C = new float[n*n];
    size_t size = n*n*sizeof(float);

    cudaMalloc(&d_A,size);
    cudaMalloc(&d_B,size);
    cudaMalloc(&d_C,size);

    cudaMemcpy(d_A, macierzA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, macierzB, size, cudaMemcpyHostToDevice);
    
    int blocks = ceil(n/float(THREADS));
    dim3 threadsPerBlock(THREADS, THREADS);
    dim3 numBlocks(blocks, blocks);
    AddKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    cudaMemcpy(macierzC, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return macierzC;
}

float* MulMatrixes_GPU(const float *macierzA, const float *macierzB, const int n){

    float *macierzC;
    macierzC = new float[n*n];
    float *d_A = new float[n*n];
    float *d_B = new float[n*n];
    float *d_C = new float[n*n];
    size_t size = n*n*sizeof(float);

    cudaMalloc(&d_A,size);
    cudaMalloc(&d_B,size);
    cudaMalloc(&d_C,size);

    cudaMemcpy(d_A, macierzA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, macierzB, size, cudaMemcpyHostToDevice);
    
    int blocks = ceil(n/float(THREADS));
    dim3 threadsPerBlock(THREADS, THREADS);
    dim3 numBlocks(blocks, blocks);
    MulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    cudaMemcpy(macierzC, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C);

    return macierzC;    
}

float* TransposeMatrix_GPU(const float *macierzA, const int n){

    float *macierzC;
    macierzC = new float[n*n];
    float *d_A = new float[n*n];
    float *d_C = new float[n*n];
    size_t size = n*n*sizeof(float);

    cudaMalloc(&d_A,size);
    cudaMalloc(&d_C,size);

    cudaMemcpy(d_A, macierzA, size, cudaMemcpyHostToDevice);
    
    int blocks = ceil(n/float(THREADS));
    dim3 threadsPerBlock(THREADS, THREADS);
    dim3 numBlocks(blocks, blocks);
    TransposeKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_C, n);
    cudaMemcpy(macierzC, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A); 
    cudaFree(d_C);

    return macierzC;
}

float* MulMatrix_value_GPU(const float *macierzA, const float val, const int n){

    float *macierzC;
    macierzC = new float[n*n];
    float *d_A;
    float *d_val;
    float *d_C;
    size_t size = n*n*sizeof(float);

    cudaMalloc((void **)&d_A,size);
    cudaMalloc((void **)&d_val,sizeof(float));
    cudaMalloc((void **)&d_C,size);
    

    cudaMemcpy(d_A, macierzA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, &val, sizeof(float), cudaMemcpyHostToDevice);
    
    int blocks = ceil(n/float(THREADS));
    dim3 threadsPerBlock(THREADS, THREADS);
    dim3 numBlocks(blocks, blocks);
    MulValKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_val, d_C, n);
    cudaMemcpy(macierzC, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C); 
    cudaFree(d_val);

    return macierzC;
}

float **computationsGPU(float **macierzA, float** macierzB, const float u, const float w, const int n){

    //Wykonanie obliczeń na GPU, wywołanie odpowiednich funkcji i przygotowanie tablic
    float* macierzC = new float[n*n];
    float **C;
    C = new float*[n];
    for(int i = 0; i < n; ++i) C[i] = new float[n];
    float* A = new float[n*n];
    float* B = new float[n*n];
    float* A_T = new float[n*n];
    float* B2 = new float[n*n];
    
    for(int i =0; i < n; ++i){
        for(int j=0; j < n; ++j){
            A[(i*n)+j] = macierzA[i][j]; //"spłaszczanie" tablic
            B[(i*n)+j] = macierzB[i][j];
       }    
     }
     auto t_start = std::chrono::high_resolution_clock::now();
     macierzC = MulMatrixes_GPU(A,B,n); //A*B
     A_T = TransposeMatrix_GPU(A, n); //A transponowane
     A_T = MulMatrix_value_GPU(A_T, u, n); //A transponowane razy u
     macierzC = AddingMatrixes_GPU(macierzC, A_T, n); //Dodanie poprzednich wyników
     macierzC = AddingMatrixes_GPU(macierzC, A, n); // Dodanie poprzedniego wyniku do macierzy A
     B2 = MulMatrix_value_GPU(B, -w, n); //Pomnożenie macierzy B razy -w
     macierzC = AddingMatrixes_GPU(macierzC, B2, n); //Dodanie ostatniej wartości(dzięki -w nie musimy odejmować, wystarczy dodać)
     auto t_end = std::chrono::high_resolution_clock::now();
     double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
     std::cout<<"It took: "<< elapsed_time_ms << " ms"<<"\n\n";
    
    for(int i =0; i < n; ++i){
        for(int j=0; j < n; ++j){
            C[i][j] = macierzC[(i*n) + j]; //przepisywanie z spłaszczonej tablicy do tablicy 2D
       } 
    }
    //sprzątanie
     delete [] A;
     delete [] B;
     delete [] A_T;
     delete [] B2;
     delete [] macierzC;
    
    cudaDeviceReset();
    return C;
}

void generatingMatrixes(float **&macierzA, float **&macierzB, int n, float maks, float mini){
    
    macierzA = new float*[n];
    macierzB = new float*[n];
    for (int i = 0;i<n;i++) {macierzA[i] = new float[n]; macierzB[i] = new float[n];} 
    //Generating matrixes A and B using pseudorandom values
    for(int i = 0; i<n; ++i){
        for(int j = 0; j<n; ++j){
            macierzA[i][j] = ((float)rand() / RAND_MAX) * (maks - mini) + mini;
            macierzB[i][j] = ((float)rand() / RAND_MAX) * (maks - mini) + mini;
        }
    }
}

//Pokazywanie macierzy
void showMatrix(float **macierz, int n){

    for(int i = 0; i<n; ++i){
        for(int j = 0; j<n; ++j){
            std::cout<<macierz[i][j]<<"\t";
        }
        std::cout<<std::endl;
    }
}

//=============================CPU part====================================================================

float** AddingMatrixes(float **macierzA, float **macierzB, int n){

    float **macierzC;
    macierzC = new float*[n];
    for (int i = 0;i<n;i++) macierzC[i] = new float[n];
    
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
            macierzC[i][j] = macierzA[i][j] + macierzB[i][j];
        }
    }
    return macierzC;
}

float** MulMatrixes(float **macierzA, float **macierzB, int n){

    float **macierzC;
    macierzC = new float*[n];
    for (int i = 0;i<n;i++) macierzC[i] = new float[n];
    
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
            for(int k=0;k<n;++k){
            macierzC[i][j] += macierzA[i][k] * macierzB[k][j];
            }       
        }
    }
    return macierzC;
}

float** TransposeMatrix(float **macierzA, int n){

    float **macierzC;
    macierzC = new float*[n];
    for (int i = 0;i<n;i++) macierzC[i] = new float[n];
    
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
            macierzC[i][j] = macierzA[j][i];                 
        }
    }
    return macierzC;
}

float** MulMatrix_value(float **macierz, float value, int n){

    float **macierzC;
    macierzC = new float*[n];
    for (int i = 0;i<n;i++) macierzC[i] = new float[n];
    
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
            macierzC[i][j] = macierz[i][j] *value;
        }
    }
    return macierzC;
}
/*
//=================================CPU równolegle=============================================
//Niestety nie wiem jak skompilować to na cudzie, a raczej na ts-tigerze, natomiast u mnie na komputerze działa
//przy pomocy komendy: g++ -g -o macierzCPU macierzCPU.cpp -fopenmp
float** MulMatrixes_MP(float **macierzA, float **macierzB, int n){
    
    float **macierzC;
    macierzC = new float*[n];
    int i,j,k = 0;
    #pragma omp parallel for private(i) shared(macierzC)
    for (int i = 0;i<n;i++) macierzC[i] = new float[n];
    #pragma omp parallel for private(i, j, k) shared(macierzA, macierzB, macierzC)
    for(i = 0; i < n; i++ ) {
        //std::cout<<"There are "<<omp_get_num_threads()<<" threads"<<std::endl;
        for(j = 0; j < n; j++) {
            for(k = 0; k < n; k++){
                macierzC[i][j] += macierzA[i][k] * macierzB[k][j]; 
            }   
        }  
    }
    return macierzC; 
}
float** MulMatrix_value_MP(float **macierz, float value, int n){

    float **macierzC;
    macierzC = new float*[n];
    int i,j = 0;
    #pragma omp parallel for private(i) shared(macierzC)
    for (int i = 0;i<n;i++) macierzC[i] = new float[n];
    #pragma omp parallel for private(i, j) shared(macierz, macierzC)
    for(i=0;i<n;++i){
        for(j=0;j<n;++j){
            macierzC[i][j] = macierz[i][j] *value;
        }
    }
    return macierzC;
}
float** TransposeMatrix_MP(float **macierz, int n){

    float **macierzC;
    macierzC = new float*[n];
    int i,j = 0;
    #pragma omp parallel for private(i) shared(macierzC)
    for (int i = 0;i<n;i++) macierzC[i] = new float[n];
    #pragma omp parallel for private(i, j) shared(macierz, macierzC)
    for(i=0;i<n;++i){
        for(j=0;j<n;++j){
            macierzC[i][j] = macierz[j][i];                 
        }
    }
    return macierzC;
}
float** AddingMatrixes_MP(float **macierzA, float **macierzB, int n){

    float **macierzC;
    macierzC = new float*[n];
    int i,j = 0;
    #pragma omp parallel for private(i) shared(macierzC)
    for (int i = 0;i<n;i++) macierzC[i] = new float[n];
    #pragma omp parallel for private(i, j) shared(macierzA, macierzB, macierzC)
    for(i=0;i<n;++i){
        for(j=0;j<n;++j){
            macierzC[i][j] = macierzA[i][j] + macierzB[i][j];
        }
    }
    return macierzC;
}
*/

int main(){

    //X = AB + uAT + A − wB
    srand(time(NULL));
    float **macierzA;
    float **macierzB;
    float **A_T, **B2;
    float maks = 3.0, mini = -3.0;
    int n = 2;
    std::cout<<"Podaj rozmiar macierzy kwadratowych A i B: ";
    std::cin>>n;
    std::cout<<"Podaj maksymalną wartośc elementów macierzy A i B: ";
    std::cin>>maks;
    std::cout<<"Podaj minimalną wartośc elementów macierzy A i B: ";
    std::cin>>mini;
    float **C_gpu;
    float **C_cpu;
    float w = 4.0;
    float u = 8.0;
    generatingMatrixes(macierzA, macierzB, n, maks, mini);
    //=================================Kod na CPU=================================
    auto t_start = std::chrono::high_resolution_clock::now();
    C_cpu = MulMatrixes(macierzA,macierzB,n); //A*B
    A_T = TransposeMatrix(macierzA, n); //A transponowane
    A_T = MulMatrix_value(A_T, u, n); //A transponowane razy u
    C_cpu = AddingMatrixes(C_cpu, A_T, n); //Dodanie poprzednich wyników
    C_cpu = AddingMatrixes(C_cpu, macierzA, n); // Dodanie poprzedniego wyniku do macierzy A
    B2 = MulMatrix_value(macierzB, -w, n); //Pomnożenie macierzy B razy -w
    C_cpu = AddingMatrixes(C_cpu, B2, n); //Dodanie ostatniej wartości(dzięki -w nie musimy odejmować, wystarczy dodać)
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    std::cout<<"It took: "<< elapsed_time_ms << " ms"<<"\n\n";
    
    //============================================Kod na GPU ================================
    C_gpu = computationsGPU(macierzA,macierzB,u,w,n);

    char ans= 'n';
    std::cout<<"Wyświetlić macierz końcową(y/n)?: ";
    std::cin>>ans;

    if(ans == 'y'){
        std::cout<<"\n"<<"Macierz:\n";
        showMatrix(C_cpu,n);
    }
    std::cout<<"Wyświetlić macierze składowe A i B(y/n)?: ";
    std::cin>>ans;

    if(ans == 'y'){
        std::cout<<"\n"<<"MacierzA:\n";
        showMatrix(macierzA,n);
        std::cout<<"\n"<<"MacierzB:\n";
        showMatrix(macierzB,n);
    }

    /*
    //===================================================OpenMP standard==============================
    //nie znalazłem niestety informacji jak zkompilować plik w standardzie OpenMP na komputerze ts-tiger, 
    //natomiast na moim komputerze kod działa poprawnie.
    t_start = std::chrono::high_resolution_clock::now();
    omp_set_num_threads(omp_get_num_procs()); //używanie maksymalnej liczby wątków
    C = MulMatrixes_MP(macierzA,macierzB,n); //A*B
    A_T = TransposeMatrix_MP(macierzA, n); //A transponowane
    A_T = MulMatrix_value_MP(A_T, u, n); //A transponowane razy u
    C = AddingMatrixes_MP(C, A_T, n); //Dodanie poprzednich wyników
    C = AddingMatrixes_MP(C, macierzA, n); // Dodanie poprzedniego wyniku do macierzy A
    B2 = MulMatrix_value_MP(macierzB, -w, n); //Pomnożenie macierzy B razy -w
    C = AddingMatrixes_MP(C, B2, n); //Dodanie ostatniej wartości(dzięki -w nie musimy odejmować, wystarczy dodać)
    t_end = std::chrono::high_resolution_clock::now();
    //showMatrix(C,n);
    elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    std::cout<<"It took: "<< elapsed_time_ms << " ms\n";
    */

    //Sprzątanie
    for(int i=0; i<n; ++i){
        delete [] macierzA[i];
        delete [] macierzB[i];
        delete [] C_cpu[i];
        delete [] C_gpu[i];
    }
    delete [] macierzA;
    delete [] macierzB;
    delete [] C_cpu;
    delete [] C_gpu;

    return 0;
}