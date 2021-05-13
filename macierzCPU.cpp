#include <iostream>
#include <random>
#include <ctime>
#include <chrono>
#include <omp.h>

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
//CPU part
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

//CPU równolegle
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

//Pokazywanie macierzy
void showMatrix(float **macierz, int n){

    for(int i = 0; i<n; ++i){
        for(int j = 0; j<n; ++j){
            std::cout<<macierz[i][j]<<"\t";
        }
        std::cout<<std::endl;
    }
}

int main(){
    //g++ -g -o macierzCPU macierzCPU.cpp -fopenmp  do skompilowania we standardzie openMP
    //srand(time(NULL));
    float **macierzA;
    float **macierzB;
    float **C;
    float **A_T, **B2;
    int n = 1;
    float w =4, u =8;
    float maks = 3.0, mini = -3.0;
    std::cout<<"Podaj rozmiar macierzy kwadratowych A i B: ";
    std::cin>>n;
    // std::cout<<"Podaj maksymalną wartośc elemnetu macierzy A: ";
    // std::cin>>maks;
    // std::cout<<"Podaj minimalną wartośc elementu macierzy B: ";
    // std::cin>>mini;
    generatingMatrixes(macierzA, macierzB, n, maks, mini); //Utworzenie macierzy A i B

    //Kod na CPU
    auto t_start = std::chrono::high_resolution_clock::now();
    C = MulMatrixes(macierzA,macierzB,n); //A*B
    A_T = TransposeMatrix(macierzA, n); //A transponowane
    A_T = MulMatrix_value(A_T, u, n); //A transponowane razy u
    C = AddingMatrixes(C, A_T, n); //Dodanie poprzednich wyników
    C = AddingMatrixes(C, macierzA, n); // Dodanie poprzedniego wyniku do macierzy A
    B2 = MulMatrix_value(macierzB, -w, n); //Pomnożenie macierzy B razy -w
    C = AddingMatrixes(C, B2, n); //Dodanie ostatniej wartości(dzięki -w nie musimy odejmować, wystarczy dodać)
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    std::cout<<"It took: "<< elapsed_time_ms << " ms"<<"\n\n";

    //OpenMP standard
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

    //Dodać możliwość wyświetlania macierzy
    //Sprzątanie
    for(int i=0; i<n; ++i){
        delete [] macierzA[i];
        delete [] macierzB[i];
        delete [] B2[i];
        delete [] C[i];
        delete [] A_T[i];
    }
    delete [] macierzA;
    delete [] macierzB;
    delete [] B2;
    delete [] C;
    delete [] A_T;
}
