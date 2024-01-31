#include <omp.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <functional>

using namespace std;
using namespace chrono;

void CheckOpenMPSupport() {
    #ifdef _OPENMP
            cout << "OpenMP is supported" << endl;

            #ifdef OMP_NESTED
                    cout << "Nested parallelism is supported" << endl;
            #else
                    cout << "Nested parallelism is NOT supported" << endl;
            #endif

    #else
            cout << "OpenMP is NOT supported" << endl;
    #endif
}


double MaxOfMin(const vector<vector<double>>& matrix);

/* For DEBUG */
void TestResult(double r1, double r2) {
    if (r1 == r2)
        cout << "Results are identical!" << endl;
    else
        cerr << "Results are NOT identical!" << endl;
}

/* Generate matrix with random values in range (a, b) */
vector<vector<double>> generateSquareMatrix(long long n, double a, double b) {
    vector<vector<double>> matrix(n, vector<double>(n, 0.0));

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(a, b);

    for (long long i = 0; i < n; ++i) {
        for (long long j = 0; j < n; ++j) {
            matrix[i][j] = dist(gen);
        }
    }

    return matrix;
}

/* Do experiment and save results in csv file */
void doExperiment(const string& filename, const function<double(int, const vector<vector<double>>&)>& func) {
    ofstream csv_file(filename);
    if (!csv_file.is_open()) {
        cerr << "Open file error!" << endl;
        return;
    }
    csv_file << "Num_Threads,Iter,Time\n";

    for (long long j = 10; j <= 10000; j *= 10) {
        vector<vector<double>> matrix = generateSquareMatrix(j, 1 , 1000);
        for (int i = 1; i <= 16; ++i) {
            auto start_time = high_resolution_clock::now();
            func(i, matrix);
            //TestResult(func(i, matrix), MaxOfMin(matrix));
            auto end_time = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end_time - start_time);
            csv_file << i << "," << j << "," << duration.count() << "\n";
        }
    }

    csv_file.close();
}


// 0.
double MaxOfMin(const vector<vector<double>>& matrix) {
    long long numRows = matrix.size();
    long long numCols = matrix[0].size();

    double maxOfMin = numeric_limits<double>::min();

    for (long long i = 0; i < numRows; ++i) {
        double minOfRow = matrix[i][0];

        for (long long  j = 1; j < numCols; ++j)
            minOfRow = std::min(minOfRow, matrix[i][j]);

        maxOfMin = std::max(maxOfMin, minOfRow);
    }

    return maxOfMin;
}


// 1.
double MaxOfMin1(int num_thr, const vector<vector<double>>& matrix) {
    omp_set_num_threads(num_thr);

    long long numRows = matrix.size();
    long long numCols = matrix[0].size();

    double maxOfMin = numeric_limits<double>::min();

#pragma omp parallel for reduction(max: maxOfMin)
    for (long long i = 0; i < numRows; ++i) {
        double minOfRow = matrix[i][0];

        for (long long  j = 1; j < numCols; ++j)
            minOfRow = std::min(minOfRow, matrix[i][j]);

        maxOfMin = std::max(maxOfMin, minOfRow);
    }

    return maxOfMin;
}


// 2.
double MaxOfMin2(int num_thr, const vector<vector<double>>& matrix) {
    omp_set_num_threads(num_thr);

    long long numRows = matrix.size();
    long long numCols = matrix[0].size();

    double maxOfMin = numeric_limits<double>::min();


    #pragma omp parallel for reduction(max: maxOfMin)
    for (long long i = 0; i < numRows; ++i) {
        double minOfRow = matrix[i][0];

        #pragma omp parallel for reduction(min: minOfRow)
        for (long long  j = 1; j < numCols; ++j)
            minOfRow = std::min(minOfRow, matrix[i][j]);

        maxOfMin = std::max(maxOfMin, minOfRow);
    }

    return maxOfMin;
}


int main() {
    CheckOpenMPSupport();
    doExperiment(R"(D:\Projects\PyCharm_Projects\OpenMP\data\matrix_1.csv)", MaxOfMin1);
    doExperiment(R"(D:\Projects\PyCharm_Projects\OpenMP\data\matrix_2.csv)", MaxOfMin2);
    return 0;
}
