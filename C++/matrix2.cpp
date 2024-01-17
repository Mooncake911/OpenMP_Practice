#include <omp.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>

using namespace std;
using namespace chrono;

/* Generate triangle-matrix with random values in (a, b) */
vector<vector<double>> generateSquareMatrix(long long n, double a, double b) {
    vector<vector<double>> matrix(n, vector<double>(n, 0.0));

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(a, b);

    for (long long i = 0; i < n; ++i) {
        for (long long j = i; j < n; ++j) {
            matrix[i][j] = dist(gen);
        }
    }

    return matrix;
}


/* Find Max of Min in triangle-matrix */
double find_max_of_min(int num_thr, const vector<vector<double>>& matrix) {
    omp_set_num_threads(num_thr);

    long long numRows = matrix.size();
    long long numCols = matrix[0].size();

    double maxOfMin = numeric_limits<double>::min();

#pragma omp parallel for reduction(max: maxOfMin) //schedule(dynamic) // schedule(guided)
    for (long long i = 0; i < numRows; ++i) {
        double minOfRow = matrix[i][i];

        for (long long  j = i + 1; j < numCols; ++j) {
            if (matrix[i][j] < minOfRow) {
                minOfRow = matrix[i][j];
            }
        }

        if (minOfRow > maxOfMin) {
            maxOfMin = minOfRow;
        }
    }

    return maxOfMin;
}


int main() {
    ofstream csv_file(R"(D:\Projects\PyCharm_Projects\OpenMP\data\matrix2.csv)");
    if (!csv_file.is_open()) {
        cerr << "Open file error!" << endl;
        return 1;
    }
    csv_file << "Num_Threads,Iter,Time\n";

    for (long long j = 10; j < 100000; j *= 10) {
        vector<vector<double>> matrix = generateSquareMatrix(j, 1 , 1000);
        for (int i = 1; i <= 16; ++i) {
            auto start_time = high_resolution_clock::now();
            find_max_of_min(i, matrix);
            auto end_time = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end_time - start_time);
            csv_file << i << "," << j << "," << duration.count() << "\n";
        }
    }

    csv_file.close();
    return 0;
}
