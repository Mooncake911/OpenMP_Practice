#include <omp.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>

using namespace std;


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
long long find_max_of_min(int num_thr, const vector<vector<double>>& matrix) {
    omp_set_num_threads(num_thr);

    long long numRows = matrix.size();
    long long numCols = matrix[0].size();

    double maxOfMin = numeric_limits<double>::min();

    auto start_time = chrono::high_resolution_clock::now();

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

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);

//    cout << "Max of Min in Matrix: " << maxOfMin << endl;
//    cout << "Duration (microseconds): " << duration.count() << endl;

    return duration.count();
}


int main() {
    ofstream csv_file(R"(D:\Projects\PyCharm_Projects\OpenMP\data\matrix2.csv)");
    if (!csv_file.is_open()) {
        cerr << "Open file error!" << endl;
        return 1;
    }
    csv_file << "Num_Threads,Matrix_len,Time\n";

    long long t;
    for (long long j = 10; j < 100000; j *= 10) {
        vector<vector<double>> matrix = generateSquareMatrix(j, 1 , 1000);
        cout << endl;
        for (int i = 1; i <= 16; ++i) {
            t =  find_max_of_min(i, matrix);
            cout << i << "," << j << "," << t << endl;
            csv_file << i << "," << j << "," << t << "\n";
        }
    }

    csv_file.close();
    return 0;
}
