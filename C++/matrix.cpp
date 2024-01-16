#include <omp.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <functional>

using namespace std;


/* Generate matrix with random values in (a, b) */
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

/* Do experiment and save results to csv file */
void doExperiment(const string& filename, const function<long long(int, const vector<vector<double>>&)>& func) {
    ofstream csv_file(filename);
    if (!csv_file.is_open()) {
        cerr << "Open file error!" << endl;
        return;
    }
    csv_file << "Num_Threads,Matrix_len,Time\n";

    long long t;
    for (long long j = 10; j <= 10000; j *= 10) {
        vector<vector<double>> matrix = generateSquareMatrix(j, 1 , 1000);
        for (int i = 1; i <= 16; ++i) {
            t =  func(i, matrix);
            csv_file << i << "," << j << "," << t << "\n";
        }
    }
    csv_file.close();
}



long long MaxOfMin1(int num_thr, const vector<vector<double>>& matrix) {
    omp_set_num_threads(num_thr);

    long long numRows = matrix.size();
    long long numCols = matrix[0].size();

    double maxOfMin = numeric_limits<double>::min();

    auto start_time = chrono::high_resolution_clock::now();

#pragma omp parallel for reduction(max: maxOfMin)
    for (long long i = 0; i < numRows; ++i) {
        double minOfRow = matrix[i][0];

        for (long long  j = 1; j < numCols; ++j)
            minOfRow = std::min(minOfRow, matrix[i][j]);

        maxOfMin = std::max(maxOfMin, minOfRow);
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);

//    cout << "Max of Min in Matrix: " << maxOfMin << endl;
//    cout << "Duration (microseconds): " << duration.count() << endl;

    return duration.count();
}


long long MaxOfMin2(int num_thr, const vector<vector<double>>& matrix) {
    omp_set_num_threads(num_thr);

    long long numRows = matrix.size();
    long long numCols = matrix[0].size();

    double maxOfMin = numeric_limits<double>::min();

    auto start_time = chrono::high_resolution_clock::now();

#pragma omp parallel for reduction(max: maxOfMin)
    for (long long i = 0; i < numRows; ++i) {
        double minOfRow = matrix[i][0];

#pragma omp parallel for reduction(min: minOfRow)
        for (long long  j = 1; j < numCols; ++j)
            minOfRow = std::min(minOfRow, matrix[i][j]);

        maxOfMin = std::max(maxOfMin, minOfRow);
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);

//    cout << "Max of Min in Matrix: " << maxOfMin << endl;
//    cout << "Duration (microseconds): " << duration.count() << endl;

    return duration.count();
}

int main() {
    doExperiment(R"(D:\Projects\PyCharm_Projects\OpenMP\data\matrix_1.csv)", MaxOfMin1);
    doExperiment(R"(D:\Projects\PyCharm_Projects\OpenMP\data\matrix_2.csv)", MaxOfMin2);
    return 0;
}
