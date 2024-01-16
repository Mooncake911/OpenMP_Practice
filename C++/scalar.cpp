#include <omp.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <functional>

using namespace std;


/* Generate vector with random values in (a, b) */
vector<double> rand_vector(long long n, double a, double b) {
    vector<double> data;

    random_device rd;   // non-deterministic generator
    mt19937 gen(rd());  // to seed mersenne twister.
    uniform_real_distribution<double> dist(a,b); // distribute results between a and b inclusive.

    for (long long i = 0; i < n; ++i) {
        data.push_back(dist(gen));
    }

    return data;
}

/* Do experiment and save results to csv file */
void doExperiment(const string& filename, const function<long long(int, const vector<double>&, const vector<double>&)>& func) {
    ofstream csv_file(filename);
    if (!csv_file.is_open()) {
        cerr << "Open file error!" << endl;
        return;
    }
    csv_file << "Num_Threads,Iter,Time\n";

    long long t;
    for (long long j = 90; j <= 9000000; j *= 10) {
        vector<double> vector1 = rand_vector(j, 1, 1000);
        vector<double> vector2 = rand_vector(j, 1, 1000);
        for (int i = 1; i <= 16; ++i) {
            t =  func(i, vector1, vector2);
            csv_file << i << "," << j << "," << t << "\n";
        }
    }
    csv_file.close();
}



// 1.
long long scalarAtomic(int num_thr, const vector<double>& v1, const vector<double>& v2) {
    omp_set_num_threads(num_thr);

    // Length check
    if (v1.size() != v2.size()) {
        cerr << "Error: Vector sizes do not match." << endl;
        return 0.0;
    }

    auto start_time = chrono::high_resolution_clock::now();

    double result = 0.0;
#pragma omp parallel for
    for (long long i = 0; i < v1.size(); ++i) {
#pragma omp atomic
        result += v1[i] * v2[i];
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);

//    cout << "Scalar Product: " << result << endl;
//    cout << "Duration (microseconds): " << duration.count() << endl;

    return duration.count();
}


// 2.
long long scalarCritical(int num_thr, const vector<double>& v1, const vector<double>& v2) {
    omp_set_num_threads(num_thr);

    // Length check
    if (v1.size() != v2.size()) {
        cerr << "Error: Vector sizes do not match." << endl;
        return 0.0;
    }

    auto start_time = chrono::high_resolution_clock::now();

    double result = 0.0;
#pragma omp parallel for
    for (long long i = 0; i < v1.size(); ++i) {
#pragma omp critical
        result += v1[i] * v2[i];
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);

//    cout << "Scalar Product: " << result << endl;
//    cout << "Duration (microseconds): " << duration.count() << endl;

    return duration.count();
}


// 3.
long long scalarLock(int num_thr, const vector<double>& v1, const vector<double>& v2) {
    omp_set_num_threads(num_thr);

    // Length check
    if (v1.size() != v2.size()) {
        cerr << "Error: Vector sizes do not match." << endl;
        return 0.0;
    }

    auto start_time = chrono::high_resolution_clock::now();

    double result = 0.0;
    omp_lock_t lock;
    omp_init_lock(&lock);

#pragma omp parallel for
    for (long long i = 0; i < v1.size(); ++i) {
        omp_set_lock(&lock);
        result += v1[i] * v2[i];
        omp_unset_lock(&lock);
    }

    omp_destroy_lock(&lock);

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);

//    cout << "Scalar Product: " << result << endl;
//    cout << "Duration (microseconds): " << duration.count() << endl;

    return duration.count();
}


// 4.
long long scalarReduction(int num_thr, const vector<double>& v1, const vector<double>& v2) {
    omp_set_num_threads(num_thr);

    // Length check
    if (v1.size() != v2.size()) {
        cerr << "Error: Vector sizes do not match." << endl;
        return 0.0;
    }

    auto start_time = chrono::high_resolution_clock::now();

    double result = 0.0;
#pragma omp parallel for reduction(+:result)
    for (long long i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);

//    cout << "Scalar Product: " << result << endl;
//    cout << "Duration (microseconds): " << duration.count() << endl;

    return duration.count();
}


int main() {
    doExperiment(R"(D:\Projects\PyCharm_Projects\OpenMP\data\scalar_atomic.csv)", scalarAtomic);
    doExperiment(R"(D:\Projects\PyCharm_Projects\OpenMP\data\scalar_critical.csv)", scalarCritical);
    doExperiment(R"(D:\Projects\PyCharm_Projects\OpenMP\data\scalar_lock.csv)", scalarLock);
    doExperiment(R"(D:\Projects\PyCharm_Projects\OpenMP\data\scalar_reduction.csv)", scalarReduction);
    return 0;
}
