#include <omp.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <functional>

using namespace std;
using namespace chrono;

struct VectorPair {
    vector<double> v1;
    vector<double> v2;
};

double scalar(const VectorPair& p);

/* For DEBUG */
void TestResult(double r1, double r2) {
    if (r1 == r2)
        cout << "Results are identical!" << endl;
    else
        cerr << "Results are NOT identical!" << endl;
}

/* Generate vector with random values in range (a, b) */
vector<double> rand_vector(long long n, double a, double b) {
    vector<double> data(n);

    random_device rd;   // non-deterministic generator
    mt19937 gen(rd());  // to seed mersenne twister.
    uniform_real_distribution<double> dist(a,b); // distribute results between a and b inclusive.

    for (double& i : data)
        i = dist(gen);

    return data;
}

/* Do experiment and save results in csv file */
void doExperiment(const string& filename, const function<double(int, const VectorPair&)>& func) {
    ofstream csv_file(filename);
    if (!csv_file.is_open()) {
        cerr << "Open file error!" << endl;
        return;
    }
    csv_file << "Num_Threads,Iter,Time\n";

    VectorPair pair;
    for (long long j = 10; j <= 1000000; j *= 10) {
        pair.v1 = rand_vector(j, 1, 1000);
        pair.v2 = rand_vector(j, 1, 1000);
        for (int i = 1; i <= 16; ++i) {
            auto start_time = high_resolution_clock::now();
            func(i, pair);
            //TestResult(func(i, pair), scalar(pair));
            auto end_time = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end_time - start_time);
            csv_file << i << "," << j << "," << duration.count() << "\n";
        }
    }
    csv_file.close();
}


// 0.
double scalar(const VectorPair& p) {
    double result = 0.0;
    for (long long i = 0; i < p.v1.size(); ++i) {
        result += p.v1[i] * p.v2[i];
    }
    return result;
}


// 1.
double scalarAtomic(int num_thr, const VectorPair& p) {
    omp_set_num_threads(num_thr);

    double result = 0.0;
#pragma omp parallel for
    for (long long i = 0; i < p.v1.size(); ++i) {
#pragma omp atomic
        result += p.v1[i] * p.v2[i];
    }

    return result;
}


// 2.
double scalarCritical(int num_thr, const VectorPair& p) {
    omp_set_num_threads(num_thr);

    double result = 0.0;
#pragma omp parallel
    {

        double local_result = 0.0;
#pragma omp for
        for (long long i = 0; i < p.v1.size(); ++i)
            local_result += p.v1[i] * p.v2[i];

#pragma omp critical
        result += local_result;
    }

    return result;
}


// 3.
double scalarLock(int num_thr, const VectorPair& p) {
    omp_set_num_threads(num_thr);
    omp_lock_t lock;
    omp_init_lock(&lock);

    double result = 0.0;
#pragma omp parallel
    {

        double local_result = 0.0;
#pragma omp for
        for (long long i = 0; i < p.v1.size(); ++i)
            local_result += p.v1[i] * p.v2[i];

        omp_set_lock(&lock);
        result += local_result;
        omp_unset_lock(&lock);
    }
    omp_destroy_lock(&lock);
    return result;
}


// 4.
double scalarReduction(int num_thr, const VectorPair& p) {
    omp_set_num_threads(num_thr);

    double result = 0.0;
#pragma omp parallel for reduction(+:result)
    for (long long i = 0; i < p.v1.size(); ++i) {
        result += p.v1[i] * p.v2[i];
    }

    return result;
}


int main() {
    doExperiment(R"(D:\Projects\PyCharm_Projects\OpenMP\data\scalar_atomic.csv)", scalarAtomic);
    doExperiment(R"(D:\Projects\PyCharm_Projects\OpenMP\data\scalar_critical.csv)", scalarCritical);
    doExperiment(R"(D:\Projects\PyCharm_Projects\OpenMP\data\scalar_lock.csv)", scalarLock);
    doExperiment(R"(D:\Projects\PyCharm_Projects\OpenMP\data\scalar_reduction.csv)", scalarReduction);
    return 0;
}
