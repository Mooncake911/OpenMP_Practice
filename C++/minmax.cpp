#include <omp.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <limits>
#include <chrono>
#include <functional>

using namespace std;
using namespace chrono;

struct Result {
    double min = numeric_limits<double>::max();
    double max = numeric_limits<double>::min();
};

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
void doExperiment(const string& filename, const function<Result(int, const vector<double>&)>& func) {
    ofstream csv_file(filename);
    if (!csv_file.is_open()) {
        cerr << "Open file error!" << endl;
        return;
    }
    csv_file << "Num_Threads,Iter,Time\n";

    for (long long j = 90; j <= 9000000; j *= 10){
        vector<double> data = rand_vector(j, 1 , 1000);
        for (int i = 1; i <= 16; ++i) {
            auto start_time = high_resolution_clock::now();
            func(i, data);
            auto end_time = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end_time - start_time);
            csv_file << i << "," << j << "," << duration.count() << "\n";
        }
    }
    csv_file.close();
}



Result MinMaxAtomic(int num_thr, const vector<double>& data) {
    // Don't support min/max
    omp_set_num_threads(num_thr);
    Result result;

#pragma omp parallel
    {
#pragma omp for
        for (long long i = 1; i < data.size(); ++i) {
//#pragma omp atomic
            result.min = std::min(result.min, data[i]);
//#pragma omp atomic
            result.max = std::max(result.max, data[i]);
        }
    }

    return result;
}


Result MinMaxCritical(int num_thr, const vector<double>& data) {
    omp_set_num_threads(num_thr);
    Result result;

#pragma omp parallel
    {
#pragma omp for
        for (long long i = 1; i < data.size(); ++i) {
#pragma omp critical
            {
                result.min = std::min(result.min, data[i]);
                result.max = std::max(result.max, data[i]);
            }
        }
    }

    return result;
}


Result MinMaxLock(int num_thr, const vector<double>& data) {
    omp_set_num_threads(num_thr);
    Result result;
    omp_lock_t lock;
    omp_init_lock(&lock);

#pragma omp parallel
    {
#pragma omp for
        for (long long i = 1; i < data.size(); ++i) {
            omp_set_lock(&lock);
            result.min = std::min(result.min, data[i]);
            result.max = std::max(result.max, data[i]);
            omp_unset_lock(&lock);
        }
    }
    omp_destroy_lock(&lock);

    return result;
}


Result MinMaxReduction(int num_thr, const vector<double>& data) {
    omp_set_num_threads(num_thr);
    Result result;
    double min_value = numeric_limits<double>::max();
    double max_value = numeric_limits<double>::min();

#pragma omp parallel for reduction(min:min_value) reduction(max:max_value)
    for (long long i = 1; i < data.size(); ++i) {
        min_value = std::min(min_value, data[i]);
        max_value = std::max(max_value, data[i]);
    }

    result.min = min_value;
    result.max = max_value;

    return result;
}


int main() {
    //doExperiment(R"(D:\Projects\PyCharm_Projects\OpenMP\data\minmax_atomic.csv)", MinMaxAtomic);
    doExperiment(R"(D:\Projects\PyCharm_Projects\OpenMP\data\minmax_critical.csv)", MinMaxCritical);
    doExperiment(R"(D:\Projects\PyCharm_Projects\OpenMP\data\minmax_lock.csv)", MinMaxLock);
    doExperiment(R"(D:\Projects\PyCharm_Projects\OpenMP\data\minmax_reduction.csv)", MinMaxReduction);
    return 0;
}