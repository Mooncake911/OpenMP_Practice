#include <omp.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <limits>
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
void doExperiment(const string& filename, const function<long long(int, const vector<double>&)>& func) {
    ofstream csv_file(filename);
    if (!csv_file.is_open()) {
        cerr << "Open file error!" << endl;
        return;
    }
    csv_file << "Num_Threads,Iter,Time\n";

    long long t;
    for (long long j = 90; j <= 9000000; j *= 10){
        vector<double> data = rand_vector(j, 1 , 1000);
        for (int i = 1; i <= 16; ++i) {
            t = func(i, data);
            csv_file << i << "," << j << "," << t << "\n";
        }
    }
    csv_file.close();
}



long long MinMaxAtomic(int num_thr, const vector<double>& data) {
    // Don't support min/max
    omp_set_num_threads(num_thr);

    double min_value = numeric_limits<double>::max();
    double max_value = numeric_limits<double>::min();

    auto start_time = chrono::high_resolution_clock::now();

//#pragma omp parallel
//    {
//#pragma omp for
//        for (long long i = 1; i < data.size(); ++i) {
//#pragma omp atomic
//            min_value = std::min(min_value, data[i]);
//
//#pragma omp atomic
//            max_value = std::max(max_value, data[i]);
//        }
//    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);

    cout << "Min: " << min_value << endl;
    cout << "Max: " << max_value << endl;
    cout << "Duration (microseconds): " << duration.count() << endl;

    return duration.count();
}


long long MinMaxCritical(int num_thr, const vector<double>& data) {
    omp_set_num_threads(num_thr);

    double min_value = numeric_limits<double>::max();
    double max_value = numeric_limits<double>::min();

    auto start_time = chrono::high_resolution_clock::now();

#pragma omp parallel
    {
#pragma omp for
        for (long long i = 1; i < data.size(); ++i) {
#pragma omp critical
            {
                min_value = std::min(min_value, data[i]);
                max_value = std::max(max_value, data[i]);
            }
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);

//    cout << "Min: " << min_value << endl;
//    cout << "Max: " << max_value << endl;
//    cout << "Duration (microseconds): " << duration.count() << endl;

    return duration.count();
}


long long MinMaxLock(int num_thr, const vector<double>& data) {
    omp_set_num_threads(num_thr);

    double min_value = numeric_limits<double>::max();
    double max_value = numeric_limits<double>::min();

    auto start_time = chrono::high_resolution_clock::now();

    omp_lock_t lock;
    omp_init_lock(&lock);
#pragma omp parallel
    {
#pragma omp for
        for (long long i = 1; i < data.size(); ++i) {
            omp_set_lock(&lock);
            min_value = std::min(min_value, data[i]);
            max_value = std::max(max_value, data[i]);
            omp_unset_lock(&lock);
        }
    }
    omp_destroy_lock(&lock);

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);

//    cout << "Min: " << min_value << endl;
//    cout << "Max: " << max_value << endl;
//    cout << "Duration (microseconds): " << duration.count() << endl;

    return duration.count();
}


long long MinMaxReduction(int num_thr, const vector<double>& data) {
    omp_set_num_threads(num_thr);

    double min_value = numeric_limits<double>::max();
    double max_value = numeric_limits<double>::min();

    auto start_time = chrono::high_resolution_clock::now();

#pragma omp parallel for reduction(min:min_value) reduction(max:max_value)
    for (long long i = 1; i < data.size(); ++i) {
        min_value = std::min(min_value, data[i]);
        max_value = std::max(max_value, data[i]);
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);

//    cout << "Min: " << min_value << endl;
//    cout << "Max: " << max_value << endl;
//    cout << "Duration (microseconds): " << duration.count() << endl;

    return duration.count();
}


int main() {
    //doExperiment(R"(D:\Projects\PyCharm_Projects\OpenMP\data\minmax_atomic.csv)", MinMaxAtomic);
    doExperiment(R"(D:\Projects\PyCharm_Projects\OpenMP\data\minmax_critical.csv)", MinMaxCritical);
    doExperiment(R"(D:\Projects\PyCharm_Projects\OpenMP\data\minmax_lock.csv)", MinMaxLock);
    doExperiment(R"(D:\Projects\PyCharm_Projects\OpenMP\data\minmax_reduction.csv)", MinMaxReduction);
    return 0;
}