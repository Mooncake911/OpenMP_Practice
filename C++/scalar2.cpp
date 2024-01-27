#include <omp.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <functional>

using namespace std;
using namespace chrono;

struct VectorPair {
    vector<double> v1;
    vector<double> v2;
};

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
void doExperiment(const string& filename, const function<double(const vector<double>&, const vector<double>&)>& func) {
    ofstream csv_file(filename);
    if (!csv_file.is_open()) {
        cerr << "Open file error!" << endl;
        return;
    }
    csv_file << "Num_Threads,Iter,Time\n";

    for (int i = 1; i <= 16; ++i) {
        omp_set_num_threads(i);
        vector<VectorPair> pairs(6);

#pragma omp parallel sections
        {
#pragma omp section
            {
                int k = 0;
                for (auto j = 10; j <= 1000000; j *= 10) {
                    VectorPair pair;
                    pair.v1 = rand_vector(j, 1, 1000);
                    pair.v2 = rand_vector(j, 1, 1000);

#pragma omp critical
                    {
                        pairs[k] = pair;
                        k += 1;
                    }

                }
            }

#pragma omp section
            {
                for (auto & pair : pairs) {
                    while (pair.v1.empty() || pair.v2.empty()){} // wait when pair will be initialized
                    auto start_time = omp_get_wtime();
                    func(pair.v1, pair.v2);
                    auto end_time = omp_get_wtime();
                    auto duration = (end_time - start_time) * 1e6;

#pragma omp critical
                    {
                        csv_file << i << "," << pair.v1.size() << "," << duration << "\n";
                    }

                }
            }
        }

    }

    csv_file.close();
}


// 1.
double scalarSections(const vector<double>& v1, const vector<double>& v2) {

    double result = 0.0;
#pragma omp parallel for reduction(+:result)
    for (long long i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }

    return result;
}


int main() {
    doExperiment(R"(D:\Projects\PyCharm_Projects\OpenMP\data\scalar_sections.csv)", scalarSections);
    return 0;
}
