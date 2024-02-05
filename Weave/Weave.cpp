#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

using namespace std;



/***** Options *****/

// The dice configuration to search for, e.g. "{12, 12, 12, 12}" means 4d12.
constexpr array kSides = { 12, 12, 12, 12 };

// Determines whether found solutions will be printed. If not, they are merely
// counted.
constexpr bool kPrintSolutions = false;

// Enforce only searching for palindromic solutions.
constexpr bool kEnforceMirror = false;

// Solution counts are reported after at least |kUpdatePerSolutions| solutions
// have been found and at least |kSecondsPerUpdate| seconds have passed since
// the last report.
constexpr int kUpdatePerSolutions = 1000;
constexpr int kSecondsPerUpdate = 1;

// If true, run the search across all available cores. If false, run only on a
// single thread.
constexpr bool kMultithreaded = true;

// How many threads to use in parallel. "thread::hardware_concurrency()" means
// use one thread per available core.
const int kNumThreads = kMultithreaded ? thread::hardware_concurrency() : 1;

// How deep into depthN to split the search across different threads. Recommend
// keeping at 3.
constexpr int kThreadDepth = 3;

// How many threaded search jobs will be queued.
const int kJobLimit = 8 * kNumThreads;

// Search from |kSearchFrom| to |kSearchTo| range of solutions at the
// |kThreadDepth| depth. Index starts at 1, not 0. These can also be overrided
// by the first two command-line arguments, respectively.
int kSearchFrom = 1;
int kSearchTo = -1; // -1 means unlimited

/*******************/



constexpr int factorial(int n) {
    return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

constexpr int fact[9] = { factorial(0), factorial(1), factorial(2), factorial(3),
  factorial(4), factorial(5), factorial(6), factorial(7), factorial(8) };

constexpr int kNumDice = kSides.size();

constexpr int num_all_perms() {
    int total = 0;
    for (int i = 0; i <= kNumDice; ++i) {
        total += factorial(kNumDice) / factorial(kNumDice - i);
    }
    return total;
}
constexpr int kNumAllPerms = num_all_perms();

constexpr array<int, kNumDice + 1> init_total_sides() {
    array<int, kNumDice + 1> total_sides{};
    total_sides[0] = 0;
    for (int i = 0; i < kNumDice; ++i) {
        total_sides[i + 1] = total_sides[i] + kSides[i];
    }
    return total_sides;
}
constexpr array<int, kNumDice + 1> kTotalSides = init_total_sides();

const vector<int> init_real_sides() {
    vector<int> real_sides(kNumDice);
    for (int i = 0; i < kNumDice; ++i) {
        real_sides[i] = kSides[i] / (kEnforceMirror ? 2 : 1);
    }
    return real_sides;
}
const vector<int> kRealSides = init_real_sides();

vector<int> targetC(kNumDice + 1);
vector<string> all_perms;
vector<vector<int>> append_map;
vector<time_t> next_times(kNumDice + 2, 0);
// TODO: change to array;
vector<vector<vector<int>>> weave_map(kNumDice + 1);

time_t minute_report;

vector<long> global_solutions(kNumDice + 1);
mutex print_mutex;
mutex solutions_mutex;
mutex jobs_mutex;
mutex time_mutex;

template <int N, int S, int I> struct ThreadData {
    string strings2[N + 1];
    int indices[N + 1][S];
    int insertCounts[N + 1][I][fact[N]];
    int currentCounts[N + 1][S + 1][fact[N]];
    int minCounts2[N + 1][S + 1][I][fact[N]];
    int maxCounts2[N + 1][S + 1][I][fact[N]];
    int minCountsSum[N + 1][S + 1][I][fact[N]];
    int maxCountsSum[N + 1][S + 1][I][fact[N]];
    int lowerBound[N + 1][S][I];
    array<array<int, S>, N + 1> upperBound;
};

queue<ThreadData<kNumDice, kSides[kNumDice - 1], kTotalSides[kNumDice] + 1>> jobs;
bool keep_searching = true;
thread_local ThreadData<kNumDice, kSides[kNumDice - 1], kTotalSides[kNumDice] + 1> td;
thread_local vector<long> solutions(kNumDice + 1);


template <typename T> void print_vector(const vector<T>& v, bool newline = false) {
    cout << "[";
    for (const auto& element : v) {
        cout << element << ", ";
    }
    cout << "\b\b] ";
    if (newline) {
        cout << endl;
    }
}

template <typename T, size_t N> void print_array(const array<T, N>& a, bool newline = false) {
    cout << "[";
    for (const auto& element : a) {
        cout << element << ", ";
    }
    cout << "\b\b] ";
    if (newline) {
        cout << endl;
    }
}

void report(int n) {
    cout << n << ' ' << global_solutions[n - 1] << ' ' << td.strings2[n - 1] << " ";
    print_vector(global_solutions, true);
}

char big_char(const string& s) {
    char max = 0;
    for (char c : s) {
        if (c > max) {
            max = c;
        }
    }
    return max;
}

bool compare_bigger_char(const string& a, const string& b) {
    return big_char(a) < big_char(b);
}

void depthN(int n);

void depthS(int n, int s) {
    if (s > 0 && s <= kRealSides[n - 1]) {
        //do counts
        int lastIndex = td.indices[n][s - 1];
        for (int i = 0; i < fact[n]; ++i) {
            int& current = td.currentCounts[n][s][i];
            current = td.currentCounts[n][s - 1][i] + td.insertCounts[n][lastIndex][i];
            if (current + td.minCountsSum[n][s][lastIndex][i] > targetC[n] ||
                current + td.maxCountsSum[n][s][lastIndex][i] < targetC[n]) {
                return;
            }
        }
    }

    if (s == kRealSides[n - 1]) {
        // Mirror the indices if needed.
        if (kEnforceMirror) {
            for (int i = 0; i < s; ++i) {
                td.indices[n][kSides[n - 1] - i - 1] = kTotalSides[n - 1] - td.indices[n][i];
            }
        }

        char c = 'a' + n - 1;
        for (int i = 0, j = 0; j < kSides[n - 1] || i < kTotalSides[n - 1];) {
            if (j < kSides[n - 1] && td.indices[n][j] == i) {
                td.strings2[n][i + j] = c;
                ++j;
            } else {
                td.strings2[n][i + j] = td.strings2[n - 1][i];
                ++i;
            }
        }

        // Next depth.
        if (n + 1 == kThreadDepth) {
            ++solutions[n];
            if (solutions[n] < kSearchFrom) {
                return;
            }
            if (kSearchTo > 0 && solutions[n] > kSearchTo) {
                return;
            }
            while (jobs.size() > kJobLimit) {
                // TODO: is this too frequent?
                this_thread::sleep_for(chrono::milliseconds(10));
            }
            jobs_mutex.lock();
            jobs.push(td);
            jobs_mutex.unlock();
        } else {
            depthN(n + 1);
        }
    } else {
        int min = td.lowerBound[n][s][s == 0 ? 0 : td.indices[n][s - 1]];
        int max = td.upperBound[n][s];

        if (n > 1 && s == 0 && kSides[n - 1] == kSides[n - 2]) {
            max = td.indices[n - 1][0];
        }

        for (int i = min; i <= max; ++i) {
            td.indices[n][s] = i;
            depthS(n, s + 1);
        }
    }
}

void depthN(int n) {
    ++solutions[n - 1];

    if (solutions[n - 1] % kUpdatePerSolutions == 0) {
        if (n >= kThreadDepth) {
            solutions_mutex.lock();
            for (int i = kThreadDepth - 1; i <= kNumDice; ++i) {
                global_solutions[i] += solutions[i];
            }
            solutions_mutex.unlock();
            for (int i = kThreadDepth - 1; i <= kNumDice; ++i) {
                solutions[i] = 0;
            }
        }
        auto nt = time(NULL);
        if (nt >= next_times[n]) {
            time_mutex.lock();
            if (nt >= next_times[n]) {
                print_mutex.lock();
                cout << (nt - next_times[n] + kSecondsPerUpdate) << " ";
                report(n);
                print_mutex.unlock();
                next_times[n] = nt + kSecondsPerUpdate; // Only once per |kSecondsPerUpdate| seconds.
            }
            time_mutex.unlock();
        }
    }

    if (n > kNumDice) {
        // Found a solution.
        if (kPrintSolutions) {
            print_mutex.lock();
            cout << td.strings2[n - 1] << endl;
            print_mutex.unlock();
        }
        return;
    }

    const string& s2 = td.strings2[n - 1];

    const int slen = kTotalSides[n - 1];
    const int slenmin1 = slen - 1;
    const int insertPoints = slen + 1;
    const int halfPoints = insertPoints / 2 + 1;

    int countsBeforeInt[insertPoints][kNumAllPerms + 1];
    int countsAfterInt[insertPoints][kNumAllPerms + 1];

    for (int i = 0; i != insertPoints; ++i) {
        for (int j = 0; j != kNumAllPerms + 1; ++j) {
            countsBeforeInt[i][j] = 0;
            countsAfterInt[i][j] = 0;
        }
    }

    countsBeforeInt[0][0] = 1;
    countsAfterInt[0][0] = 1;

    //do the before and after counts
    for (int i = 0; i < slen; ++i) {
        int cb = s2[i] - 'a';
        int ca = s2[slenmin1 - i] - 'a';
        for (int j = 0; j < kNumAllPerms; ++j) {
            countsBeforeInt[i + 1][j] += countsBeforeInt[i][j];
            countsBeforeInt[i + 1][append_map[j][cb]] += countsBeforeInt[i][j];
            countsAfterInt[i + 1][j] += countsAfterInt[i][j];
            countsAfterInt[i + 1][append_map[j][ca]] += countsAfterInt[i][j];
        }
    }

    //merge the 2 count
    for (int i = 0; i < insertPoints; ++i) {
        for (int j = 0; j < fact[n]; ++j) {
            int cb = countsBeforeInt[i][weave_map[n][j][0]];
            int ca = countsAfterInt[insertPoints - i - 1][weave_map[n][j][1]];
            td.insertCounts[n][i][j] = cb * ca;
        }
    }

    //if we enforce mirroring, load all the counts to the first half
    if (kEnforceMirror) {
        //set the data to include it's mirror
        for (int i = 0; i < halfPoints; ++i) {
            int ii = insertPoints - i - 1;
            for (int j = 0; j < fact[n]; ++j) {
                td.insertCounts[n][i][j] += td.insertCounts[n][ii][j];
            }
        }

        // zero the data of the mirrors
        for (int i = halfPoints; i < insertPoints; ++i) {
            for (int j = 0; j < fact[n]; ++j) {
                td.insertCounts[n][i][j] = 0;
            }
        }
    }


    //calc the min/max
    for (int currentS = kRealSides[n - 1] - 1; currentS >= 0; --currentS) {
        int max = td.upperBound[n][currentS];
        for (int lastIndex = 0; lastIndex < insertPoints; ++lastIndex) {
            int min = td.lowerBound[n][currentS][lastIndex];
            if (lastIndex > 0 && min == td.lowerBound[n][currentS][lastIndex - 1] || lastIndex > max) {
                copy(begin(td.minCounts2[n][currentS][lastIndex - 1]), end(td.minCounts2[n][currentS][lastIndex - 1]), begin(td.minCounts2[n][currentS][lastIndex]));
                copy(begin(td.maxCounts2[n][currentS][lastIndex - 1]), end(td.maxCounts2[n][currentS][lastIndex - 1]), begin(td.maxCounts2[n][currentS][lastIndex]));
                copy(begin(td.minCountsSum[n][currentS][lastIndex - 1]), end(td.minCountsSum[n][currentS][lastIndex - 1]), begin(td.minCountsSum[n][currentS][lastIndex]));
                copy(begin(td.maxCountsSum[n][currentS][lastIndex - 1]), end(td.maxCountsSum[n][currentS][lastIndex - 1]), begin(td.maxCountsSum[n][currentS][lastIndex]));
                continue;
            }
            for (int i = 0; i < fact[n]; ++i) {
                int& min_count = td.minCounts2[n][currentS][lastIndex][i];
                int& max_count = td.maxCounts2[n][currentS][lastIndex][i];
                min_count = 1000000;
                max_count = -1000000;

                for (int j = min; j <= max; ++j) {
                    const auto& insert_count = td.insertCounts[n][j][i];
                    if (min_count > insert_count) {
                        min_count = insert_count;
                    }
                    if (max_count < insert_count) {
                        max_count = insert_count;
                    }
                }
                if (currentS > 0) {
                    td.minCountsSum[n][currentS][lastIndex][i] =
                        td.minCountsSum[n][currentS + 1][lastIndex][i]
                        + td.minCounts2[n][currentS][lastIndex][i];
                    td.maxCountsSum[n][currentS][lastIndex][i] =
                        td.maxCountsSum[n][currentS + 1][lastIndex][i]
                        + td.maxCounts2[n][currentS][lastIndex][i];
                }
            }
        }
    }

    depthS(n, 0);
}

void threaded_search() {
    while (true) {
        jobs_mutex.lock();
        if (jobs.empty()) {
            jobs_mutex.unlock();
            if (keep_searching) {
                //cout << "worker thread wait" << endl;
                this_thread::sleep_for(chrono::milliseconds(1));
                continue;
            }
            break;
        }
        td = move(jobs.front());
        jobs.pop();
        jobs_mutex.unlock();
        depthN(kThreadDepth);
    }

    // Count the solutions from this thread before exiting.
    solutions_mutex.lock();
    for (int i = kThreadDepth - 1; i <= kNumDice; ++i) {
        global_solutions[i] += solutions[i];
    }
    solutions_mutex.unlock();
}

void search() {
    for (int n = 1; n <= kNumDice; ++n) {
        int insertPoints = kTotalSides[n] + 1;

        for (int s = 0; s < kSides[n - 1]; ++s) {
            int max = kTotalSides[n - 1];
            if (kEnforceMirror) {
                max /= 2;
            }

            td.upperBound[n][s] = max;
            for (int i = 0; i < insertPoints; ++i) {
                td.lowerBound[n][s][i] = i;
            }
        }
        cout << "bounds n=" << n << endl;
        for (int s = 0; s < kSides[n - 1]; ++s) {
            cout << td.lowerBound[n][s][0] << " ";
        }
        cout << endl;
        print_array(td.upperBound[n], true);
    }

    for (int n = 0, product = 1; n <= kNumDice; ++n) {
        targetC[n] = product / fact[n];
        if (n < kNumDice) {
            product *= kSides[n];
        }
    }

    cout << "----------------------" << endl << "Sides: ";
    print_array(kSides, true);

    depthN(1);

    keep_searching = false;
}

void init() {
    for (int i = 0; i < kNumDice; ++i) {
        td.strings2[i + 1] = string(kTotalSides[i + 1], '.');
    }

    vector<int> perm_index(kNumDice + 2, 0);
    perm_index[1] = 1;

    vector<vector<string>> perms;
    vector<string> t;
    t.push_back("");
    perms.push_back(t);

    for (int i = 0; i < kNumDice; ++i) {
        t = perms[i];
        vector<string> t2;
        for (const string& s : t) {
            for (int j = 0; j < kNumDice; ++j) {
                char c = 'a' + j;
                if (s.find(c) == string::npos) {
                    string s2 = s + c;
                    t2.push_back(s2);
                }
            }
        }
        perm_index[i + 2] = perm_index[i + 1] + t2.size();
        cout << t2.size() << " ";
        print_vector(t2, true);
        perms.push_back(t2);
    }
    cout << "        - ";
    print_vector(perm_index, true);

    all_perms = vector<string>();
    all_perms.reserve(kNumAllPerms);
    //sort
    for (int i = 0; i <= kNumDice; ++i) {
        vector<string>& t = perms[i];
        sort(t.begin(), t.end(), compare_bigger_char);
        cout << t.size() << " ";
        print_vector(t, true);
        all_perms.insert(all_perms.end(), t.begin(), t.end());
    }
    cout << endl;
    print_vector(perm_index, true);
    print_vector(all_perms, true);
    cout << kNumAllPerms << endl;

    //build map
    // TODO: change to array. Also swap the dimensions.
    append_map = vector<vector<int>>(kNumAllPerms + 1, vector<int>(kNumDice));
    for (int i = 0; i < kNumAllPerms; ++i) {
        for (int j = 0; j < kNumDice; ++j) {
            string s = all_perms[i];
            s.push_back('a' + j);
            int k;
            for (k = 0; k < kNumAllPerms; ++k) {
                if (all_perms[k] == s) {
                    break;
                }
            }
            append_map[i][j] = k;
        }
    }
    print_vector(append_map[0], true);
    print_vector(append_map[1], true);

    //weave mapper
    for (int n = 0; n <= kNumDice; ++n) {
        int size = perms[n].size();
        // TODO: Change to array and optimize.
        weave_map[n] = vector<vector<int>>(size, vector<int>(2));

        // TODO: There are better ways to do this.
        char c = 'a' + n - 1;
        for (int a = 0; a < perm_index[n]; ++a) {
            for (int b = 0; b < perm_index[n]; ++b) {
                string rev = all_perms[a];
                reverse(rev.begin(), rev.end());
                string s = all_perms[b] + c + rev;
                int si = 0;
                for (; si < perms[n].size(); ++si) {
                    if (perms[n][si] == s) {
                        break;
                    }
                }
                if (si != perms[n].size()) {
                    weave_map[n][si][0] = b;
                    weave_map[n][si][1] = a;
                }
            }
        }

        cout << "weave " << n << " " << size << endl;
        for (int si = 0; si < perms[n].size(); ++si) {
            cout << weave_map[n][si][0] << " ";
        }
        cout << endl;
        for (int si = 0; si < perms[n].size(); ++si) {
            cout << weave_map[n][si][1] << " ";
        }
        cout << endl;
    }

    global_solutions[1] = global_solutions[0] = 1;
    if (kThreadDepth <= kNumDice) {
        global_solutions[kThreadDepth - 1] = kSearchFrom - 1;
    }
}

int main(int argc, char** argv) {
    auto start_time = time(NULL);

    // Override the search range.
    if (argc > 1) {
        kSearchFrom = atoi(argv[1]);
    }
    if (argc > 2) {
        kSearchTo = atoi(argv[2]);
    }

    init();

    // Start up the threads.
    vector<thread> threads;
    for (int i = 0; i != kNumThreads; ++i) {
        threads.emplace_back(threaded_search);
    }

    search();

    // Wait for the threads.
    while (threads.size()) {
        threads.back().join();
        threads.pop_back();
    }

    cout << "Search Complete - ";
    print_vector(global_solutions);
    cout << "- " << (time(NULL) - start_time) << " sec - " << endl;
    cout << "Sides: ";
    print_array(kSides, true);
    cout << "Mirroring: " << kEnforceMirror << endl;

    return 0;
}