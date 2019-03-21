#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <map>
#include <iomanip>

using namespace std;

void solve() {
    /**
     * k1 - number of unique values of feature 1
     * k2 - number of unique values of feature 2
     * n - number of objects
     */
    size_t rows_size, columns_size, objects_size;
    cin >> rows_size >> columns_size >> objects_size;

    map<pair<unsigned, unsigned>, unsigned> contingency_table;
    vector<unsigned> row_sum(rows_size), column_sum(columns_size);

    for (unsigned i = 0; i < objects_size; ++i) {
        unsigned row, column;
        cin >> row >> column;
        --row;
        --column;
        auto key = make_pair(row, column);
        ++contingency_table[key];
        ++row_sum[row];
        ++column_sum[column];
    }

    double answer = 0.0;
    vector<unsigned> row_present_columns_sum(rows_size, 0);

    // Add non-zero cells to answer
    for (const auto &[key, value] : contingency_table) {
        auto[row, column] = key;
        double observed_frequency = value;
        double expected_frequency = row_sum[row] * column_sum[column] / static_cast<double>(objects_size);
        if (expected_frequency > 0) {
            answer += pow(observed_frequency - expected_frequency, 2) / expected_frequency;
        }
        row_present_columns_sum[row] += column_sum[column];
    }

    // Add zero cells to answer
    for (unsigned row = 0; row < rows_size; ++row) {
        answer += row_sum[row] * (objects_size - row_present_columns_sum[row]) / static_cast<double>(objects_size);
    }

    cout << setprecision(10) << answer;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

#ifdef DEBUG
    ifstream input("input.txt");
    ofstream output("output.txt");
    streambuf *cin_buffer(cin.rdbuf());
    streambuf *cout_buffer(cout.rdbuf());
    cin.rdbuf(input.rdbuf());
    cout.rdbuf(output.rdbuf());
#else
    streambuf *cerr_buffer(cerr.rdbuf());
    cerr.rdbuf(nullptr);
#endif

    cout << fixed;
    solve();
    cout.flush();

#ifdef DEBUG
    cin.rdbuf(cin_buffer);
    cout.rdbuf(cout_buffer);
    input.close();
    output.close();
#else
    cerr.rdbuf(cerr_buffer);
#endif
    return 0;
}
