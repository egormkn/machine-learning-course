#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <map>
#include <iomanip>

using namespace std;

void solve() {
    /**
     * kx - number of unique values of feature x
     * ky - number of unique values of feature y
     * n - number of objects
     */
    unsigned rows_size, columns_size, objects_size;
    cin >> rows_size >> columns_size >> objects_size;

    map<pair<unsigned, unsigned>, unsigned> contingency_table;

    vector<unsigned> row_sum(rows_size);

    for (unsigned i = 0; i < objects_size; ++i) {
        unsigned row, column;
        cin >> row >> column;
        --row;
        --column;
        auto key = make_pair(row, column);
        ++contingency_table[key];
        ++row_sum[row];
    }

    double answer = 0.0;
    for (const auto &[key, value] : contingency_table) {
        auto[i, j] = key;
        if (value == 0 || row_sum[i] == 0) continue;
        double p = value / static_cast<double>(objects_size);
        double log_p = log(value / static_cast<double>(row_sum[i]));
        answer -= p * log_p;
    }

    cout << setprecision(10) << answer << endl;
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
