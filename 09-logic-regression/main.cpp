#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

void solve() {
    /**
     * m - number of variables in boolean function
     */
    size_t var_size;
    cin >> var_size;

    size_t two_power_m = 1u << var_size;

    // Build perfect DNF by the given truth table
    vector<unsigned> dnf;
    for (unsigned i = 0, result; i < two_power_m; ++i) {
        cin >> result;
        if (result) dnf.push_back(i);
    }

    // If function is identically zero, print one layer and exit
    if (dnf.empty()) {
        cout << 2 << ' ' << var_size << ' ' << 1 << endl;
        for (unsigned i = 0; i < var_size; ++i) {
            cout << 0 << ' ';
        }
        cout << -0.5;
        return;
    }

    // Print conjunction layer
    cout << 3 << ' ' << var_size << ' ' << dnf.size() << ' ' << 1 << endl;
    for (unsigned conjunction : dnf) {
        unsigned bit_count = 0;
        for (unsigned j = 0; j < var_size; ++j) {
            cout << (conjunction & 1u ? 1 : -1) << ' ';
            bit_count += conjunction & 1u;
            conjunction >>= 1;
        }
        cout << 0.5 - bit_count << endl;
    }

    // Print disjunction layer
    for (unsigned i = 0; i < dnf.size(); ++i) {
        cout << 1 << ' ';
    }
    cout << -0.5;
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
