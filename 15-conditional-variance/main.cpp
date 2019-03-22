#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <cmath>

using namespace std;

void solve() {
    /**
     * k - number of unique values of feature x
     * n - number of objects
     */
    unsigned k, n;
    cin >> k >> n;

    vector<vector<int>> grouped_objects(k);
    for (unsigned i = 0; i < n; ++i) {
        unsigned x;
        int y;
        cin >> x >> y;
        grouped_objects[x - 1].push_back(y);
    }

    double answer = 0.0;
    for (const vector<int> &objects : grouped_objects) {
        double y_mean = accumulate(objects.begin(), objects.end(), 0.0) / objects.size();
        for (int y : objects) {
            answer += pow(y - y_mean, 2) / n;
        }
    }

    cout << answer << endl;
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
