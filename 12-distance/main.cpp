#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

using namespace std;

long long distance(vector<int> objects) {
    sort(objects.begin(), objects.end());

    long long result = 0;
    for (long long i = 0; i < objects.size(); ++i) {
        result += i * objects[i] - (objects.size() - 1 - i) * objects[i];
    }

    return result * 2;
}

void solve() {
    /**
     * k - number of unique values of feature y
     * n - number of objects
     */
    unsigned k, n;
    cin >> k >> n;

    vector<int> objects;
    vector<vector<int>> grouped_objects(k);

    for (unsigned i = 0; i < n; ++i) {
        int x;
        unsigned y;
        cin >> x >> y;
        objects.push_back(x);
        grouped_objects[y - 1].push_back(x);
    }

    long long intra_class_distance = 0;
    for (const vector<int> &intra_class : grouped_objects) {
        intra_class_distance += distance(intra_class);
    }
    long long inter_class_distance = distance(objects) - intra_class_distance;

    cout << intra_class_distance << endl;
    cout << inter_class_distance << endl;
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
