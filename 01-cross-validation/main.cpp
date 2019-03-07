#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

void solve() {
    /**
     * n - number of objects
     * m - number of classes
     * k - number of parts
     */
    unsigned n, m, k;
    cin >> n >> m >> k;

    vector<vector<int>> classes(m), parts(k);

    for (unsigned object = 0, class_id; object < n; object++) {
        cin >> class_id;
        classes[--class_id].push_back(object);
    }

    int current_part = 0;
    for (auto &cls : classes) {
        auto size = static_cast<unsigned>(cls.size());
        auto block_size = size / k;
        for (int i = 0; i < size; i++) {
            bool extra = i >= size - size % k;
            parts[extra ? current_part : (i / block_size)].push_back(cls[i]);
            if (extra) current_part = (current_part + 1) % k;
        }
    }

    for (auto &part : parts) {
        auto size = static_cast<unsigned>(part.size());
        cout << size << ' ';
        for (auto &object : part) {
            cout << object + 1 << ' ';
        }
        cout << endl;
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

#ifdef DEBUG
    ifstream input("input.txt");
    ofstream output("output.txt");
    cin.rdbuf(input.rdbuf());
    cout.rdbuf(output.rdbuf());
#endif

    solve();
    cout.flush();

#ifdef DEBUG
    input.close();
    output.close();
#endif
    return 0;
}
