#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cmath>

using namespace std;

typedef int feature_t;

class object {
private:
    vector<feature_t> features;

public:
    vector<double> rank;

    explicit object(const vector<feature_t> &features) : features(features), rank(features.size(), 0.0) {}

    size_t size() const { return features.size(); }

    feature_t operator[](size_t index) const { return features[index]; }
};

void assign_ranks(vector<object> &objects, unsigned feature_id) {
    sort(objects.begin(), objects.end(), [feature_id](const object &a, const object &b) {
        return a[feature_id] < b[feature_id];
    });

    objects[0].rank[feature_id] = 1;
    unsigned equal_rank_sum = 1, equal_size = 1;
    for (unsigned i = 1; i < objects.size(); ++i) {
        object &previous = objects[i - 1], &current = objects[i];

        unsigned current_rank = i + 1;
        if (previous[feature_id] == current[feature_id]) {
            equal_rank_sum += current_rank;
            ++equal_size;
            continue;
        }

        if (equal_size > 1) {
            for (unsigned j = i - equal_size; j < i; ++j) {
                objects[j].rank[feature_id] = equal_rank_sum / static_cast<double>(equal_size);
            }
        }
        objects[i].rank[feature_id] = current_rank;
        equal_rank_sum = current_rank;
        equal_size = 1;
    }
}

void solve() {
    /**
     * n - number of objects
     */
    unsigned objects_size;
    cin >> objects_size;

    vector<object> objects;
    for (unsigned i = 0; i < objects_size; ++i) {
        vector<feature_t> x(2);
        cin >> x[0] >> x[1];
        objects.emplace_back(x);
    }

    if (objects_size == 1) {
        cout << 1 << endl;
        return;
    }

    assign_ranks(objects, 0);
    assign_ranks(objects, 1);

    double difference_sum = 0.0;
    for (const auto &object : objects) {
        difference_sum += pow(object.rank[0] - object.rank[1], 2);
    }

    cout << setprecision(10) << 1 - 6 * difference_sum / (objects_size * (pow(objects_size, 2) - 1));
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
