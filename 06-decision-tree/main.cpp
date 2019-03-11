#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <memory>
#include <queue>
#include <cmath>
#include <functional>

using namespace std;

typedef double feature_t;

typedef unsigned class_t;

typedef pair<unsigned, double> predicate_t;

class object {
private:
    size_t index;
    vector<feature_t> features;
    class_t class_id;
public:
    object(size_t index, vector<feature_t> features, class_t class_id) :
            index(index), features(move(features)), class_id(class_id) {}

    size_t size() const { return features.size(); }

    size_t get_index() const { return index; }

    feature_t &operator[](size_t index) { return features[index]; }

    feature_t operator[](size_t index) const { return features[index]; }

    class_t get_class() const { return class_id; }
};

namespace gain {
    int gini(const vector<object> &objects, const function<bool(const object &)> &predicate_fn) {
        int score = 0;
        for (int i = 0; i < objects.size(); ++i) {
            for (int j = i + 1; j < objects.size(); ++j) {
                const auto &x = objects[i], &y = objects[j];
                if (predicate_fn(x) == predicate_fn(y) && x.get_class() == y.get_class()) {
                    score++;
                }
            }
        }
        return score;
    }

    int donskoy(const vector<object> &objects, const function<bool(const object &)> &predicate_fn) {
        int score = 0;
        for (int i = 0; i < objects.size(); ++i) {
            for (int j = i + 1; j < objects.size(); ++j) {
                const auto &x = objects[i], &y = objects[j];
                if (predicate_fn(x) != predicate_fn(y) && x.get_class() != y.get_class()) {
                    score++;
                }
            }
        }
        return score;
    }

    int gini_donskoy(const vector<object> &objects, const function<bool(const object &)> &predicate_fn) {
        int score = 0;
        for (int i = 0; i < objects.size(); ++i) {
            for (int j = i + 1; j < objects.size(); ++j) {
                const auto &x = objects[i], &y = objects[j];
                if (predicate_fn(x) == predicate_fn(y) && x.get_class() == y.get_class()) {
                    score++;
                }
                if (predicate_fn(x) != predicate_fn(y) && x.get_class() != y.get_class()) {
                    score++;
                }
            }
        }
        return score;
    }

    double entropy(const vector<object> &objects, const function<bool(const object &)> &predicate_fn) {
        double result = 0.0;
        vector<pair<unsigned, unsigned>> distribution;
        unsigned true_objects = 0;
        for (const object &object : objects) {
            class_t obj_class = object.get_class();
            if (obj_class >= distribution.size()) distribution.resize(obj_class + 1);
            if (predicate_fn(object)) {
                distribution[obj_class].second++;
                true_objects++;
            } else {
                distribution[obj_class].first++;
            }
        }

        auto h = [](double z) { return z < 1e-10 ? 0 : -z * log2(z); };
        auto P = [&distribution](class_t c) { return distribution[c].first + distribution[c].second; };
        auto p = [&distribution](class_t c) { return distribution[c].second; };
        double p_all = true_objects;
        double l = objects.size();
        const double epsilon = 1e-10;

        for (class_t c = 0; c < distribution.size(); c++) {
            if (fabs(l) < epsilon || fabs(p_all) < epsilon || fabs(l - p_all) < epsilon) continue;
            double sum_1 = h(P(c) / l);
            double sum_2 = (p_all / l) * h(p(c) / p_all);
            double sum_3 = ((l - p_all) / l) * h((P(c) - p(c)) / (l - p_all));
            result += sum_1 - sum_2 - sum_3;
        }
        return result;
    }
}

class decision_tree {
public:
    static decision_tree make_tree(size_t features_size, size_t classes_size, const vector<object> &train_set) {
        const size_t m = features_size;
        const size_t k = classes_size;
        const size_t n = train_set.size();

        // Collect feature values along with the classes they represent
        vector<vector<pair<feature_t, class_t>>> feature_values(m);
        for (size_t feature_id = 0; feature_id < m; ++feature_id) {
            for (const object &object : train_set) {
                feature_values[feature_id].emplace_back(object[feature_id], object.get_class());
            }
            sort(feature_values[feature_id].begin(), feature_values[feature_id].end());
        }

        // Create threshold lists for use in rules
        vector<vector<double>> thresholds(m);
        for (size_t feature_id = 0; feature_id < m; ++feature_id) {
            auto &feature_values_vector = feature_values[feature_id];
            for (int i = 1; i < feature_values_vector.size(); ++i) {
                const auto &previous = feature_values_vector[i - 1];
                const auto &current = feature_values_vector[i];
                bool class_changed = previous.second != current.second;
                if (class_changed && fabs(previous.first - current.first) > epsilon) {
                    double threshold = (previous.first + current.first) / 2.0f;
                    thresholds[feature_id].push_back(threshold);
                }
            }
        }

        decision_tree root = build_tree(train_set, thresholds, 0, classes_size);

        return root;
    }

    size_t size() const { return total_size; }

    const shared_ptr<decision_tree> &get_left() const { return left; }

    const shared_ptr<decision_tree> &get_right() const { return right; }

    const class_t get_class() const { return class_id; }

    const unsigned get_feature() const { return predicate.first; }

    const double get_threshold() const { return predicate.second; }

    const bool is_leaf() const { return total_size == 1; }

    friend ostream &operator<<(ostream &out, const decision_tree &root) {
        out << fixed << setprecision(10);
        out << root.size() << endl;
        queue<shared_ptr<decision_tree>> nodes;
        nodes.emplace(make_shared<decision_tree>(root));
        int node_index = 1;
        while (!nodes.empty()) {
            const auto &node = nodes.front();
            if (node->is_leaf()) {
                out << 'C' << ' ' << node->get_class() + 1 << endl;
            } else {
                out << 'Q' << ' ' << node->get_feature() + 1 << ' ' << node->get_threshold() << ' ';
                nodes.emplace(node->right);
                out << ++node_index << ' ';
                nodes.emplace(node->left);
                out << ++node_index << endl;
            }
            nodes.pop();
        }
        return out;
    }

private:
    static constexpr auto max_level = 10;
    static constexpr auto epsilon = 1e-15;

    const shared_ptr<decision_tree> left, right;
    const size_t total_size;
    const class_t class_id;
    const predicate_t predicate;

    decision_tree(const decision_tree &left,
                  const decision_tree &right,
                  const predicate_t &predicate) : // NOLINT(modernize-pass-by-value)
            left(make_shared<decision_tree>(left)),
            right(make_shared<decision_tree>(right)),
            total_size(left.size() + right.size() + 1),
            class_id(numeric_limits<class_t>::max()),
            predicate(predicate) {}

    explicit decision_tree(class_t class_id) :
            left(nullptr),
            right(nullptr),
            total_size(1),
            class_id(class_id),
            predicate() {}

    static auto get_predicate_fn(predicate_t predicate) {
        return [predicate](const object &object) {
            return object[predicate.first] < predicate.second;
        };
    }

    static vector<unsigned> class_distribution(const vector<object> &objects, size_t classes_size) {
        vector<unsigned> distribution(classes_size);
        for (const auto &object : objects) {
            distribution[object.get_class()]++;
        }
        return distribution;
    }

    static class_t most_common_class(const vector<object> &objects, size_t classes_size) {
        vector<unsigned> class_count = class_distribution(objects, classes_size);
        auto max_iterator = max_element(class_count.begin(), class_count.end());
        return static_cast<class_t>(distance(class_count.begin(), max_iterator));
    }

    static decision_tree build_tree(const vector<object> &objects,
                                    const vector<vector<double>> &thresholds,
                                    int level, size_t classes_size) {
        if (level == max_level) {
            return decision_tree(most_common_class(objects, classes_size));
        }

        // Check if all objects belong to one class
        auto not_equal_classes = [](const object &a, const object &b) {
            return a.get_class() != b.get_class();
        };
        if (adjacent_find(objects.begin(), objects.end(), not_equal_classes) == objects.end()) {
            class_t only_class = objects[0].get_class();
            return decision_tree(only_class);
        }

        // Select predicate
        predicate_t predicate = select_predicate(objects, thresholds);
        auto predicate_fn = get_predicate_fn(predicate);

        // Split objects by predicate
        vector<object> left_objects, right_objects;
        for (const object &object : objects) {
            if (predicate_fn(object)) {
                right_objects.push_back(object);
            } else {
                left_objects.push_back(object);
            }
        }

        if (left_objects.empty() || right_objects.empty()) {
            return decision_tree(most_common_class(objects, classes_size));
        }

        // Split thresholds
        vector<vector<double>> left_thresholds = thresholds, right_thresholds = thresholds;
        auto &left_modified = left_thresholds[predicate.first];
        auto &right_modified = right_thresholds[predicate.first];
        auto left_filter = [t = predicate.second](double value) { return value < t; };
        auto right_filter = [t = predicate.second](double value) { return value >= t; };
        left_modified.erase(remove_if(left_modified.begin(), left_modified.end(), left_filter), left_modified.end());
        right_modified.erase(remove_if(right_modified.begin(), right_modified.end(), right_filter),
                             right_modified.end());

        decision_tree left = build_tree(left_objects, left_thresholds, level + 1, classes_size);
        decision_tree right = build_tree(right_objects, right_thresholds, level + 1, classes_size);

        return decision_tree(left, right, predicate);
    }

    static predicate_t select_predicate(const vector<object> &objects, const vector<vector<double>> &thresholds) {
        double best_gain = numeric_limits<double>::min();
        predicate_t best_predicate;

        for (unsigned feature_id = 0; feature_id < thresholds.size(); ++feature_id) {
            for (double threshold : thresholds[feature_id]) {
                predicate_t predicate(feature_id, threshold);
                auto predicate_fn = get_predicate_fn(predicate);
                double gain = gain::entropy(objects, predicate_fn);
                if (gain > best_gain) {
                    best_gain = gain;
                    best_predicate = predicate;
                }
            }
        }
        return best_predicate;
    }
};

void solve() {
    /**
     * m - number of features
     * k - number of classes
     * n - number of objects in training set
     */
    size_t m, k, n;
    cin >> m >> k >> n;

    vector<object> train_set;

    // Read training set
    for (unsigned object_id = 0; object_id < n; ++object_id) {
        vector<feature_t> features(m);
        for (feature_t &feature : features) {
            cin >> feature;
        }
        class_t class_id;
        cin >> class_id;
        train_set.emplace_back(object_id, features, class_id - 1);
    }

    auto tree = decision_tree::make_tree(m, k, train_set);
    cout << tree << endl;
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
