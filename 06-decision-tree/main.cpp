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

typedef tuple<feature_t, class_t, size_t> feature_value_t;

class object {
private:
    vector<feature_t> features;
    class_t class_id;
public:
    object(vector<feature_t> features, class_t class_id) :
            features(move(features)), class_id(class_id) {}

    size_t size() const { return features.size(); }

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

        vector<vector<feature_value_t>> feature_values(features_size);
        for (size_t feature_id = 0; feature_id < features_size; ++feature_id) {
            for (size_t object_id = 0; object_id < train_set.size(); ++object_id) {
                const object &object = train_set[object_id];
                feature_values[feature_id].emplace_back(object[feature_id], object.get_class(), object_id);
            }
            sort(feature_values[feature_id].begin(), feature_values[feature_id].end());
        }

        decision_tree_builder builder(classes_size, train_set);
        return builder.build_tree(0, train_set, feature_values);
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

    static constexpr predicate_t null_predicate = make_pair(0, numeric_limits<double>::min()); // NOLINT(cert-err58-cpp)

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

    class decision_tree_builder {
    private:
        const size_t classes_size;
        const vector<object> objects;

        vector<unsigned> class_distribution(const vector<object> &objects) {
            vector<unsigned> distribution(classes_size);
            for (const auto &object : objects) {
                ++distribution[object.get_class()];
            }
            return distribution;
        }

        class_t most_common_class(const vector<unsigned> &distribution) {
            auto max_iterator = max_element(distribution.begin(), distribution.end());
            return static_cast<class_t>(distance(distribution.begin(), max_iterator));
        }

        predicate_t select_predicate(const vector<vector<feature_value_t>> &feature_values,
                                     const vector<unsigned> &distribution) {
            double best_gain = numeric_limits<double>::min();
            predicate_t best_predicate = null_predicate;

            for (unsigned feature_id = 0; feature_id < feature_values.size(); ++feature_id) {
                const auto &fixed_feature_values = feature_values[feature_id];
                if (fixed_feature_values.size() < 2) continue;

                size_t true_size = 0, false_size = fixed_feature_values.size();
                vector<pair<unsigned, unsigned>> predicate_distribution(classes_size);
                for (size_t class_id = 0; class_id < classes_size; ++class_id) {
                    predicate_distribution[class_id].first = distribution[class_id];
                }

                for (size_t i = 1; i < fixed_feature_values.size(); ++i) {
                    const auto &previous_value = get<0>(fixed_feature_values[i - 1]);
                    const auto &previous_class = get<1>(fixed_feature_values[i - 1]);
                    const auto &current_value = get<0>(fixed_feature_values[i]);
                    const auto &current_class = get<1>(fixed_feature_values[i]);

                    // Add previous value
                    true_size++;
                    false_size--;
                    predicate_distribution[previous_class].first--;
                    predicate_distribution[previous_class].second++;

                    if (previous_class == current_class || fabs(previous_value - current_value) < epsilon) continue;

                    double threshold = (previous_value + current_value) / 2.0f;

                    auto h = [](double z) { return z < 1e-10 ? 0 : -z * log2(z); };
                    auto P = [&predicate_distribution](class_t c) {
                        return predicate_distribution[c].first + predicate_distribution[c].second;
                    };
                    auto p = [&predicate_distribution](class_t c) {
                        return predicate_distribution[c].second;
                    };
                    double p_all = true_size;
                    double l = true_size + false_size;

                    double entropy_gain = 0.0;
                    for (class_t c = 0; c < predicate_distribution.size(); c++) {
                        if (fabs(l) < epsilon || fabs(p_all) < epsilon || fabs(l - p_all) < epsilon) continue;
                        double sum_1 = h(P(c) / l);
                        double sum_2 = (p_all / l) * h(p(c) / p_all);
                        double sum_3 = ((l - p_all) / l) * h((P(c) - p(c)) / (l - p_all));
                        entropy_gain += sum_1 - sum_2 - sum_3;
                    }

                    if (entropy_gain > best_gain) {
                        best_gain = entropy_gain;
                        best_predicate = make_pair(feature_id, threshold);
                    }
                }
            }

            cerr << "New predicate: A[" << best_predicate.first << "] < " << best_predicate.second << ", gain = "
                 << best_gain << endl << endl;

            return best_predicate;
        }

    public:
        decision_tree_builder(const size_t classes_size, vector<object> objects) :
                classes_size(classes_size),
                objects(move(objects)) {}

        decision_tree build_tree(int level, const vector<object> &current_objects,
                                 const vector<vector<feature_value_t>> &feature_values) {
            const size_t features_size = feature_values.size();

            auto distribution = class_distribution(current_objects);

            if (level == max_level) {
                return decision_tree(most_common_class(distribution));
            }

            // Check if all objects belong to one class
            auto not_equal_classes = [](const object &a, const object &b) {
                return a.get_class() != b.get_class();
            };
            if (adjacent_find(current_objects.begin(), current_objects.end(), not_equal_classes) ==
                current_objects.end()) {
                class_t only_class = current_objects[0].get_class();
                return decision_tree(only_class);
            }

            // Select predicate
            predicate_t predicate = select_predicate(feature_values, distribution);
            auto predicate_fn = [&predicate](const object &object) {
                return object[predicate.first] < predicate.second;
            };

            if (predicate == null_predicate) {
                return decision_tree(most_common_class(distribution));
            }

            // Split objects by predicate
            vector<vector<feature_value_t>> left_feature_values(features_size), right_feature_values(features_size);
            for (size_t feature_id = 0; feature_id < features_size; ++feature_id) {
                for (const auto &feature : feature_values[feature_id]) {
                    size_t index = get<2>(feature);
                    if (predicate_fn(objects[index])) {
                        right_feature_values[feature_id].push_back(feature);
                    } else {
                        left_feature_values[feature_id].push_back(feature);
                    }
                }
            }

            size_t left_objects_size = left_feature_values[0].size(), right_objects_size = right_feature_values[0].size();

            if (left_objects_size == 0 || right_objects_size == 0) {
                return decision_tree(most_common_class(distribution));
            }

            vector<object> left_objects, right_objects;
            for (const object &object : current_objects) {
                if (predicate_fn(object)) {
                    right_objects.push_back(object);
                } else {
                    left_objects.push_back(object);
                }
            }

            decision_tree left = build_tree(level + 1, left_objects, left_feature_values);
            decision_tree right = build_tree(level + 1, right_objects, right_feature_values);

            return decision_tree(left, right, predicate);
        }
    };
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
        train_set.emplace_back(features, class_id - 1);
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
    streambuf *cin_buffer(cin.rdbuf());
    streambuf *cout_buffer(cout.rdbuf());
    cin.rdbuf(input.rdbuf());
    cout.rdbuf(output.rdbuf());
#else
    streambuf *cerr_buffer(cerr.rdbuf());
    cerr.rdbuf(nullptr);
#endif

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
