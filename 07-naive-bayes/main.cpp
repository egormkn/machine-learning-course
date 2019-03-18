#include <utility>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cmath>
#include <unordered_set>

using namespace std;

typedef unsigned feature_t;

typedef unsigned class_t;

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

    vector<feature_t>::const_iterator begin() const { return features.begin(); }

    vector<feature_t>::const_iterator end() const { return features.end(); }
};

class naive_bayes {
public:
    static unique_ptr<naive_bayes>
    make_classifier(const vector<object> &train_set, const vector<double> &class_importance, size_t features_max) {
        const auto &objects_size = train_set.size();
        const auto &classes_size = class_importance.size();

        vector<unsigned> class_distribution(classes_size, 0);
        vector<double> class_probability(classes_size, 0.0);
        vector<vector<double>> feature_probability(features_max, vector<double>(classes_size, 0.001));

        for (const object &object : train_set) {
            class_t class_id = object.get_class();

            class_distribution[class_id]++;
            class_probability[class_id] += 1.0 / objects_size;

            unordered_set<feature_t> unique_features;
            for (const feature_t &feature : object) {
                unique_features.insert(feature);
            }
            for (const feature_t &feature : unique_features) {
                feature_probability[feature][class_id] += 1.0;
            }
        }

        for (vector<double> &feature_probability_classes : feature_probability) {
            for (unsigned class_id = 0; class_id < classes_size; ++class_id) {
                feature_probability_classes[class_id] /= class_distribution[class_id];
            }
        }

        return make_unique<naive_bayes>(class_probability, class_importance, feature_probability);
    }

    class_t get_class(const object &object) {
        const auto &classes_size = class_probability.size();

        double best_score = numeric_limits<double>::lowest();
        unsigned best_class_id = 0;

        for (unsigned class_id = 0; class_id < classes_size; ++class_id) {
            double score = log(class_importance[class_id] * class_probability[class_id]);
            for (const feature_t &feature : object) {
                score += log(feature_probability[feature][class_id]);
            }
            if (score > best_score) {
                best_score = score;
                best_class_id = class_id;
            }
        }

        return best_class_id;
    }

    naive_bayes(vector<double> class_probability, vector<double> class_importance,
                vector<vector<double>> feature_probability) :
            class_probability(move(class_probability)), class_importance(move(class_importance)),
            feature_probability(move(feature_probability)) {}

private:
    const vector<double> class_probability, class_importance;
    const vector<vector<double>> feature_probability;
};

void solve() {
    /**
     * n - number of objects in training set
     */
    size_t n;
    cin >> n;

    vector<object> train_set;

    // Read training set
    for (unsigned object_id = 0; object_id < n; ++object_id) {
        size_t k;
        char c;
        cin >> k >> c;
        vector<feature_t> features(k);
        for (feature_t &feature : features) {
            cin >> feature;
        }
        train_set.emplace_back(features, c == 'S' ? 0 : 1);
    }

    /**
     * t - number of objects in training set
     */
    size_t t;
    cin >> t;

    vector<object> test_set;

    // Read test set
    for (unsigned object_id = 0; object_id < t; ++object_id) {
        size_t k;
        cin >> k;
        vector<feature_t> features(k);
        for (feature_t &feature : features) {
            cin >> feature;
        }
        test_set.emplace_back(features, -1);
    }

    auto bayes = naive_bayes::make_classifier(train_set, {1e-30, 1e30}, 1000001);
    for (const object &object : test_set) {
        cout << (bayes->get_class(object) == 0 ? 'S' : 'L') << endl;
    }
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
