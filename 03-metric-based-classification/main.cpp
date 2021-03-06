#include <utility>
#include <random>
#include <iostream>
#include <fstream>
#include <functional>
#include <sstream>
#include <iomanip>
#include <memory>

using namespace std;

typedef double feature_t;

typedef int class_t;

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

    vector<feature_t>::const_iterator begin() const { return features.begin(); }

    vector<feature_t>::const_iterator end() const { return features.end(); }
};

typedef function<double(const object &, const object &)> metric_t;

namespace metric {

    const double epsilon = 1e-15;

    // sum [a_i == b_i]
    double hamming(const object &a, const object &b) {
        const size_t m = min(a.size(), b.size());
        double result = 0;
        for (size_t i = 0; i < m; ++i) {
            result += fabs(a[i] - b[i]) < epsilon ? 1 : 0;
        }
        return result;
    }

    // sqrt(sum (a_i - b_i)^2)
    double euclidean(const object &a, const object &b) {
        const size_t m = min(a.size(), b.size());
        double result = 0;
        for (size_t i = 0; i < m; ++i) {
            result += pow(a[i] - b[i], 2.0f);
        }
        return sqrt(result);
    }

    // max |a_i - b_i|
    double chebyshev(const object &a, const object &b) {
        const size_t m = min(a.size(), b.size());
        double result = 0;
        for (size_t i = 0; i < m; ++i) {
            result = max(result, fabs(a[i] - b[i]));
        }
        return result;
    }

    // sum |a_i - b_i|
    double manhattan(const object &a, const object &b) {
        const size_t m = min(a.size(), b.size());
        double result = 0;
        for (size_t i = 0; i < m; ++i) {
            result += fabs(a[i] - b[i]);
        }
        return result;
    }

    // sum (ln (1 + |a_i - b_i|))
    double lorentzian(const object &a, const object &b) {
        const size_t m = min(a.size(), b.size());
        double result = 0;
        for (size_t i = 0; i < m; ++i) {
            result += log(1 + fabs(a[i] - b[i]));
        }
        return result;
    }

    // sum |a_i - b_i|/(|a_i| + |b_i|)
    double canberra(const object &a, const object &b) {
        const size_t m = min(a.size(), b.size());
        double result = 0;
        for (size_t i = 0; i < m; ++i) {
            double denominator = fabs(a[i]) + fabs(b[i]);
            result += denominator < epsilon ? 0 : fabs(a[i] - b[i]) / denominator;
        }
        return result;
    }

    // sum |a_i - b_i| / sum (a_i + b_i)
    double sorensen(const object &a, const object &b) {
        const size_t m = min(a.size(), b.size());
        double numerator = 0, denominator = 0;
        for (size_t i = 0; i < m; ++i) {
            numerator += fabs(a[i] - b[i]);
            denominator += a[i] + b[i];
        }
        return denominator < epsilon ? 0.0 : numerator / denominator;
    }

    // 2 * sum ((a_i - b_i)^2 / (a_i + b_i)^2)
    double divergence(const object &a, const object &b) {
        const size_t m = min(a.size(), b.size());
        double result = 0;
        for (size_t i = 0; i < m; ++i) {
            double denominator = pow(a[i] + b[i], 2.0f);
            result += denominator < epsilon ? 0 : pow(a[i] - b[i], 2.0f) / denominator;
        }
        return result * 2.0f;
    }

    // arccos(sum (a_i * b_i) / (sqrt(sum a_i^2) * sqrt(sum b_i^2)))
    double cosine_similarity(const object &a, const object &b) {
        const size_t m = min(a.size(), b.size());
        double numerator = 0, sum_a = 0, sum_b = 0;
        for (size_t i = 0; i < m; ++i) {
            numerator += a[i] * b[i];
            sum_a += pow(a[i], 2.0f);
            sum_b += pow(b[i], 2.0f);
        }
        double denominator = sqrt(sum_a) * sqrt(sum_b);
        return denominator < epsilon ? 1.0 : 1 - (numerator / denominator);
    }

    const vector<metric_t> all = { // NOLINT(cert-err58-cpp)
            metric::euclidean,
            metric::chebyshev,
            metric::manhattan,
            metric::hamming,
            // metric::lorentzian,
            metric::canberra,
            metric::sorensen,
            // metric::divergence,
            metric::cosine_similarity
    };
}

typedef function<double(double)> kernel_t;

namespace kernel {

    const double PI = M_PI;

    double uniform(double x) {
        return fabs(x) <= 1 ? 0.5 : 0;
    }

    double triangular(double x) {
        return fabs(x) <= 1 ? 1 - fabs(x) : 0;
    }

    double epanechnikov(double x) {
        return fabs(x) <= 1 ? 0.75f * (1 - pow(x, 2.f)) : 0;
    }

    double biweight(double x) {
        return 15.0 / 16.0 * pow(1 - pow(x, 2.f), 2.f);
    }

    double triweight(double x) {
        return 35.0 / 32.0 * pow(1 - pow(x, 2.f), 3.f);
    }

    double tricube(double x) {
        return fabs(x) <= 1 ? pow(1 - pow(fabs(x), 3.f), 3.f) : 0;
    }

    double gaussian(double x) {
        return 1.0 / sqrt(2 * PI) * exp(-0.5 * pow(x, 2.f));
    }

    double cosine(double x) {
        return fabs(x) <= 1 ? (PI / 4) * cos(PI / 2 * x) : 0;
    }

    double logistic(double x) {
        return 1 / (exp(x) + 2 + exp(-x));
    }

    double sigmoid(double x) {
        return 2 / PI / (exp(x) + exp(-x));
    }

    const vector<kernel_t> all = { // NOLINT(cert-err58-cpp)
            kernel::uniform,
            kernel::triangular,
            kernel::epanechnikov,
            kernel::biweight,
            kernel::triweight,
            kernel::tricube,
            kernel::gaussian,
            kernel::cosine,
            kernel::logistic,
            kernel::sigmoid
    };
}

typedef function<void(size_t features_size, vector<object> &, vector<object> &)> normalization_t;

namespace normalization {

    void minimax(size_t features_size, vector<object> &train_set, vector<object> &test_set) {
        vector<feature_t> max_features(features_size, -numeric_limits<feature_t>::max());
        vector<feature_t> min_features(features_size, numeric_limits<feature_t>::max());

        for (const object &object : train_set) {
            for (int feature_id = 0; feature_id < features_size; ++feature_id) {
                max_features[feature_id] = max(max_features[feature_id], object[feature_id]);
                min_features[feature_id] = min(min_features[feature_id], object[feature_id]);
            }
        }
        for (const object &object : test_set) {
            for (int feature_id = 0; feature_id < features_size; ++feature_id) {
                max_features[feature_id] = max(max_features[feature_id], object[feature_id]);
                min_features[feature_id] = min(min_features[feature_id], object[feature_id]);
            }
        }

        for (object &object : train_set) {
            for (int feature_id = 0; feature_id < features_size; ++feature_id) {
                object[feature_id] -= min_features[feature_id];
                object[feature_id] /= max_features[feature_id] - min_features[feature_id];
            }
        }

        for (object &object : test_set) {
            for (int feature_id = 0; feature_id < features_size; ++feature_id) {
                object[feature_id] -= min_features[feature_id];
                object[feature_id] /= max_features[feature_id] - min_features[feature_id];
            }
        }
    }

    void z_mean(size_t features_size, vector<object> &train_set, vector<object> &test_set) {
        const auto train_size = train_set.size();
        const auto test_size = test_set.size();

        vector<double> mean_features(features_size, 0);
        for (const object &object : train_set) {
            for (int feature_id = 0; feature_id < features_size; ++feature_id) {
                mean_features[feature_id] += object[feature_id];
            }
        }
        for (const object &object : test_set) {
            for (int feature_id = 0; feature_id < features_size; ++feature_id) {
                mean_features[feature_id] += object[feature_id];
            }
        }
        for (double &mean : mean_features) {
            mean /= train_size + test_size;
        }

        vector<double> deviation_features(features_size, 0);
        for (const object &object : train_set) {
            for (int feature_id = 0; feature_id < features_size; ++feature_id) {
                deviation_features[feature_id] +=
                        pow(object[feature_id] - mean_features[feature_id], 2) / (train_size + test_size - 1);
            }
        }
        for (const object &object : test_set) {
            for (int feature_id = 0; feature_id < features_size; ++feature_id) {
                deviation_features[feature_id] +=
                        pow(object[feature_id] - mean_features[feature_id], 2) / (train_size + test_size - 1);
            }
        }
        for (double &deviation : deviation_features) {
            deviation = sqrt(deviation);
        }
        for (object &object : train_set) {
            for (int feature_id = 0; feature_id < features_size; ++feature_id) {
                object[feature_id] -= mean_features[feature_id];
                object[feature_id] /= deviation_features[feature_id];
            }
        }
        for (object &object : test_set) {
            for (int feature_id = 0; feature_id < features_size; ++feature_id) {
                object[feature_id] -= mean_features[feature_id];
                object[feature_id] /= deviation_features[feature_id];
            }
        }
    }
}

namespace score {

    double f1_micro(const vector<vector<unsigned>> &confusion_matrix) {
        const size_t k = confusion_matrix.size();

        vector<int> row_sum(k, 0), column_sum(k, 0), diag(k, 0);
        int total_sum = 0;

        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                unsigned value = confusion_matrix[i][j];
                row_sum[i] += value;
                column_sum[j] += value;
                total_sum += value;
                if (i == j) diag[i] = value;
            }
        }

        double result = 0;
        for (int i = 0; i < k; ++i) {
            double precision = 0, recall = 0, weight = (double) row_sum[i] / total_sum;
            if (column_sum[i] > 0) precision = (double) diag[i] / column_sum[i] * weight;
            if (row_sum[i] > 0) recall = (double) diag[i] / row_sum[i] * weight;
            result += precision + recall < 1e-8 ? 0 : 2 * precision * recall / (precision + recall);
        }

        return result;
    }
}

class knn_classifier {
public:
    static unique_ptr<knn_classifier>
    make_classifier(const vector<object> &train_set, size_t features_size, size_t classes_size) {
        const size_t objects_size = train_set.size();

        vector<metric_t> metrics(metric::all);
        vector<kernel_t> kernels(kernel::all);
        vector<size_t> neighbors = {3, 5, 7, 9, 11};

        auto random_generator = mt19937(random_device()());
        shuffle(metrics.begin(), metrics.end(), random_generator);
        shuffle(kernels.begin(), kernels.end(), random_generator);
        shuffle(neighbors.begin(), neighbors.end(), random_generator);

        vector<vector<unsigned>> confusion_matrix(classes_size);

        double best_score = -numeric_limits<double>::max();
        unique_ptr<knn_classifier> best_classifier;

        // Optimize hyperparameters
        for (const auto &metric : metrics) {
            for (const auto &kernel : kernels) {
                for (const auto &neighbor : neighbors) {
                    if (neighbor >= objects_size) continue;

                    // Clear confusion matrix
                    for (auto &confusion_matrix_row : confusion_matrix) {
                        confusion_matrix_row.assign(classes_size, 0);
                    }

                    // Fill confusion matrix using LOO
                    auto classifier = make_unique<knn_classifier>(train_set, metric, kernel, neighbor);
                    for (unsigned i = 0; i < objects_size; ++i) {
                        class_t real_class = train_set[i].get_class();
                        class_t predicted_class = classifier->get_class(train_set[i], i);
                        confusion_matrix[real_class][predicted_class]++;
                    }

                    // Check score
                    double score = score::f1_micro(confusion_matrix);
                    if (score > best_score) {
                        // cerr << "New score: " << score << endl;
                        best_score = score;
                        best_classifier = move(classifier);
                    }
                }
            }
        }

        return best_classifier;
    }

    class_t get_class(const object &object) {
        return get_class(object, objects.size());
    }

    string info(const object &object) {
        const size_t n = objects.size();

        stringstream result;
        result << fixed << setprecision(8);
        vector<pair<double, size_t>> neighbor_data;
        for (size_t i = 0; i < n; ++i) {
            neighbor_data.emplace_back(metric(object, objects[i]), objects[i].get_index());
        }
        sort(neighbor_data.begin(), neighbor_data.end());

        result << neighbor << " ";
        for (size_t i = 0; i < neighbor; ++i) {
            result << neighbor_data[i].second + 1 << " ";
            result << kernel(neighbor_data[i].first / neighbor_data[neighbor].first)
                   << " ";
        }
        return result.str();
    }

    knn_classifier(vector<object> samples, metric_t metric, kernel_t kernel, size_t neighbor) :
            objects(move(samples)), metric(move(metric)), kernel(move(kernel)), neighbor(neighbor) {}

private:
    const vector<object> objects;
    const metric_t metric;
    const kernel_t kernel;
    const size_t neighbor;

    class_t get_class(const object &object, size_t leave_id) {
        vector<double> weights = get_weights(object, leave_id);
        auto max_score = max_element(weights.begin(), weights.end());
        return static_cast<class_t>(distance(weights.begin(), max_score));
    }

    double get_margin(const object &object, size_t leave_id) {
        vector<double> weights = get_weights(object, leave_id);
        nth_element(weights.begin(), weights.begin() + 1, weights.end(), greater<>());
        return weights[0] - weights[1];
    }

    vector<double> get_weights(const object &object, size_t leave_id) {
        const size_t objects_size = objects.size();

        vector<pair<double, class_t>> neighbor_data;
        for (size_t i = 0; i < objects_size; ++i) {
            if (i == leave_id) continue;
            double distance = metric(object, objects[i]);
            class_t class_id = objects[i].get_class();
            neighbor_data.emplace_back(distance, class_id);
        }

        sort(neighbor_data.begin(), neighbor_data.end());

        vector<double> weights;
        for (unsigned i = 0; i < neighbor; ++i) {
            class_t class_id = neighbor_data[i].second;
            while (weights.size() <= class_id) weights.emplace_back(0.0f);
            weights[class_id] += kernel(neighbor_data[i].first / neighbor_data[neighbor].first);
        }
        return weights;
    }
};

void solve() {
    /**
     * m - number of features
     * k - number of classes
     * n - number of objects in training sample
     */
    size_t features_size, classes_size, train_size;
    cin >> features_size >> classes_size >> train_size;

    vector<object> train_set;

    // Read training set
    for (int object_id = 0; object_id < train_size; ++object_id) {
        vector<feature_t> features(features_size);
        for (feature_t &feature : features) {
            cin >> feature;
        }
        class_t class_id;
        cin >> class_id;
        train_set.emplace_back(object_id, features, class_id - 1);
    }

    /**
     * q - number of objects in test sample
     */
    size_t test_size;
    cin >> test_size;

    vector<object> test_set;

    // Read test set
    for (int object_id = 0; object_id < test_size; ++object_id) {
        vector<feature_t> features(features_size);
        for (feature_t &feature : features) {
            cin >> feature;
        }
        test_set.emplace_back(object_id, features, 0);
    }

    // Apply Z-mean normalization
    normalization::z_mean(features_size, train_set, test_set);

    // Run classifier
    auto classifier = knn_classifier::make_classifier(train_set, features_size, classes_size);
    for (const object &object : test_set) {
        cout << classifier->info(object) << endl;
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
