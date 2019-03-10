#include <utility>

#include <random>
#include <iostream>
#include <fstream>
#include <functional>
#include <sstream>
#include <iomanip>
#include <memory>

using namespace std;

auto random_generator = mt19937(random_device()()); // NOLINT(cert-err58-cpp)

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
};

typedef function<double(const object &, const object &)> metric_t;

namespace metric {

    const double epsilon = 1e-15;

    // sqrt(sum (a_i - b_i)^2)
    double euclidean(const object &a, const object &b) {
        const size_t m = min(a.size(), b.size());
        double result = 0;
        for (size_t i = 0; i < m; ++i) {
            result += pow(a[i] - b[i], 2.0f);
        }
        return sqrt(result);
    }

    // sqrt(sum (a_i - b_i)^2 / m)
    double normalized_euclidean(const object &a, const object &b) {
        const size_t m = min(a.size(), b.size());
        return euclidean(a, b) / sqrt(m);
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

    // sum |a_i - b_i|/(|a_i| + |b_i|)
    double canberra(const object &a, const object &b) {
        const size_t m = min(a.size(), b.size());
        double result = 0;
        for (size_t i = 0; i < m; ++i) {
            result += fabs(a[i] - b[i]) / (fabs(a[i]) + fabs(b[i]));
        }
        return result;
    }

    // 1/2 * sum (a_i - b_i)^2 / (a_i + b_i)
    double chi_square(const object &a, const object &b) {
        const size_t m = min(a.size(), b.size());
        double result = 0;
        for (size_t i = 0; i < m; ++i) {
            double denominator = a[i] + b[i];
            if (denominator < epsilon) return INFINITY;
            result += pow(a[i] - b[i], 2.0f) / denominator;
        }
        return result / 2.0f;
    }

    // sum |a_i - b_i| / sum (a_i + b_i)
    double lance_willams(const object &a, const object &b) {
        const size_t m = min(a.size(), b.size());
        double numerator = 0, denominator = 0;
        for (size_t i = 0; i < m; ++i) {
            numerator += fabs(a[i] - b[i]);
            denominator += a[i] + b[i];
        }
        return numerator / denominator;
    }

    // 1 - sum (a_i * b_i) / sqrt(sum a_i^2 * sum b_i^2)
    double cosine_similarity(const object &a, const object &b) {
        const size_t m = min(a.size(), b.size());
        double numerator = 0, sum_a = 0, sum_b = 0;
        for (size_t i = 0; i < m; ++i) {
            numerator += a[i] * b[i];
            sum_a += a[i] * a[i];
            sum_b += b[i] * b[i];
        }
        double denominator = sqrt(sum_a * sum_b);
        if (denominator < epsilon) return 0.0;
        return 1 - numerator / denominator;
    }
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
    static knn_classifier make_classifier(size_t class_size, const vector<object> &samples) {
        const size_t n = samples.size();
        const size_t k = class_size;

        vector<unsigned> neighbors;
        for (unsigned i = 1; neighbors.size() < 4; i += 2) {
            neighbors.emplace_back(i);
        }
        shuffle(neighbors.begin(), neighbors.end(), random_generator);

        vector<vector<unsigned>> confusion_matrix(k);

        double best_score = 0;
        double best_margin = 0;
        unique_ptr<knn_classifier> best_classifier;

        metric_t best_metric = nullptr;
        kernel_t best_kernel = nullptr;
        size_t best_neighbor = 0;

        // Optimize hyperparameters
        for (const auto &metric : metrics) {
            for (const auto &kernel : kernels) {
                for (const auto &neighbor : neighbors) {
                    if (neighbor > n) continue;

                    // Clear confusion matrix
                    for (auto &confusion_matrix_row : confusion_matrix) {
                        confusion_matrix_row.assign(k, 0);
                    }
                    // Fill confusion matrix using LOO
                    auto classifier = make_unique<knn_classifier>(samples, metric, kernel, neighbor);
                    for (unsigned i = 0; i < n; ++i) {
                        class_t real_class = samples[i].get_class();
                        class_t predicted_class = classifier->get_class(samples[i], i);
                        confusion_matrix[real_class][predicted_class]++;
                    }
                    // Check score
                    double score = score::f1_micro(confusion_matrix);
                    if (score > best_score) {
                        best_score = score;
                        best_metric = metric;
                        best_kernel = kernel;
                        best_neighbor = neighbor;
                        best_classifier = move(classifier);
                    }
                }
            }
        }

        // Prototype selection
        vector<object> best_samples;
        for (unsigned i = 0; i < n; ++i) {
            if (best_classifier->get_margin(samples[i], i) > 0) {
                best_samples.push_back(samples[i]);
            }
        }

        if (best_neighbor >= best_samples.size()) {
            best_neighbor = best_samples.size() - 1;
        }

        return knn_classifier(best_samples, best_metric, best_kernel, best_neighbor);
    }

    knn_classifier(vector<object> samples, metric_t metric, kernel_t kernel, size_t neighbor) :
            samples(move(samples)), metric(move(metric)), kernel(move(kernel)), neighbor(neighbor) {}


    class_t get_class(const object &object) {
        return get_class(object, samples.size());
    }

    string info(const object &object) {
        const size_t n = samples.size();

        stringstream result;
        vector<pair<double, size_t>> neighbor_data;
        for (size_t i = 0; i < n; ++i) {
            neighbor_data.emplace_back(metric(object, samples[i]), samples[i].get_index());
        }
        sort(neighbor_data.begin(), neighbor_data.end(),
             [](const pair<double, size_t> &a, const pair<double, size_t> &b) {
                 return a.first < b.first;
             });

        result << neighbor << " ";
        for (size_t i = 0; i < neighbor; ++i) {
            result << neighbor_data[i].second + 1 << " ";
            result << fixed << setprecision(8) << kernel(neighbor_data[i].first / neighbor_data[neighbor].first)
                   << " ";
        }
        return result.str();
    }

    static void initialize() {
        shuffle(metrics.begin(), metrics.end(), random_generator);
        shuffle(kernels.begin(), kernels.end(), random_generator);
    }

private:
    static vector<metric_t> metrics;
    static vector<kernel_t> kernels;

    const vector<object> samples;
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
        const size_t n = samples.size();

        vector<pair<double, class_t>> neighbor_data;
        for (size_t i = 0; i < n; ++i) {
            if (i == leave_id) continue;
            double distance = metric(object, samples[i]);
            class_t class_id = samples[i].get_class();
            neighbor_data.emplace_back(distance, class_id);
        }

        sort(neighbor_data.begin(), neighbor_data.end(),
             [](const pair<double, class_t> &a, const pair<double, class_t> &b) {
                 return a.first < b.first;
             });

        vector<double> weights;
        for (unsigned i = 0; i < neighbor; ++i) {
            class_t class_id = neighbor_data[i].second;
            while (weights.size() <= class_id) weights.emplace_back(0.0f);
            weights[class_id] += kernel(neighbor_data[i].first / neighbor_data[neighbor].first);
        }
        return weights;
    }
};

vector<metric_t> knn_classifier::metrics = { // NOLINT(cert-err58-cpp)
        metric::euclidean,
        metric::normalized_euclidean,
        metric::chebyshev,
        metric::manhattan,
        metric::canberra,
        metric::chi_square,
        metric::lance_willams,
        metric::cosine_similarity
};

vector<kernel_t> knn_classifier::kernels = { // NOLINT(cert-err58-cpp)
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

void solve() {
    /**
     * m - number of features
     * k - number of classes
     * n - number of objects in training sample
     */
    size_t m, k, n;
    cin >> m >> k >> n;

    vector<feature_t> max_features(m, numeric_limits<feature_t>::min());
    vector<feature_t> min_features(m, numeric_limits<feature_t>::max());

    vector<object> samples;

    // Read training samples
    for (int object_id = 0; object_id < n; ++object_id) {
        vector<feature_t> features(m);
        for (int feature_id = 0; feature_id < m; ++feature_id) {
            cin >> features[feature_id];
            max_features[feature_id] = max(max_features[feature_id], features[feature_id]);
            min_features[feature_id] = min(min_features[feature_id], features[feature_id]);
        }
        class_t class_id;
        cin >> class_id;
        samples.emplace_back(object_id, features, class_id - 1);
    }

    /**
     * q - number of objects in test sample
     */
    size_t q;
    cin >> q;

    vector<object> tests;

    // Read tests
    for (int object_id = 0; object_id < q; ++object_id) {
        vector<feature_t> features(m);
        for (int feature_id = 0; feature_id < m; ++feature_id) {
            cin >> features[feature_id];
            max_features[feature_id] = max(max_features[feature_id], features[feature_id]);
            min_features[feature_id] = min(min_features[feature_id], features[feature_id]);
        }
        tests.emplace_back(object_id, features, 0);
    }

    // Normalize features (minimax)
    for (object &sample : samples) {
        for (int feature_id = 0; feature_id < m; ++feature_id) {
            sample[feature_id] -= min_features[feature_id];
            sample[feature_id] /= max_features[feature_id] - min_features[feature_id];
        }
    }

    for (object &test : tests) {
        for (int feature_id = 0; feature_id < m; ++feature_id) {
            test[feature_id] -= min_features[feature_id];
            test[feature_id] /= max_features[feature_id] - min_features[feature_id];
        }
    }

    knn_classifier::initialize();
    auto classifier = knn_classifier::make_classifier(k, samples);
    for (const object &test : tests) {
        cout << classifier.info(test) << endl;
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
