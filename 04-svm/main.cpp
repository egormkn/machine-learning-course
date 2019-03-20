#include <utility>

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <memory>
#include <functional>
#include <cassert>
#include <sstream>
#include <iomanip>
#include <set>

using namespace std;

auto random_generator = mt19937(random_device()()); // NOLINT

typedef double feature_t;

typedef int class_t;

class object {
private:
    vector<feature_t> features;
    class_t class_id;
public:
    object(size_t index, vector<feature_t> features, class_t class_id) :
            features(move(features)), class_id(class_id) {}

    size_t size() const { return features.size(); }

    feature_t &operator[](size_t index) { return features[index]; }

    feature_t operator[](size_t index) const { return features[index]; }

    class_t get_class() const { return class_id; }

    vector<feature_t>::const_iterator begin() const { return features.begin(); }

    vector<feature_t>::const_iterator end() const { return features.end(); }
};

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

namespace kernels {

    class kernel {
    public:
        virtual double apply(const object &a, const object &b) const = 0;

        virtual string str(const vector<string> &a, const vector<string> &b) const = 0;

        string str(size_t dimension) const {
            vector<string> a(dimension), b(dimension);
            for (unsigned i = 0; i < dimension; ++i) {
                a[i] = "A" + to_string(i);
                b[i] = "B" + to_string(i);
            }
            return str(a, b);
        }
    };

    typedef shared_ptr<kernel> kernel_ptr;

    /**
     * K(a, b) = c
     */
    class constant : public kernel {
    public:
        explicit constant(double value) : value(value) {}

        double apply(const object &a, const object &b) const override {
            return value;
        }

        string str(const vector<string> &a, const vector<string> &b) const override {
            return to_string(value);
        }

    private:
        const double value;
    };

    /**
     * K(a, b) = <a, b>
     */
    class scalar_product : public kernel {
    public:
        double apply(const object &a, const object &b) const override {
            return inner_product(a.begin(), a.end(), b.begin(), 0.0);
        }

        string str(const vector<string> &a, const vector<string> &b) const override {
            stringstream out;
            out << "sum(";
            for (unsigned i = 0; i < min(a.size(), b.size()); ++i) {
                if (i > 0) out << ',';
                out << "prod(" << a[i] << "," << b[i] << ")";
            }
            out << ")";
            return out.str();
        }
    };

    /**
     * K(a, b) = c_1 * K_1(a, b) + c_2 * K_2(a, b)
     */
    class kernel_sum : public kernel {
    public:
        kernel_sum(double c1, kernel_ptr k1, double c2, kernel_ptr k2) :
                c1(c1),
                k1(move(k1)),
                c2(c2),
                k2(move(k2)) {
            assert(c1 > 0);
            assert(c2 > 0);
        }

        double apply(const object &a, const object &b) const override {
            return c1 * k1->apply(a, b) + c2 * k2->apply(a, b);
        }

        string str(const vector<string> &a, const vector<string> &b) const override {
            stringstream out;
            out << fixed;
            out << "sum(prod(" << c1 << "," << k1->str(a, b) << "),prod(" << c2 << "," << k2->str(a, b) << "))";
            return out.str();
        }

    private:
        const double c1, c2;
        const kernel_ptr k1, k2;
    };

    /**
     * K(a, b) = K_1(a, b) * K_2(a, b)
     */
    class kernel_product : public kernel {
    public:
        kernel_product(kernel_ptr k1, kernel_ptr k2) : k1(move(k1)), k2(move(k2)) {}

        double apply(const object &a, const object &b) const override {
            return k1->apply(a, b) * k2->apply(a, b);
        }

        string str(const vector<string> &a, const vector<string> &b) const override {
            stringstream out;
            out << "prod(" << k1->str(a, b) << "," << k2->str(a, b) << ")";
            return out.str();
        }

    private:
        const kernel_ptr k1, k2;
    };

    /**
     * K(a, b) = f(a) * f(b)
     */
    class real_function : public kernel {
    public:
        real_function(function<double(const object &)> f, function<string(const vector<string> &)> f_string) : f(
                move(f)), f_string(move(f_string)) {}

        double apply(const object &a, const object &b) const override {
            return f(a) * f(b);
        }

        string str(const vector<string> &a, const vector<string> &b) const override {
            stringstream out;
            out << "prod(" << f_string(a) << "," << f_string(b) << ")";
            return out.str();
        }

    private:
        const function<double(const object &)> f;
        const function<string(const vector<string> &)> f_string;
    };

    /**
     * K(a, b) = K_0(f(a), f(b))
     */
    class vector_function : public kernel {
    public:
        vector_function(kernel_ptr k0, function<object(const object &)> f,
                        function<vector<string>(const vector<string> &)> f_string) : k0(move(k0)), f(move(f)),
                                                                                     f_string(move(f_string)) {}

        double apply(const object &a, const object &b) const override {
            return k0->apply(f(a), f(b));
        }

        string str(const vector<string> &a, const vector<string> &b) const override {
            return k0->str(f_string(a), f_string(b));
        }

    private:
        const kernel_ptr k0;
        const function<object(const object &)> f;
        const function<vector<string>(const vector<string> &)> f_string;
    };

    /**
     * K(a, b) = f(K_0(a, b))
     */
    class kernel_function : public kernel {
    public:
        kernel_function(kernel_ptr k0, function<double(double)> f, function<string(const string &)> f_string) : k0(
                move(k0)), f(move(f)), f_string(move(f_string)) {}

        double apply(const object &a, const object &b) const override {
            return f(k0->apply(a, b));
        }

        string str(const vector<string> &a, const vector<string> &b) const override {
            return f_string(k0->str(a, b));
        }

    private:
        const kernel_ptr k0;
        const function<double(double)> f;
        const function<string(const string &)> f_string;
    };

    /**
     * K(a, b) = ||a - b||^2
     */
    class squared_euclidean_distance : public kernel { // FIXME: Not sure if it's actually a kernel
    public:
        double apply(const object &a, const object &b) const override {
            double result = 0.0;
            for (unsigned i = 0; i < min(a.size(), b.size()); ++i) {
                result += pow(a[i] - b[i], 2.0);
            }
            return result;
        }

        string str(const vector<string> &a, const vector<string> &b) const override {
            stringstream out;
            out << "sum(";
            for (unsigned i = 0; i < min(a.size(), b.size()); ++i) {
                if (i > 0) out << ",";
                out << "pow(sub(" << a[i] << "," << b[i] << "),2.0)";
            }
            out << ")";
            return out.str();
        }
    };

    kernel_ptr polynomial(double c, double d) {
        kernel_ptr k;
        if (fabs(c) < 1e-10) {
            k = make_shared<scalar_product>();
        } else {
            k = make_shared<kernel_sum>(1.0, make_shared<scalar_product>(), 1.0, make_shared<constant>(c));
        }
        if (fabs(d - 1.0) < 1e-10) {
            return k;
        }
        auto f = [d](double value) {
            return pow(value, d);
        };
        auto f_string = [d](const string &str) {
            stringstream out;
            out << fixed;
            out << "pow(" << str << "," << d << ")";
            return out.str();
        };
        return make_shared<kernel_function>(k, f, f_string);
    }

    kernel_ptr rbf(double gamma) {
        auto k = make_shared<squared_euclidean_distance>();
        auto f = [gamma](double value) {
            return exp(-gamma * value);
        };
        auto f_string = [gamma](const string &str) {
            stringstream out;
            out << fixed;
            out << "pow(" << M_E << ",prod(" << -gamma << "," << str << "))";
            return out.str();
        };
        return make_shared<kernel_function>(k, f, f_string);
    }
}

using namespace kernels;

class svm {
public:
    static unique_ptr<svm> make_svm(const vector<object> &train_set, size_t features_size) {
        const size_t &objects_size = train_set.size();

        vector<kernel_ptr> kernels = {
                polynomial(0, 1),
                polynomial(0, 2),
                polynomial(0, 3),
                polynomial(0, 4),
                polynomial(1, 1),
                polynomial(1, 2),
                polynomial(1, 3),
                polynomial(1, 4),
                rbf(1),
                rbf(2),
                rbf(3)
        };
        vector<double> c_values = {0.01, 0.1, 1, 10, 100, 1000, 10000, 100000};

        shuffle(kernels.begin(), kernels.end(), random_generator);
        shuffle(c_values.begin(), c_values.end(), random_generator);

        vector<vector<unsigned>> confusion_matrix(2, vector<unsigned>(2));

        double best_score = -numeric_limits<double>::max();
        unique_ptr<svm> best_classifier;

        vector<vector<double>> kernel_cache(objects_size, vector<double>(objects_size));

        // Select kernel
        for (const kernel_ptr &kernel : kernels) {

            // Fill kernel matrix
            for (unsigned i = 0; i < objects_size; ++i) {
                for (unsigned j = i; j < objects_size; ++j) {
                    kernel_cache[i][j] = kernel_cache[j][i] = kernel->apply(train_set[i], train_set[j]);
                }
            }

            // Select c
            for (double c : c_values) {

                // Clear confusion matrix
                for (auto &confusion_matrix_row : confusion_matrix) {
                    confusion_matrix_row.assign(2, 0);
                }

                // Find lambdas
                smo_solver solver(train_set, features_size, kernel, kernel_cache, c);
                const vector<double> &lambdas = solver.get_lambdas();
                const double &b = solver.get_b();

                // Fill confusion matrix using LOO
                auto classifier = make_unique<svm>(train_set, features_size, kernel, lambdas, b);
                for (unsigned object_id = 0; object_id < objects_size; ++object_id) {
                    class_t real_class = train_set[object_id].get_class();
                    class_t predicted_class = classifier->get_class(train_set[object_id], object_id, kernel_cache);
                    confusion_matrix[(real_class + 1) / 2][(predicted_class + 1) / 2]++;
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

        return best_classifier;
    }

    class_t get_class(const object &x) const {
        const auto &objects_size = objects.size();

        double result = -b;
        for (unsigned object_id = 0; object_id < objects_size; ++object_id) {
            if (fabs(lambdas[object_id]) < epsilon) continue;
            const object &object = objects[object_id];
            result += lambdas[object_id] * object.get_class() * kernel->apply(object, x);
        }
        return result > 0 ? 1 : -1;
    }

    kernel_ptr get_kernel() const {
        return kernel;
    }

    vector<double> get_lambdas() const {
        return lambdas;
    }

    double get_b() const {
        return b;
    }

    svm(vector<object> objects, size_t features_size, kernel_ptr kernel, vector<double> lambdas, double b)
            : objects(move(objects)), features_size(features_size), kernel(move(kernel)), lambdas(move(lambdas)),
              b(b) {}

private:
    static constexpr double epsilon = 1e-3;

    const vector<object> objects;
    const size_t features_size;
    const kernel_ptr kernel;
    const vector<double> lambdas;
    const double b;

    class_t get_class(const object &x, size_t x_id) const {
        const auto &objects_size = objects.size();

        double result = -b;
        for (unsigned object_id = 0; object_id < objects_size; ++object_id) {
            if (object_id == x_id) continue;
            const object &object = objects[object_id];
            result += lambdas[object_id] * object.get_class() * kernel->apply(x, object);
        }
        return result > 0 ? 1 : -1;
    }

    class_t get_class(const object &x, size_t x_id, const vector<vector<double>> &kernel_cache) const {
        const auto &objects_size = objects.size();

        double result = -b;
        for (unsigned object_id = 0; object_id < objects_size; ++object_id) {
            if (object_id == x_id) continue;
            const object &object = objects[object_id];
            result += lambdas[object_id] * object.get_class() * kernel_cache[x_id][object_id];
        }
        return result > 0 ? 1 : -1;
    }

    // Sequential Minimal Optimization
    class smo_solver {
    public:
        smo_solver(const vector<object> &objects, size_t features_size, const kernel_ptr &kernel,
                   const vector<vector<double>> &kernel_cache, double c)
                : objects(objects), features_size(features_size), kernel(kernel), kernel_cache(kernel_cache), c(c) {
            lambdas.resize(objects.size(), 0.0);
            solve();
        }

        vector<double> get_lambdas() {
            return lambdas;
        }

        double get_b() {
            return b;
        }

    protected:
        static constexpr double tolerance = 1e-3;
        static constexpr double eps = 1e-3;

        const vector<object> &objects;
        const size_t features_size;
        const kernel_ptr &kernel;
        const vector<vector<double>> &kernel_cache;
        const double c;

        vector<double> lambdas;
        double b = 0.0;

        vector<double> error_cache;
        set<unsigned> non_bound;

        double f(const object &x) {
            double result = -b;
            for (unsigned object_id = 0; object_id < objects.size(); ++object_id) {
                const object &object = objects[object_id];
                result += lambdas[object_id] * object.get_class() * kernel->apply(x, object);
            }
            return result;
        };

        double f(unsigned i) {
            double result = -b;
            for (unsigned object_id = 0; object_id < objects.size(); ++object_id) {
                result += lambdas[object_id] * objects[object_id].get_class() * kernel_cache[i][object_id];
            }
            return result;
        };

        inline bool between_bounds(double alpha) {
            return fabs(alpha) > epsilon && fabs(alpha) < c - epsilon;
        }

        inline bool on_bounds(double alpha) {
            return fabs(alpha) < epsilon || fabs(alpha - c) < epsilon;
        }

        bool take_step(unsigned i1, unsigned i2, double E2) {
            if (i1 == i2) return false;

            const object &x1 = objects[i1], &x2 = objects[i2];
            class_t y1 = x1.get_class(), y2 = x2.get_class();
            double alpha1 = lambdas[i1], alpha2 = lambdas[i2];

            double E1 = between_bounds(alpha1) ? error_cache[i1] : (f(i1) - y1); // FIXME: Check later
            double s = y1 * y2;

            double L = y1 == y2 ? max(0.0, alpha1 + alpha2 - c) : max(0.0, alpha2 - alpha1);
            double H = y1 == y2 ? min(c, alpha1 + alpha2) : min(c, c + alpha2 - alpha1);

            if (fabs(L - H) < epsilon) return false;

            double k11 = kernel_cache[i1][i1];
            double k12 = kernel_cache[i1][i2];
            double k22 = kernel_cache[i2][i2];

            double eta = k11 + k22 - 2 * k12;

            double a1, a2;

            if (eta > 0) {
                a2 = alpha2 + y2 * (E1 - E2) / eta;
                if (a2 < L) {
                    a2 = L;
                } else if (a2 > H) {
                    a2 = H;
                }
            } else {
                // FIXME: check or replace
                double f1 = y1 * (E1 + b) - alpha1 * k11 - s * alpha2 * k12;
                double f2 = y2 * (E2 + b) - s * alpha1 * k12 - alpha2 * k22;
                double L1 = alpha1 + s * (alpha2 - L);
                double H1 = alpha1 + s * (alpha2 - H);
                double objL = L1 * f1 + L * f2 + 0.5 * L1 * L1 * k11 + 0.5 * L * L * k22 + s * L * L1 * k12;
                double objH = H1 * f1 + H * f2 + 0.5 * H1 * H1 * k11 + 0.5 * H * H * k22 + s * H * H1 * k12;
                if (objL < objH - eps) {
                    a2 = L;
                } else if (objL > objH + eps) {
                    a2 = H;
                } else {
                    a2 = alpha2;
                }
            }

            if (fabs(a2 - alpha2) < eps * (a2 + alpha2 + eps)) return false;

            a1 = alpha1 + s * (alpha2 - a2);

            // FIXME: Below
            /*if (a1 < 0) {
                a2 += s * a1;
                a1 = 0;
            } else if (a1 > c) {
                a2 += s * (a1 - c);
                a1 = c;
            }*/
            // FIXME: Above


            double b1 = b + E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12;
            double b2 = b + E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22;

            double b_new;
            if (0 < a1 && a1 < c) {
                b_new = b1;
            } else if (0 < a2 && a2 < c) {
                b_new = b2;
            } else {
                b_new = (b1 + b2) / 2.0;
            }

            double deltaB = b_new - b;
            b = b_new;

            double deltaA1 = a1 - alpha1;
            double deltaA2 = a2 - alpha2;

            // Update error cache
            for (unsigned i : non_bound) {
                error_cache[i] += deltaA1 * y1 * kernel_cache[i1][i] +
                                  deltaA2 * y2 * kernel_cache[i2][i] - deltaB;
            }
            error_cache[i1] = 0.0; // FIXME: Check
            error_cache[i2] = 0.0;

            // Update set of multipliers on bounds
            if (!on_bounds(a1)) {
                non_bound.insert(i1);
            } else {
                non_bound.erase(i1);
            }

            if (!on_bounds(a2)) {
                non_bound.insert(i2);
            } else {
                non_bound.erase(i2);
            }

            // Store results in the array
            lambdas[i1] = a1;
            lambdas[i2] = a2;

            return true;
        }

        bool examine_example(unsigned i2) {
            const object &x2 = objects[i2];
            class_t y2 = x2.get_class();
            double alpha2 = lambdas[i2];

            double E2 = between_bounds(alpha2) ? error_cache[i2] : (f(i2) - y2); // FIXME: Check later
            double r2 = E2 * y2;
            if ((r2 < -tolerance && alpha2 < c - epsilon) || (r2 > tolerance && alpha2 > epsilon)) {
                if (non_bound.size() > 1) {
                    unsigned i1 = 0;
                    for (unsigned i : non_bound) {
                        if (fabs(E2 - error_cache[i]) > fabs(E2 - error_cache[i1])) i1 = i;
                    }
                    if (take_step(i1, i2, E2)) return true;
                }
                vector<unsigned> non_bound_ids(non_bound.begin(), non_bound.end());
                shuffle(non_bound_ids.begin(), non_bound_ids.end(), random_generator);
                for (unsigned i1 : non_bound_ids) {
                    if (take_step(i1, i2, E2)) return true;
                }
                unsigned random_start = random_generator() % objects.size();
                for (unsigned i = random_start; i < random_start + objects.size(); ++i) {
                    auto i1 = static_cast<unsigned>(i % objects.size());
                    if (take_step(i1, i2, E2)) return true;
                }
            }
            return false;
        }

        void solve() {
            const auto &objects_size = objects.size();

            error_cache.resize(objects_size, 0.0);

            int num_changed = 0;
            bool examine_all = true;
            while (num_changed > 0 || examine_all) {
                num_changed = 0;
                if (examine_all) {
                    for (unsigned i = 0; i < objects_size; ++i) {
                        if (examine_example(i)) num_changed++;
                    }
                    examine_all = false;
                } else {
                    for (unsigned i : non_bound) {
                        // if (!on_bounds(lambdas[i]))
                        if (examine_example(i)) num_changed++;
                    }
                    if (num_changed == 0) {
                        examine_all = true;
                    }
                }
            }
        }
    };
};

void solve() {
    /**
     * m - number of features
     * n - number of objects in training set
     */
    size_t features_size, objects_size;
    cin >> features_size >> objects_size;

    vector<object> train_set;

    // Read training set
    for (unsigned object_id = 0; object_id < objects_size; ++object_id) {
        vector<feature_t> features(features_size);
        for (feature_t &feature : features) {
            cin >> feature;
        }
        char c;
        cin >> c;
        train_set.emplace_back(object_id, features, c == '+' ? 1 : -1);
    }

    auto svm = svm::make_svm(train_set, features_size);

    cout << setprecision(8);
    cout << svm->get_kernel()->str(features_size) << endl;
    for (const auto &lambda : svm->get_lambdas()) {
        cout << lambda << endl;
    }
    cout << svm->get_b();
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