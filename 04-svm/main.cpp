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
                rbf(0.125),
                rbf(0.25),
                rbf(0.5),
                rbf(1),
                rbf(2),
                rbf(2.5)
        };
        vector<double> c_values = {0.1, 0.5, 1, 2, 3, 4, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 5000, 10000,
                                   100000};

        shuffle(kernels.begin(), kernels.end(), random_generator);
        shuffle(c_values.begin(), c_values.end(), random_generator);

        vector<vector<unsigned>> confusion_matrix(2, vector<unsigned>(2));

        double best_score = -numeric_limits<double>::max();
        unique_ptr<svm> best_classifier;

        // Select hyperparameters
        for (const kernel_ptr &kernel : kernels) {
            for (double c : c_values) {

                // Clear confusion matrix
                for (auto &confusion_matrix_row : confusion_matrix) {
                    confusion_matrix_row.assign(2, 0);
                }

                // Find lambdas
                svm::solver solver(train_set, features_size, kernel, c);
                const vector<double> &lambdas = solver.get_lambdas();
                const double &b = solver.get_b();

                // Fill confusion matrix using LOO
                auto classifier = make_unique<svm>(train_set, features_size, kernel, lambdas, b);
                for (unsigned object_id = 0; object_id < objects_size; ++object_id) {
                    class_t real_class = train_set[object_id].get_class();
                    class_t predicted_class = classifier->get_class(train_set[object_id], object_id);
                    confusion_matrix[(real_class + 1) / 2][(predicted_class + 1) / 2]++;
                }

                // Check score
                double score = score::f1_micro(confusion_matrix);
                if (score > best_score) {
                    cerr << "New score: " << score << endl;
                    best_score = score;
                    best_classifier = move(classifier);
                }
            }
        }

        return best_classifier;
    }

    class_t get_class(const object &x) const {
        return get_class(x, objects.size());
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

    class_t get_class(const object &x, size_t leave_id) const {
        const auto &objects_size = objects.size();

        double result = -b;
        for (unsigned object_id = 0; object_id < objects_size; ++object_id) {
            if (object_id == leave_id) continue;
            const object &object = objects[object_id];
            result += lambdas[object_id] * object.get_class() * kernel->apply(object, x);
        }
        return result > 0 ? 1 : -1;
    }

    class solver {
    private:
        const vector<object> objects;
        const size_t features_size;
        const kernel_ptr kernel;
        const double c;

        vector<double> lambdas;
        double b = 0.0;

        void solve_smo() {
            const auto &objects_size = objects.size();

            vector<double> lambdas(objects_size, 0.0);
            double b = 0.0;

            set<unsigned> non_bound;
            vector<double> error_cache(objects_size, -b);
            for (unsigned i = 0; i < objects_size; ++i) {
                error_cache[i] -= objects[i].get_class();
                // Lambdas = 0
            }

            const auto &f_old = [this, &lambdas, &b](const object &x) {
                double result = b;
                for (unsigned object_id = 0; object_id < objects.size(); ++object_id) {
                    const object &object = objects[object_id];
                    result += lambdas[object_id] * object.get_class() * kernel->apply(x, object);
                }
                return result;
            };

            const auto &take_step = [this, &lambdas, &error_cache, &non_bound, &b](unsigned i1, unsigned i2,
                                                                                   double E2) {
                if (i1 == i2) return false;
                const auto &x2 = objects[i2];
                const auto &y2 = x2.get_class();
                double a2 = lambdas[i2];
                const auto &x1 = objects[i1];
                const auto &y1 = x1.get_class();
                double a1 = lambdas[i1];
                double E1 = error_cache[i1];
                double s = y1 * y2;

                double a1new, a2new;

                double L = y1 == y2 ? max(0.0, a1 + a2 - c) : max(0.0, a2 - a1);
                double H = y1 == y2 ? min(c, a1 + a2) : min(c, c + a2 - a1);

                if (L == H) return false;

                double k11 = kernel->apply(x1, x1);
                double k12 = kernel->apply(x1, x2);
                double k22 = kernel->apply(x2, x2);

                double eta = k11 + k22 - 2 * k12;
                if (eta > 0) {
                    a2new = a2 + y2 * (E1 - E2) / eta;
                    if (a2new < L) {
                        a2new = L;
                    } else if (a2new > H) {
                        a2new = H;
                    }
                } else {
                    double f1 = y1 * (E1 + b) - a1 * k11 - s * a2 * k12;
                    double f2 = y2 * (E2 + b) - s * a1 * k12 - a2 * k22;
                    double L1 = a1 + s * (a2 - L);
                    double H1 = a1 + s * (a2 - H);
                    double objL = L1 * f1 + L * f2 + 0.5 * L1 * L1 * k11 + 0.5 * L * L * k22 + s * L * L1 * k12;
                    double objH = H1 * f1 + H * f2 + 0.5 * H1 * H1 * k11 + 0.5 * H * H * k22 + s * H * H1 * k12;
                    if (objL < objH - epsilon) {
                        a2new = L;
                    } else if (objL > objH + epsilon) {
                        a2new = H;
                    } else {
                        a2new = a2;
                    }
                }

                if (fabs(a2 - a2new) < epsilon * (a2 + a2new + epsilon)) return false;

                a1new = a1 + s * (a2 - a2new);
                double b1 = b + E1 + y1 * (a1new - a1) * k11 + y2 * (a2new - a2) * k12;
                double b2 = b + E2 + y1 * (a1new - a1) * k12 + y2 * (a2new - a2) * k22;

                double deltaB;
                if (0 < a1new && a1new < c) {
                    deltaB = b1 - b;
                    b = b1;
                } else if (0 < a2new && a2new < c) {
                    deltaB = b2 - b;
                    b = b2;
                } else {
                    deltaB = (b1 + b2) / 2.0 - b;
                    b = (b1 + b2) / 2.0;
                }

                double deltaA1 = a1new - a1;
                double deltaA2 = a2new - a2;

                for (unsigned i = 0; i < objects.size(); ++i) {
                    error_cache[i] += deltaA1 * y1 * kernel->apply(x1, objects[i]) +
                                      deltaA2 * y2 * kernel->apply(x2, objects[i]) - deltaB;
                }

                lambdas[i1] = a1new;
                lambdas[i2] = a2new;

                if (a1new != 0 && a1new != c) {
                    non_bound.insert(i1);
                } else {
                    non_bound.erase(i1);
                }

                if (a2new != 0 && a2new != c) {
                    non_bound.insert(i2);
                } else {
                    non_bound.erase(i2);
                }

                return true;
            };

            const auto &examine_example = [this, &lambdas, &error_cache, &non_bound, &take_step](unsigned i2) {
                const auto &x2 = objects[i2];
                const auto &y2 = x2.get_class();
                double a2 = lambdas[i2];
                double E2 = error_cache[i2];
                double r2 = E2 * y2;
                if ((r2 < -epsilon && a2 < c) || (r2 > epsilon && a2 > 0)) {
                    if (non_bound.size() > 1) {
                        unsigned i1 = 0;
                        if (E2 > 0) {
                            for (unsigned i : non_bound) {
                                if (error_cache[i] < error_cache[i1]) i1 = i;
                            }
                        } else {
                            for (unsigned i : non_bound) {
                                if (error_cache[i] > error_cache[i1]) i1 = i;
                            }
                        }
                        if (take_step(i1, i2, E2)) return true;
                    }
                    vector<unsigned> i1s(non_bound.begin(), non_bound.end());
                    shuffle(i1s.begin(), i1s.end(), random_generator);
                    for (unsigned i1 : i1s) {
                        if (take_step(i1, i2, E2)) return true;
                    }
                    i1s.resize(objects.size());
                    iota(i1s.begin(), i1s.end(), 0);
                    shuffle(i1s.begin(), i1s.end(), random_generator);
                    for (unsigned i1 : i1s) {
                        if (take_step(i1, i2, E2)) return true;
                    }
                }
                return false;
            };

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
                        if (examine_example(i)) num_changed++;
                    }
                    if (num_changed == 0) examine_all = true;
                }
            }

            this->lambdas = lambdas;
            this->b = b;
        }

        void solve_simplified_smo() {
            const auto &objects_size = objects.size();
            const auto &features_size = objects[0].size();

            uniform_int_distribution<> random_range(1, static_cast<int>(objects_size - 1));

            vector<double> lambdas(objects_size, 0.0);
            double b = 0.0;

            unsigned passes = 0;
            while (passes < 100) {
                unsigned num_changed_lambdas = 0;
                for (unsigned i = 0; i < objects_size; ++i) {
                    const object &x_i = objects[i];
                    const class_t &y_i = objects[i].get_class();
                    double &a_i = lambdas[i];
                    const auto &f = [this, &lambdas, &b](const object &x) {
                        double result = b;
                        for (unsigned object_id = 0; object_id < objects.size(); ++object_id) {
                            const object &object = objects[object_id];
                            result += lambdas[object_id] * object.get_class() * kernel->apply(x, object);
                        }
                        return result;
                    };

                    double E_i = f(x_i) - y_i;

                    if ((y_i * E_i < -epsilon && a_i < c) || (y_i * E_i > epsilon && a_i > 0)) {
                        unsigned j = (i + random_range(random_generator)) % objects_size;
                        const object &x_j = objects[j];
                        const class_t &y_j = objects[j].get_class();
                        double &a_j = lambdas[j];

                        double E_j = f(x_j) - y_j;

                        double a_i_old = a_i, a_j_old = a_j;

                        double L = y_i == y_j ? max(0.0, a_i + a_j - c) : max(0.0, a_j - a_i);
                        double H = y_i == y_j ? min(c, a_i + a_j) : min(c, c + a_j - a_i);

                        if (fabs(L - H) < epsilon) continue;

                        double eta = 2 * kernel->apply(x_i, x_j) - kernel->apply(x_i, x_i) - kernel->apply(x_j, x_j);

                        if (eta >= 0) continue;

                        a_j = a_j - y_j * (E_i - E_j) / eta;
                        if (a_j > H) a_j = H;
                        if (a_j < L) a_j = L;

                        if (fabs(a_j - a_j_old) < 1e-5) continue;

                        a_i = a_i + y_i * y_j * (a_j_old - a_j);

                        double b1 = b - E_i - y_i * (a_i - a_i_old) * kernel->apply(x_i, x_i) -
                                    y_j * (a_j - a_j_old) * kernel->apply(x_i, x_j);
                        double b2 = b - E_j - y_i * (a_i - a_i_old) * kernel->apply(x_i, x_j) -
                                    y_j * (a_j - a_j_old) * kernel->apply(x_j, x_j);

                        if (0 < a_i && a_i < c) {
                            b = b1;
                        } else if (0 < a_j && a_j < c) {
                            b = b2;
                        } else {
                            b = (b1 + b2) / 2.0;
                        }

                        num_changed_lambdas++;
                    }
                }
                passes = num_changed_lambdas == 0 ? passes + 1 : 0;
            }


            this->lambdas = lambdas;
            this->b = -b;
        }

    public:
        solver(vector<object> objects, size_t features_size, kernel_ptr kernel, double c) : objects(move(objects)),
                                                                                            features_size(
                                                                                                    features_size),
                                                                                            kernel(move(kernel)), c(c) {
            solve_smo();
        }

        vector<double> get_lambdas() {
            return lambdas;
        }

        double get_b() {
            return b;
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
