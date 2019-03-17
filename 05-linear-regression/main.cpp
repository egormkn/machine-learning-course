#include <utility>

#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <sstream>
#include <iomanip>
#include <cassert>
#include <set>

using namespace std;

auto random_generator = mt19937(random_device()()); // NOLINT

typedef int feature_t;

typedef int class_t;

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

template<typename T>
class matrix {
private:
    vector<vector<T>> data;
    const size_t num_rows, num_cols;
    static constexpr double epsilon = 1e-8;

public:
    matrix(size_t rows, size_t cols, const T &value) : num_rows(rows), num_cols(cols),
                                                       data(rows, vector<T>(cols, value)) {}

    matrix(size_t rows, size_t cols) : num_rows(rows), num_cols(cols), data(rows, vector<T>(cols)) {}

    vector<T> &operator[](size_t index) { return data[index]; }

    const vector<T> &operator[](size_t index) const { return data[index]; }

    size_t rows() const {
        return num_rows;
    }

    size_t cols() const {
        return num_cols;
    }

    matrix<T> operator+(const matrix<T> &rhs) const {
        assert(num_rows == rhs.num_rows);
        assert(num_cols == rhs.num_cols);
        matrix<T> result(num_rows, num_cols);
        for (unsigned i = 0; i < result.num_rows; ++i)
            for (unsigned j = 0; j < result.num_cols; ++j)
                result.data[i][j] = data[i][j] + rhs.data[i][j];
        return result;
    }

    matrix<T> operator-(const matrix<T> &rhs) const {
        assert(num_rows == rhs.num_rows);
        assert(num_cols == rhs.num_cols);
        matrix<T> result(num_rows, num_cols);
        for (unsigned i = 0; i < result.num_rows; ++i)
            for (unsigned j = 0; j < result.num_cols; ++j)
                result.data[i][j] = data[i][j] - rhs.data[i][j];
        return result;
    }

    matrix<T> operator*(const matrix<T> &rhs) const {
        assert(num_cols == rhs.num_rows);
        matrix<T> result(num_rows, rhs.num_cols, T{});
        for (unsigned i = 0; i < num_rows; ++i)
            for (unsigned j = 0; j < rhs.num_cols; ++j)
                for (unsigned k = 0; k < rhs.num_rows; ++k)
                    result.data[i][j] += data[i][k] * rhs.data[k][j];
        return result;
    }

    matrix<T> operator*(int lhs) const {
        matrix<T> result(num_rows, num_cols);
        for (unsigned i = 0; i < result.num_rows; ++i)
            for (unsigned j = 0; j < result.num_cols; ++j)
                result.data[i][j] *= lhs;
        return result;
    }

    friend matrix<T> operator*(int lhs, const matrix<T> &rhs) {
        matrix<T> result(rhs.num_rows, rhs.num_cols);
        for (unsigned i = 0; i < result.num_rows; ++i)
            for (unsigned j = 0; j < result.num_cols; ++j)
                result.data[i][j] *= lhs;
        return result;
    }

    matrix<T> transpose() const {
        matrix<T> result(num_cols, num_rows);
        for (unsigned i = 0; i < result.num_rows; ++i)
            for (unsigned j = 0; j < result.num_cols; ++j)
                result.data[i][j] = data[j][i];
        return result;
    }

    static matrix<T> column(const vector<T> &vector) {
        matrix<T> result(vector.size(), 1);
        for (unsigned i = 0; i < result.num_rows; ++i)
            result[i][0] = vector[i];
        return result;
    }

    static matrix<T> row(const vector<T> &vector) {
        matrix<T> result(1, vector.size());
        result[0] = vector;
        return result;
    }

    set<unsigned> independent_rows() {
        vector<vector<double>> m = data;
        vector<bool> row_used(num_rows, false);
        for (unsigned i = 0; i < num_cols; ++i) {
            unsigned j;
            for (j = 0; j < num_rows; ++j)
                if (!row_used[j] && abs(m[j][i]) > epsilon)
                    break;
            if (j != num_rows) {
                row_used[j] = true;
                for (int p = i + 1; p < num_cols; ++p)
                    m[j][p] /= m[j][i];
                for (int k = 0; k < num_rows; ++k)
                    if (k != j && abs(m[k][i]) > epsilon)
                        for (int p = i + 1; p < num_cols; ++p)
                            m[k][p] -= m[j][p] * m[k][i];
            }
        }
        set<unsigned> independent;
        for (unsigned i = 0; i < num_rows; ++i) {
            if (row_used[i]) independent.insert(i);
        }
        return independent;
    }

    static matrix<double> gauss(matrix<double> A, matrix<double> b) {
        matrix<double> result(A.num_cols, 1);

        vector<int> where(A.num_cols, -1);
        for (int col = 0, row = 0; col < A.num_cols && row < A.num_rows; ++col) {
            int sel = row;
            for (int i = row; i < A.num_rows; ++i)
                if (abs(A[i][col]) > abs(A[sel][col]))
                    sel = i;
            if (abs(A[sel][col]) < epsilon) continue;
            for (int i = col; i < A.num_cols; ++i)
                swap(A[sel][i], A[row][i]);
            swap(b[sel][0], b[row][0]);
            where[col] = row;

            for (int i = 0; i < A.num_rows; ++i)
                if (i != row) {
                    double c = A[i][col] / A[row][col];
                    for (int j = col; j < A.num_cols; ++j)
                        A[i][j] -= A[row][j] * c;
                    b[i][0] -= b[row][0] * c;
                }
            ++row;
        }

        for (int i = 0; i < A.num_cols; ++i)
            if (where[i] != -1)
                result[i][0] = b[where[i]][0] / A[where[i]][i];
        return result;
    }
};

using matrix_t = matrix<double>;

class linear_regression {
public:
    static unique_ptr<linear_regression> make_regression(const vector<object> &train_set) {
        const auto &objects_size = train_set.size();
        const auto &features_size = train_set[0].size();

        // F^t[features+1 × objects]
        matrix_t Ft(features_size + 1, objects_size);
        for (unsigned i = 0; i < objects_size; ++i) {
            for (unsigned j = 0; j < features_size; ++j) {
                Ft[j][i] = train_set[i][j];
            }
            Ft[features_size][i] = 1.0;
        }

        // y[objects × 1]
        matrix_t y(objects_size, 1);
        for (unsigned i = 0; i < objects_size; ++i) {
            y[i][0] = train_set[i].get_class();
        }

        // Get linearly independent rows of F^t (that are also independent columns of F)
        set<unsigned> independent = Ft.independent_rows();

        // Build reduced F^t matrix
        matrix_t Ft_fixed(independent.size(), objects_size);
        int current_row = 0;
        for (unsigned i : independent) {
            Ft_fixed[current_row++] = Ft[i];
        }

        // Get reduced F matrix as a transposition
        matrix_t F_fixed = Ft_fixed.transpose();

        // F^t * F [features+1 × features+1]
        matrix_t F_cov = Ft_fixed * F_fixed;

        // F^t * y [features+1 × 1]
        matrix_t Fty = Ft_fixed * y;

        // a[features+1 × 1]
        matrix_t alpha = matrix_t::gauss(F_cov, Fty);

        vector<double> coefficients(features_size + 1, 0.0);
        current_row = 0;
        for (unsigned ind : independent) {
            coefficients[ind] = alpha[current_row++][0];
        }

        return make_unique<linear_regression>(coefficients);
    }

    string info() const {
        stringstream out;
        out << fixed << setprecision(8);
        for (double c : coefficients) {
            out << c << endl;
        }
        return out.str();
    }

    explicit linear_regression(vector<double> coefficients) : coefficients(move(coefficients)) {}

private:
    static constexpr double epsilon = 1e-8;
    const vector<double> coefficients;
};

void solve() {
    /**
     * m - number of features
     * n - number of objects in training set
     */
    size_t m, n;
    cin >> m >> n;

    vector<object> train_set;

    // Read training set
    for (unsigned object_id = 0; object_id < n; ++object_id) {
        vector<feature_t> features(m);
        for (feature_t &feature : features) {
            cin >> feature;
        }
        class_t class_id;
        cin >> class_id;
        train_set.emplace_back(features, class_id);
    }

    auto regression = linear_regression::make_regression(train_set);
    cout << regression->info() << endl;
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
