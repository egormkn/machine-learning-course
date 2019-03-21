#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

void solve() {
    /**
     * n - number of objects
     */
    unsigned objects_size;
    cin >> objects_size;

    vector<double> x(objects_size), y(objects_size);
    double x_mean = 0.0, y_mean = 0.0;

    for (unsigned i = 0; i < objects_size; ++i) {
        cin >> x[i] >> y[i];
        x_mean += x[i] / objects_size;
        y_mean += y[i] / objects_size;
    }

    double numerator = 0, squares_sum_x = 0, squares_sum_y = 0;
    for (unsigned i = 0; i < objects_size; ++i) {
        numerator += (x[i] - x_mean) * (y[i] - y_mean);
        squares_sum_x += pow(x[i] - x_mean, 2);
        squares_sum_y += pow(y[i] - y_mean, 2);
    }

    double denominator = sqrt(squares_sum_x * squares_sum_y);

    cout << setprecision(10) << (denominator > 0 ? numerator / denominator : 0.0);
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
