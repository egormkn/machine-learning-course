#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

using namespace std;

double fscore(double precision, double recall) {
    return precision + recall < 1e-8 ? 0 : 2 * precision * recall / (precision + recall);
}

void solve() {
    /**
     * k - number of classes
     * value - [i, j] element of confusion matrix
     */
    unsigned k, value;
    cin >> k;

    vector<int> row_sum(k, 0), column_sum(k, 0), diag(k, 0);
    int total_sum = 0;

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            cin >> value;
            row_sum[i] += value;
            column_sum[j] += value;
            total_sum += value;
            if (i == j) diag[i] = value;
        }
    }

    double precision_sum = 0, recall_sum = 0, fscore_sum = 0;

    for (int i = 0; i < k; i++) {
        double precision = 0, recall = 0, weight = (double) row_sum[i] / total_sum;
        if (column_sum[i] > 0) precision = (double) diag[i] / column_sum[i] * weight;
        if (row_sum[i] > 0) recall = (double) diag[i] / row_sum[i] * weight;

        precision_sum += precision;
        recall_sum += recall;
        fscore_sum += fscore(precision, recall);
    }

    cout << setprecision(10) << fscore(precision_sum, recall_sum) << endl;
    cout << setprecision(10) << fscore_sum << endl;
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
