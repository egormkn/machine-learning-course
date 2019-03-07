#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

void solve() {
    // TODO
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
