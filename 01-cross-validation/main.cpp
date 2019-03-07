#include <iostream>
#include <vector>

using namespace std;

int main() {
    unsigned n, m, k;
    scanf("%u %u %u", &n, &m, &k);

    vector<vector<int>> classes(m), parts(k);

    for (unsigned i = 0, c; i < n; i++) {
        scanf("%u", &c);
        classes[--c].push_back(i);
    }

    int p = 0;
    for (auto &cls : classes) {
        auto size = static_cast<unsigned>(cls.size());
        auto block_size = size / k;
        for (int i = 0; i < size; i++) {
            bool extra = i >= size - size % k;
            parts[extra ? p : (i / block_size)].push_back(cls[i]);
            if (extra) p = (p + 1) % k;
        }
    }

    for (auto &part : parts) {
        printf("%u ", static_cast<unsigned int>(part.size()));
        for (auto &obj : part) {
            printf("%u ", obj + 1);
        }
        printf("\n");
    }

    return 0;
}