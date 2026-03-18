#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;
using vvector = vector<vector<int>>;
int solve(vvector &tree_m, vector<int> &weight_m, int u, vvector &tree_n, vector<int> &weight_n, int v) {
    if (weight_m[u] != weight_n[v])
        return 0;
    if (tree_m[u].empty() || tree_n[v].empty())
        return 1;
    int best = 0;
    for (int i = 0; i < tree_m[u].size(); i++) {
        int p = tree_m[u][i];
        for (int j = 0; j < tree_n[v].size(); j++) {
            int q = tree_n[v][j];
            best = max(best, solve(tree_m, weight_m, p, tree_n, weight_n, q));
        }
    }
    return best + 1;
}
int main() {
    int m, n;
    cin >> m >> n;

    vector<int> weight_m(m, 0);
    vector<int> weight_n(n, 0);

    for (int i = 0; i < m; i++)
        cin >> weight_m[i];
    for (int i = 0; i < n; i++)
        cin >> weight_n[i];

    vvector tree_m(m);
    vvector tree_n(n);
    for (int i = 0; i < m - 1; i++) {
        int u, v;
        cin >> u >> v;
        int p = max(u, v) - 1, q = min(u, v) - 1;
        tree_m[q].push_back(p);
    }
    for (int i = 0; i < n - 1; i++) {
        int u, v;
        cin >> u >> v;
        int p = max(u, v) - 1, q = min(u, v) - 1;
        tree_n[q].push_back(p);
    }
    int ans = solve(tree_m, weight_m, 0, tree_n, weight_n, 0);
    cout << ans << endl;
    return 0;
}
