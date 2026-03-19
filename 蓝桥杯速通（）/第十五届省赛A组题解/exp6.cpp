// 并非AC代码，只能处理20%的数据
#include <algorithm>
#include <iostream>
#include <queue>
#include <unordered_set>
#include <vector>
using namespace std;
using vvector = vector<vector<int>>;
using project = pair<int, int>;
int solve(vvector &road, vector<int> &c, int begin, int end) {
    int n = road.size();
    vector<int> vis(n, 0);
    vector<int> parent(n, -1);
    queue<int> q;
    q.push(begin);
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        vis[u] = 1;
        if (u == end)
            break;
        for (int i = 0; i < road[u].size(); i++) {
            int v = road[u][i];
            if (vis[v] != 1) {
                q.push(v);
                parent[v] = u;
            }
        }
    }
    unordered_set<int> res;
    int cur = end;
    while (cur != -1) {
        res.insert(c[cur]);
        if (cur == begin)
            break;
        cur = parent[cur];
    }
    return res.size();
}
int main() {
    int n, q;
    cin >> n >> q;
    vector<int> c(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> c[i];
    }
    vvector road(n);
    for (int i = 0; i < n - 1; i++) {
        int u, v;
        cin >> u >> v;
        road[u - 1].push_back(v - 1);
        road[v - 1].push_back(u - 1);
    }
    vector<project> plan(q);
    for (int i = 0; i < q; i++) {
        int u, v;
        cin >> u >> v;
        plan[i] = {u - 1, v - 1};
    }
    for (int i = 0; i < q; i++) {
        int ans = solve(road, c, plan[i].first, plan[i].second);
        cout << ans << endl;
    }
    return 0;
}
