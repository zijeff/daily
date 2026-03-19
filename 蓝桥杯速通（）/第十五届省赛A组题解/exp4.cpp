#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

bool check(int x, vector<int> &grade, int k, long long t) {
    if (x < k) return false;

    vector<int> data;
    for (int i = 0; i < x; i++)
        data.push_back(grade[i]);

    sort(data.begin(), data.end());

    vector<long long> pre1;
    vector<long long> pre2;
    long long s1 = 0, s2 = 0;

    pre1.push_back(0);
    pre2.push_back(0);

    for (int i = 0; i < data.size(); i++) {
        s1 += data[i];
        pre1.push_back(s1);
        s2 += 1LL * data[i] * data[i];
        pre2.push_back(s2);
    }

    for (int i = 1; i <= x + 1 - k; i++) {
        long long S1 = pre1[i + k - 1] - pre1[i - 1];
        long long S2 = pre2[i + k - 1] - pre2[i - 1];
        if (1LL * k * S2 - S1 * S1 < 1LL * k * k * t)
            return true;
    }
    return false;
}

int main() {
    int n, k;
    long long t;
    cin >> n >> k >> t;

    vector<int> grade(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> grade[i];
    }

    int ans = -1;
    if (check(n, grade, k, t)) {
        int l = k, r = n;
        while (l < r) {
            int mid = (l + r) / 2;
            if (check(mid, grade, k, t))
                r = mid;
            else
                l = mid + 1;
        }
        ans = l;
    }

    cout << ans << endl;
    return 0;
}
