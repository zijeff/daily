// 虽然这段代码会超时，但是这是填空题，直接本地跑完，然后交答案骗分（）
#include <iostream>
#include <vector>
using namespace std;
bool solve(int x) {
    vector<int> bits;
    while (x > 0) {
        bits.push_back(x % 10);
        x = x / 10;
    }
    int n = bits.size();
    if (n % 2 != 0)
        return false;
    int cmp1 = 0, cmp2 = 0;
    for (int i = 0; 2 * i < n; i++) {
        cmp1 += bits[i];
        cmp2 += bits[n - 1 - i];
    }
    if (cmp1 != cmp2)
        return false;
    return true;
}
int main() {
    int ans = 0;
    for (int i = 10; i < 100000000; i++) {
        if (solve(i))
            ans++;
    }
    cout << ans << endl;
    return 0;
}
