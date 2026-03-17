#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;
typedef struct {
    int price;
    int time;
} solider;

int main() {
    int n = 0;
    long long s = 0;
    cin >> n >> s;

    vector<solider> people(n);

    int max_time = 0;
    long long cost_one_time = 0;
    long long sum = 0;

    for (int i = 0; i < n; i++) {
        cin >> people[i].price >> people[i].time;
        max_time = max(max_time, people[i].time);
        cost_one_time += people[i].price;
    }
    vector<long long> finish(max_time);
    for (int i = 0; i < n; i++) {
        int k = people[i].time - 1;
        finish[k] += people[i].price;
    }
    for (int k = 0; k < max_time; k++) {
        if (cost_one_time > s)
            sum += s;
        else
            sum += cost_one_time;
        cost_one_time -= finish[k];
    }
    cout << sum << endl;
    return 0;
}
