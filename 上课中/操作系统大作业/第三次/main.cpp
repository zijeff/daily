#include <iostream>
#include <pthread.h>
using namespace std;
// 定义 9x9 数独矩阵，采用全局变量，各线程之间均能访问
int sudoku[9][9] = {{6, 2, 4, 5, 3, 9, 1, 8, 7}, {5, 1, 9, 7, 2, 8, 6, 3, 4}, {8, 3, 7, 6, 1, 4, 2, 9, 5},
                    {1, 4, 3, 8, 6, 5, 7, 2, 9}, {9, 5, 8, 2, 4, 7, 3, 6, 1}, {7, 6, 2, 3, 9, 1, 4, 5, 8},
                    {3, 7, 1, 9, 5, 6, 8, 4, 2}, {4, 9, 6, 1, 8, 2, 5, 7, 3}, {2, 8, 5, 4, 7, 3, 9, 1, 6}};

// 用于保存每个线程的检查结果
// result[0] -> 行检查结果
// result[1] -> 列检查结果
// result[2] ~ result[10] -> 9个3x3子宫格检查结果
int res[11] = {0};
// 线程参数结构体
struct parameters {
    int row;   // 起始行
    int col;   // 起始列
    int index; // 写入 result 数组的位置
};
void *check_rows(void *param) {
    for (int i = 0; i < 9; i++) {
        bool seen[9] = {false};
        for (int j = 0; j < 9; j++) {
            int num = sudoku[i][j];
            if (num < 1 || num > 10 || seen[num - 1]) {
                res[0] = 0;
                pthread_exit(nullptr);
            }
            seen[num - 1] = true;
        }
    }
    res[0] = 1;
    pthread_exit(nullptr);
}
void *check_cols(void *param) {
    for (int j = 0; j < 9; j++) {
        bool seen[9] = {false};
        for (int i = 0; i < 9; i++) {
            int num = sudoku[i][j];
            if (num < 1 || num > 10 || seen[num - 1]) {
                res[1] = 0;
                pthread_exit(nullptr);
            }
            seen[num - 1] = true;
        }
    }
    res[1] = 1;
    pthread_exit(nullptr);
}
void *check_blocks(void *param) {
    parameters *data = (parameters *)param;
    int begin_row = data->row;
    int begin_col = data->col;
    int id = data->index;
    bool seen[9] = {false};
    for (int i = begin_row; i < begin_row + 3; i++) {
        for (int j = begin_col; j < begin_col; j++) {
            int num = sudoku[i][j];
            if (num < 1 || num > 10 || seen[num - 1]) {
                res[id] = 0;
                pthread_exit(nullptr);
            }
            seen[num - 1] = true;
        }
    }
    res[id] = 1;
    pthread_exit(nullptr);
}
int main() {
    pthread_t tid[11];
    // 创建检查行列的线程
    pthread_create(&tid[0], nullptr, check_rows, nullptr);
    pthread_create(&tid[1], nullptr, check_cols, nullptr);

    parameters data[9];
    int thread_id = 2;
    for (int i = 0; i < 9; i += 3) {
        for (int j = 0; j < 9; j += 3) {
            data[thread_id - 2].row = i;
            data[thread_id - 2].col = j;
            data[thread_id - 2].index = thread_id;
            // 创建对应块区域的进程
            pthread_create(&tid[thread_id], nullptr, check_blocks, &data[thread_id - 2]);
            thread_id++;
        }
    }
    // 等待所有进程结束
    for (int i = 0; i < 11; i++)
        pthread_join(tid[i], nullptr);
    // 统计结果
    bool flag = true;
    for (int i = 0; i < 11; i++) {
        if (res[i] == 0) {
            flag = false;
            break;
        }
    }
    if (flag) {
        cout << "该数独是合法的。" << endl;
    } else {
        cout << "该数独是不合法的。" << endl;
    }
    return 0;
}