#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unistd.h>
#include <sys/wait.h>
#include <cstdlib>

using namespace std;

// 去掉字符串首尾空白
string trim(const string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// 按空格分割命令
vector<string> splitCommand(const string& input) {
    vector<string> tokens;
    istringstream iss(input);
    string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

// 显示帮助信息
void showHelp() {
    cout << "My Shell 帮助信息:\n";
    cout << "内置命令:\n";
    cout << "  help           显示帮助信息\n";
    cout << "  history        显示历史命令\n";
    cout << "  clear          清屏\n";
    cout << "  exit           退出 shell\n";
    cout << "\n";
    cout << "支持直接执行 Linux 命令，例如:\n";
    cout << "  ls\n";
    cout << "  cat filename\n";
    cout << "  pwd\n";
    cout << "  date\n";
}

// 显示历史记录
void showHistory(const vector<string>& history) {
    for (size_t i = 0; i < history.size(); ++i) {
        cout << i + 1 << "  " << history[i] << endl;
    }
}

// 执行外部命令
void executeExternalCommand(const vector<string>& args) {
    if (args.empty()) return;

    pid_t pid = fork();

    if (pid < 0) {
        perror("fork failed");
        return;
    } 
    else if (pid == 0) {
        // 子进程
        vector<char*> c_args;
        for (const auto& arg : args) {
            c_args.push_back(const_cast<char*>(arg.c_str()));
        }
        c_args.push_back(nullptr);

        execvp(c_args[0], c_args.data());
        perror("exec failed");
        exit(1);
    } 
    else {
        // 父进程等待子进程结束
        waitpid(pid, nullptr, 0);
    }
}

int main() {
    vector<string> history;
    string input;

    while (true) {
        char cwd[1024];
        if (getcwd(cwd, sizeof(cwd)) != nullptr) {
            cout << "[my-shell " << cwd << "]$ ";
        } else {
            cout << "[my-shell]$ ";
        }

        getline(cin, input);
        input = trim(input);

        if (input.empty()) {
            continue;
        }

        history.push_back(input);

        vector<string> args = splitCommand(input);
        string cmd = args[0];

        // 内置命令
        if (cmd == "exit") {
            cout << "退出 shell...\n";
            break;
        } 
        else if (cmd == "help") {
            showHelp();
        } 
        else if (cmd == "history") {
            showHistory(history);
        } 
        else if (cmd == "clear") {
            system("clear");
        } 
        else {
            // 其他命令交给 Linux 执行
            executeExternalCommand(args);
        }
    }

    return 0;
}
