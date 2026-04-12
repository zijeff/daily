## Mini Shell

一个在 Linux 环境下使用 C++ 实现的简易 Shell 程序。

该程序模拟了 Shell 的基本工作流程，能够读取用户输入、解析命令、执行内置命令和外部命令，并支持历史命令功能。

### 功能介绍

本程序支持以下功能：

#### 内置命令

- `help`  
  显示帮助信息

- `history`  
  显示历史命令记录

- `clear`  
  清屏

- `exit`  
  退出 Shell

- `!!`  
  执行最近一条历史命令

- `!n`  
  执行历史记录中的第 `n` 条命令

#### 外部命令

本程序支持执行 Linux 系统中的普通外部命令，例如：

- `ls`
- `cat filename`
- `pwd`
- `date`

这些命令通过 `fork()`、`execvp()` 和 `waitpid()` 实现。

### 运行环境

- Linux 操作系统
- g++ 编译器
- 支持 POSIX 系统调用

### 编译方法

将源代码保存为 `mini_shell.cpp`，在终端中执行：

```bash
g++ mini_shell.cpp -o mini_shell
```
