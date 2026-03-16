#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/string.h>

#define BUF_SIZE 512

static char res[BUF_SIZE];
// 利用简单的试除法判断是否为质数
int is_prime(int n)
{
    if (n < 2)
        return 0;

    for (int i = 2; i * i <= n; i++)
    {
        if (n % i == 0)
            return 0;
    }

    return 1;
}

// 生成 1~100 内的所有质数，并拼接成字符串
static void generate_primes(void)
{
    int i;
    int len = 0;

    memset(res, 0, BUF_SIZE);

    for (i = 1; i <= 100; i++) {
        if (is_prime(i))
            len += snprintf(res + len, BUF_SIZE - len, "%d ", i);
    }
}

// 加载模块
static int my_init(void)
{
    pr_info("Loading Module\n");
    pr_info("prime numbers from 1 to 100 are:\n");
    generate_primes();
    pr_info("%s\n", res);

    return 0;
}

// 卸载模块
static void my_exit(void)
{
    pr_info("Removing Module\n");
}

module_init(my_init);
module_exit(my_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Generate all primes from 1 to 100.");
MODULE_AUTHOR("zijeff and Qrx");