import numpy as np


def first_function(error_now, k):
    return 1 - (error_now / k) / (1 + error_now / k)


def second_function(error_now, k):
    scaled_error = error_now * 0.69 / k
    return np.exp(-scaled_error)


def third_function(error_now, k, p=1.5):
    # 广义幂函数
    return 1 / (1 + (error_now / k) ** p)


def fourth_function(error_now, k, p=1.0, epsilon=0.1):
    """自适应幂函数"""
    normalized_error = abs(error_now) / (k + epsilon)  # 防止除零
    return 1 / (1 + normalized_error ** p) * (1 + np.tanh(1/(normalized_error + 0.1)))


def seventh_function(error_now, k, alpha=0.5):
    scaled_error = (error_now / k) ** alpha
    return np.exp(-scaled_error)


def get_reward_complex(h_error, psi_error, v_error, max_h_diff, max_psi_diff, max_v_diff, step=0):
    # 取绝对值
    max_h_diff = abs(max_h_diff)
    max_psi_diff = abs(max_psi_diff)
    max_v_diff = abs(max_v_diff)
    h_diff = abs(h_error)
    psi_diff = abs(psi_error)
    v_diff = abs(v_error)

    # 线性奖励(完成度)
    reward_h_linear = get_reward_linear_or_nonlinear(h_diff, max_h_diff, 50, first_function, 200)
    reward_psi_linear = get_reward_linear_or_nonlinear(psi_diff, max_psi_diff, 5, first_function, 10)
    reward_v_linear = get_reward_linear_or_nonlinear(v_diff, max_v_diff, 10, first_function, 25)

    # 非线性奖励参数
    k_h = 200
    k_psi = 10
    k_v = 20

    # 非线性奖励
    reward_h_scale = first_function(h_diff, k_h)
    reward_psi_scale = first_function(psi_diff, k_psi)
    reward_v_scale = first_function(v_diff, k_v)

    return [reward_h_scale, reward_psi_scale, reward_v_scale], \
        [reward_h_linear, reward_psi_linear, reward_v_linear]


def get_reward_linear_or_nonlinear(error, max_diff, min_diff, nonlinear_func, nonlinear_k):
    if max_diff < min_diff:
        return nonlinear_func(error, nonlinear_k)
    else:
        linear = 1 - error / max(max_diff, min_diff)
        return np.clip(linear, 0, 1)


if __name__ == '__main__':  # 对比几种奖励函数的差距，可以直接运行
    import matplotlib.pyplot as plt

    # 设置参数
    k_values = [10, 16, 20]
    p_values = [1.2]
    error_now = np.linspace(0, 1000, 1000)

    # 颜色分配
    color_map = {
        'first': '#1f77b4',   # 蓝色
        'third': '#d62728',  # 红色
        'second': '#2ca02c',   # 绿色
        'fourth': '#9467bd',  # 紫色
    }
    # 线型分配
    linestyles = ['-', '--', '-.', ':']

    plt.figure(figsize=(14, 8))

    ## first_function
    for idx, k in enumerate(k_values):
        plt.plot(
            error_now,
            first_function(error_now, k),
            color=color_map['first'],
            linestyle=linestyles[idx % len(linestyles)],
            linewidth=2,
            label=f'first, k={k}'
        )

    # # second_function
    for idx, k in enumerate(k_values):
        plt.plot(
            error_now,
            second_function(error_now, k),
            color=color_map['second'],
            linestyle=linestyles[idx % len(linestyles)],
            linewidth=2,
            label=f'second, k={k}'
        )

    # third_function (不同p值)
    for p_idx, p in enumerate(p_values):
        for idx, k in enumerate(k_values):

                plt.plot(
                    error_now,
                    third_function(error_now, k, p=p),
                    color=color_map['third'],
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=1.5 + 0.5 * p_idx,  # p越大线越粗
                    alpha=0.7 + 0.1 * p_idx,      # p越大越不透明
                    label=f'third, k={k}, p={p}'   # 只在每组p的第一个k加label，避免图例太多
                )

    # # fourth_function (不同p值)
    # for p_idx, p in enumerate(p_values):
    #     for idx, k in enumerate(k_values):
    #         plt.plot(
    #             error_now,
    #             fourth_function(error_now, k, p=p),
    #             color=color_map['fourth'],
    #             linestyle=linestyles[idx % len(linestyles)],
    #             linewidth=1.5 + 0.5 * p_idx,  # p越大线越粗
    #             alpha=0.7 + 0.1 * p_idx,      # p越大越不透明
    #             label=f'fourth, k={k}, p={p}' if idx == 0 else None  # 只在每组p的第一个k加label，避免图例太多
    #         )

    # 图例只保留每种function和p的代表
    handles, labels = plt.gca().get_legend_handles_labels()
    new_handles = []
    new_labels = []
    seen = set()
    for h, l in zip(handles, labels):
        if l not in seen and l is not None:
            new_handles.append(h)
            new_labels.append(l)
            seen.add(l)
    plt.legend(new_handles, new_labels, loc='upper right', fontsize=10, ncol=2)

    plt.xlabel('Error (error_now)', fontsize=13)
    plt.ylabel('Reward Value', fontsize=13)
    plt.title('Reward Function Comparison: Color=Function, Linestyle=k, Linewidth/Alpha=p', fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
