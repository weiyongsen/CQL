import numpy as np

min_h_diff = 50
min_psi_diff = 5
min_v_diff = 10
w = [0.8, 0.2]   # 默认权重，后续根据step修改
call_count = 0


def first_function(error_now, k):
    return 1 - (error_now / k) / (1 + error_now / k)


def second_function(error_now, k):
    scaled_error = error_now * 0.69 / k
    return np.exp(-scaled_error)


def get_reward_complex(h_error, psi_error, v_error, max_h_diff, max_psi_diff, max_v_diff, step=0):
    global w
    # 初始值和目标值的差
    max_h_diff = abs(max_h_diff)
    max_psi_diff = abs(max_psi_diff)
    max_v_diff = abs(max_v_diff)
    # 当前误差
    h_diff = abs(h_error)
    psi_diff = abs(psi_error)
    v_diff = abs(v_error)
    # 线性奖励(完成度)
    reward_h_linear = 1 - h_diff / max(max_h_diff, min_h_diff)
    reward_psi_linear = 1 - psi_diff / max(max_psi_diff, min_psi_diff)
    reward_v_linear = 1 - v_diff / max(max_v_diff, min_v_diff)

    k_h = 200
    k_psi = 10
    k_v = 25
    k_wv = 15  # 控制 w_v 衰减速率的参数

    # 非线性奖励
    reward_h_scale = first_function(h_diff, k_h)
    reward_psi_scale = first_function(psi_diff, k_psi)
    reward_v_scale = first_function(v_diff, k_v)

    # 计算 w_v 的动态权重
    w_v = 0.05 + 0.25 * np.exp(-0.7 * psi_diff / k_wv)

    # 计算剩余的权重（w_h 和 w_psi 保持比例）
    remaining_w = 1 - w_v
    # w_h = 0.3 * remaining_w / 0.7
    # w_psi = 0.4 * remaining_w / 0.7
    w_h = 0.3
    w_psi = 0.4

    # 更新权重条件
    w = [w_h, w_psi, w_v]
    # if step <= 1:
    #     w = [0.4, 0.4, 0.2]
    # elif step % 5 == 0 and (max_h_diff>500 or max_psi_diff>50 or max_v_diff>100):
    #     w = adapt_weights(h_diff, psi_diff, max_h_diff, max_psi_diff)

    return [reward_h_scale, reward_psi_scale, reward_v_scale], [reward_h_linear, reward_psi_linear, reward_v_linear], w


def adapt_weights(h_diff, psi_diff, max_h_diff, max_psi_diff):
    global call_count
    call_count += 1
    # 步骤 1：归一化各误差，超过1也只会增大对应的weight
    max_h_diff, max_psi_diff = max(max_h_diff, min_h_diff), max(max_psi_diff, min_psi_diff)
    norm_h_diff = h_diff / max_h_diff
    norm_psi_diff = psi_diff / max_psi_diff

    # 误差过小时，不频繁调整
    if hasattr(adapt_weights, 'previous_weights') and h_diff < max_h_diff/20 and psi_diff < max_psi_diff/10:
        return adapt_weights.previous_weights

    # 步骤 2：计算初始权重
    h_weight = norm_h_diff / (norm_h_diff + norm_psi_diff)
    psi_weight = norm_psi_diff / (norm_h_diff + norm_psi_diff)

    # 步骤 3：确保权重最低值为 0.2，最大值为 0.5
    min_weight = 0.4
    max_weight = 0.6
    weights = [h_weight, psi_weight]
    adjusted_weights = [max(min(w, max_weight), min_weight) for w in weights]

    # 步骤 4：重新归一化权重总和为 1
    total_adjusted_weight = sum(adjusted_weights)
    final_weights = [w / total_adjusted_weight for w in adjusted_weights]

    # 步骤 5：平滑处理权重
    if hasattr(adapt_weights, 'previous_weights'):
        if call_count <= 20:
            final_weights = [0.2 * prev + 0.8 * curr for prev, curr in zip(adapt_weights.previous_weights, final_weights)]
        # if call_count < 20:
        #     final_weights = [0.2 * prev + 0.8 * curr for prev, curr in zip(adapt_weights.previous_weights, final_weights)]
        # else:
        #     final_weights = [0.8 * prev + 0.2 * curr for prev, curr in zip(adapt_weights.previous_weights, final_weights)]
    adapt_weights.previous_weights = final_weights  # 记录上次权重
    return final_weights


def get_k(max_diff, thresholds, base_k):
    """根据最大差值和阈值计算 k"""
    if max_diff < thresholds[0]:
        return base_k[0]
    elif max_diff < thresholds[1]:
        return base_k[1]
    else:
        return base_k[2]


if __name__ == '__main__':  # 对比几种奖励函数的差距，可以直接运行
    import matplotlib.pyplot as plt
    # 示例用法
    # 设置多个 k 值
    k_values = [100, 200, 300, 400]  # 你可以根据需要调整 k 的多个值

    # 生成误差值
    error_now = np.linspace(0, 6000, 1000)  # 假设误差值从 0 到 100

    # 创建图表
    plt.figure(figsize=(8, 6))

    # 为每个 k 值计算并绘制两条曲线
    for k in k_values:
        R_scale_psi_first = first_function(error_now, k)
        R_scale_psi_second = second_function(error_now, k)

        # 绘制第一个函数的曲线
        plt.plot(error_now, R_scale_psi_first, label=f'k = {k} (First Function)', linewidth=2)
        # 绘制第二个函数的曲线
        plt.plot(error_now, R_scale_psi_second, label=f'k = {k} (Second Function)', linestyle='--', linewidth=2)

    # 设置图表标签和标题
    plt.xlabel('Error (error_now)', fontsize=12)
    plt.ylabel('Reward Value (R_scale_psi)', fontsize=12)
    plt.title('Comparison of Reward Functions for Different k Values', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
