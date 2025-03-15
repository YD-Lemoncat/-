import numpy as np  # 导入 NumPy 库，用于数值计算
import matplotlib.pyplot as plt  # 导入 Matplotlib 库，用于绘图
import scipy.linalg  # 导入 SciPy 的线性代数模块，用于计算全局最小二乘（GLS）估计中的零空间
from tqdm import tqdm  # 导入 tqdm 库，用于显示进度条

# 解决 Matplotlib 显示中文字符和负号问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei，用于显示中文
plt.rcParams['axes.unicode_minus'] = False  # 设置负号显示为正常格式

#TODO 定义计算本地时间函数
def get_localtime(t_send, omega, phi, N, K):
    t_send_local = np.zeros((1, K, N))  # 初始化本地时间数组，尺寸为 1xKxN，表示一个1维数组，包含 K 个样本，每个样本有 N 个节点的时间数据
    # 遍历每个节点和样本，计算本地时间
    for i in range(N):
        for j in range(K):
            t_send_local[0, j, i] = t_send[j] * omega[i] + phi[i]  # 本地时间转换公式
    return t_send_local  # 返回本地时间

#TODO 定义计算时间戳函数
def calculate_timestamps(c, iPosition, jPosition, K, omega_i, phi_i, omega_j, phi_j, t_send):
    iReceiveTimes = np.zeros(K // 2)  # 初始化节点 i 的接收时间数组
    jReceiveTimes = np.zeros(K // 2)  # 初始化节点 j 的接收时间数组
    
    i_send_local = t_send[:K // 2] * omega_i + phi_i  # 计算节点 i 的发送时间
    jSendTimes = omega_j * t_send[K // 2:K] + phi_j  # 计算节点 j 的发送时间
    dis = np.linalg.norm(iPosition - jPosition)  # 计算两个节点之间的距离
    tau = dis / c  # 计算信号传输时间
    
    # 遍历每个样本，计算接收时间
    for i in range(K // 2):
        jReceiveTimes[i] = omega_j * (t_send[i] + tau) + phi_j  # 节点 j 的接收时间
        iReceiveTimes[i] = omega_i * (t_send[i + K // 2] + tau) + phi_i  # 节点 i 的接收时间
    
    # 合并发送和接收时间
    t_ij = np.hstack([i_send_local, iReceiveTimes])  # 节点 i 的发送和接收时间
    t_ji = np.hstack([jReceiveTimes, jSendTimes])  # 节点 j 的接收和发送时间
    return t_ij, t_ji  # 返回两个节点的时间戳

#TODO 仿真数据设置
# 位置设置
X = np.array([[615, -764, -19, 823],  # 节点的 x 坐标
              [-130, 443, 296, -973]])  # 节点的 y 坐标

# 速度和相位偏移设置
omega = np.array([1, 0.9999, 0.9994, 0.9998])  # 节点的速度因子
phi = np.array([0, 0.998, 1.000, 1.002])  # 节点的相位偏移
t_send = np.arange(0, 3.03, 0.03)  # 全局发送时间序列
N = 4  # 节点数量
c = 3e8  # 光速，用于计算信号传输速度
sigma = 0.045  #FIXME 噪声方差
N_hat = int(N * (N - 1) / 2)  # 传输路径的数量(从 N 个节点中选择 2 个节点)

# α 和 β 转换（逆速度因子和校准相位）
alpha = 1 / omega  # α 速度因子的倒数
beta = -phi / omega  # β 相位偏移的负值除以速度因子

# 计算节点之间的距离
dist = np.zeros(N_hat)  # 初始化距离数组
n = 0
for i in range(N):
    for j in range(i + 1, N):
        dist[n] = np.linalg.norm(X[:, i] - X[:, j])  # 计算每对节点的欧几里得距离
        n += 1

# 初始化仿真结果保存数组
KL = 10  # 样本数倍数
#全局最小二乘法估计误差初始化
omega_error = np.zeros((KL, 10000))  # 保存全局最小二乘法 omega 估计误差
phi_error = np.zeros((KL, 10000))  # 保存全局最小二乘法 phi 估计误差
tau_error = np.zeros((KL, 10000))  # 保存 tau 估计误差
#局部最小二乘法估计误差初始化
omega_error_lcls = np.zeros((KL, 10000))  # 保存局部最小二乘法的 omega 估计误差
phi_error_lcls = np.zeros((KL, 10000))  # 保存局部最小二乘法的 phi 估计误差
#Cramer-Rao 下界初始化
omega_CRLB = np.zeros((KL, 10000))  # 保存 omega Cramer-Rao 下界
phi_CRLB = np.zeros((KL, 10000))  # 保存 phi Cramer-Rao 下界
tau_CRLB = np.zeros((KL, 10000))  # 保存 tau Cramer-Rao 下界


#TODO 开始仿真实验
for kl in range(1, KL + 1):
    K = kl * 10  # 样本数
    for mc in tqdm(range(10), desc='Inner Loop Progress', leave=True): # FIXME 重复实验次数设置
        sigma_eta = np.linalg.inv(sigma**2 * np.eye(N_hat * K))  # 计算噪声协方差矩阵的逆
        t_send_local = get_localtime(t_send, omega, phi, N, K)  # 计算本地时间

        e = np.array([1, -1])  # 差分矩阵的基础向量
        eij = np.kron(e, np.ones(K // 2))  # 差分矩阵
        Eij = np.diag(eij)  # 生成对角矩阵
        E = np.kron(np.eye(N_hat), Eij)  # 差分矩阵扩展到所有路径

        V = np.kron(np.eye(N_hat), np.ones((K, 1)))  # 单位向量矩阵
        V_hat = E @ V  # 计算扩展的单位向量矩阵

        T = np.zeros((N_hat * K, N))  # 初始化时间戳矩阵
        T_CRB = np.zeros_like(T)  # 初始化 Cramer-Rao 矩阵
        t = np.zeros(N_hat * K)  # 初始化时间差数组
        H = np.zeros_like(T)  # 初始化 H 矩阵(E_1矩阵)

        n = 0
        for i in range(N):
            for j in range(i + 1, N):
                # 计算两个节点之间的时间戳
                t_ij, t_ji = calculate_timestamps(c, X[:, i], X[:, j], K, omega[i], phi[i], omega[j], phi[j], t_send[:K])

                # 模拟时间戳并加入噪声
                T[n * K:(n + 1) * K, i] = t_ij + sigma * np.random.randn(len(t_ij))  # 节点 i
                T[n * K:(n + 1) * K, j] = -t_ji - sigma * np.random.randn(len(t_ji))  # 节点 j
                t[n * K:(n + 1) * K] = T[n * K:(n + 1) * K, i] + T[n * K:(n + 1) * K, j]  # 计算时间差
                T_CRB[n * K:(n + 1) * K, i] = t_ij  # 保存真实时间戳
                T_CRB[n * K:(n + 1) * K, j] = -t_ji  # 保存真实时间戳

                H[n * K:(n + 1) * K, i] = 1  # H 矩阵中 i 列为 1
                H[n * K:(n + 1) * K, j] = -1  # H 矩阵中 j 列为 -1
                n += 1

        # 最小二乘（LS）估计
        H_lcls = np.zeros((N_hat * (K // 2), N))  # 初始化局部最小二乘法的 H 矩阵
        T_lcls = np.zeros((N_hat * (K // 2), N))  # 初始化局部最小二乘法的时间戳矩阵

        m = 0
        for i in range(N):
            for j in range(i + 1, N):
                # 计算局部最小二乘法的时间戳
                T_lcls[m * (K // 2):(m + 1) * (K // 2), i] = T[m * K:m * K + (K // 2), i] + T[m * K + (K // 2):(m + 1) * K, i]
                T_lcls[m * (K // 2):(m + 1) * (K // 2), j] = T[m * K:m * K + (K // 2), j] + T[m * K + (K // 2):(m + 1) * K, j]

                H_lcls[m * (K // 2):(m + 1) * (K // 2), i] = 2  # H 矩阵中 i 列为 2
                H_lcls[m * (K // 2):(m + 1) * (K // 2), j] = -2  # H 矩阵中 j 列为 -2
                m += 1

        # 广义最小二乘（GLS）估计
        tij = T[:, 0]  # 提取时间差矩阵的第一列
        A = np.hstack([T[:, 1:], H[:, 1:], V_hat])  # 构造 GLS 矩阵 A
        A_CRB = np.hstack([T_CRB, H, V_hat])  # 构造 Cramer-Rao 矩阵 A

        # 局部最小二乘法（LCLS）估计
        bij = T_lcls[:, 0]  # 提取局部最小二乘法时间差矩阵的第一列
        A_lcls = np.hstack([T_lcls[:, 1:], H_lcls[:, 1:]])  # 构造 LCLS 矩阵 A

        # 求解
        theta_estimate = np.linalg.pinv(A.T @ A) @ A.T @ -tij  # 求解 GLS
        theta_estimate_LCLS = np.linalg.pinv(A_lcls.T @ A_lcls) @ A_lcls.T @ -bij  # 求解 LCLS

        # 估计速度因子和相位偏移
        # GLS 估计
        omega_estimate = 1. / theta_estimate[:N-1]  # GLS估计的速度因子
        phi_estimate = -omega_estimate * theta_estimate[N-1:2*N-2]  # GLS估计的相位偏移
        # LCLS 估计
        omega_estimate_lcls = 1. / theta_estimate_LCLS[:N-1]  # LCLS 估计的速度因子
        phi_estimate_lcls = -omega_estimate_lcls * theta_estimate_LCLS[N-1:2*N-2]  # LCLS 估计的相位偏移

        # 误差分析
        # LCLS 误差
        omega_error_lcls[kl-1, mc] = np.sqrt(np.mean((omega_estimate_lcls - omega[1:])**2))  # LCLS omega 估计误差
        phi_error_lcls[kl-1, mc] = np.sqrt(np.mean((phi_estimate_lcls - phi[1:])**2))  # LCLS phi 估计误差
        # GLS 误差
        omega_error[kl-1, mc] = np.sqrt(np.mean((omega_estimate - omega[1:])**2))  # omega 估计误差
        phi_error[kl-1, mc] = np.sqrt(np.mean((phi_estimate - phi[1:])**2))  # phi 估计误差
        # GLS 传播延迟tau误差
        tau_true_v = dist/c  # 真实的传播延迟 tau
        tau_estimate_v = theta_estimate[2*N-2:]  # 估计的传播延迟 tau
        tau_error[kl-1, mc] = np.sqrt(np.mean((tau_estimate_v - tau_true_v)**2))  # GLS 传播延迟估计误差

        #TODO Cramer-Rao 下界计算
        C1 = np.vstack([np.hstack([np.eye(1), np.zeros((1, N-1)), np.zeros((1, N)), np.zeros((1, N_hat))]),
                        np.hstack([np.zeros((1, N)), np.eye(1), np.zeros((1, N-1)), np.zeros((1, N_hat))])])
        
        U = scipy.linalg.null_space(C1)  # 计算 C1 的零空间
        CRLB_m = U @ np.linalg.pinv(U.T @ A_CRB.T @ sigma_eta @ A_CRB @ U) @ U.T  # 计算 CRLB 矩阵

        omega_CRLB[kl-1, mc] = np.sqrt(1 / N * np.trace(CRLB_m[:N, :N]))  # omega CRLB
        phi_CRLB[kl-1, mc] = np.sqrt(1 / N * np.trace(CRLB_m[N:2*N, N:2*N]))  # phi CRLB
        tau_CRLB[kl-1, mc] = np.sqrt(1 / N_hat * np.trace(CRLB_m[2*N:, 2*N:]))  # tau CRLB

# 计算各误差和 CRLB 的平均值
o_error = np.mean(omega_error, axis=1)  # omega 估计误差的均值
p_error = np.mean(phi_error, axis=1)  # phi 估计误差的均值
o_CRLB = np.mean(omega_CRLB, axis=1)  # omega CRLB 的均值
p_CRLB = np.mean(phi_CRLB, axis=1)  # phi CRLB 的均值
t_error = np.mean(tau_error, axis=1)  # 传播延迟 tau 估计误差的均值
t_CRLB = np.mean(tau_CRLB, axis=1)  # 传播延迟 tau CRLB 的均值
o_error_lcls = np.mean(omega_error_lcls, axis=1)  # LCLS omega 估计误差的均值
p_error_lcls = np.mean(phi_error_lcls, axis=1)  # LCLS phi 估计误差的均值

#TODO 绘制分析图
plt.figure(figsize=(18, 6))

# 绘制 omega 分析图
plt.subplot(1, 3, 1)
plt.plot(o_error, '-o', color='b', linewidth=1.5, markersize=8, label='omega GLS RMSE')  # 绘制 GLS omega 估计误差图
plt.plot(o_error_lcls, '-s', color='r', linewidth=1.5, markersize=8, label='omega LCLS RMSE')  # 绘制 LCLS omega 估计误差图
plt.plot(o_CRLB, '-d', color='g', linewidth=1.5, markersize=8, label='omega CRLB')  # 绘制 omega CRLB 图
plt.legend(loc='best')  # 添加图例
plt.title('omega RMSE 分析')  # 图表标题
plt.yscale('log')  # 设置 y 轴为对数刻度
plt.xlabel('K (10样本倍数)')  # x 轴标签
plt.ylabel('均方根误差 (RMSE)')  # y 轴标签
plt.grid(True)  # 添加网格线

# 绘制 phi 分析图
plt.subplot(1, 3, 2)
plt.plot(p_error, '-o', color='b', linewidth=1.5, markersize=8, label='phi GLS RMSE')  # 绘制GLS phi 估计误差图
plt.plot(p_error_lcls, '-s', color='r', linewidth=1.5, markersize=8, label='phi LCLS RMSE')  # 绘制 LCLS phi 估计误差图
plt.plot(p_CRLB, '-d', color='g', linewidth=1.5, markersize=8, label='phi CRLB')  # 绘制 phi CRLB 图
plt.legend(loc='best')  # 添加图例
plt.title('phi RMSE 分析')  # 图表标题
plt.yscale('log')  # 设置 y 轴为对数刻度
plt.xlabel('K (10样本倍数)')  # x 轴标签
plt.ylabel('均方根误差 (RMSE)')  # y 轴标签
plt.grid(True)  # 添加网格线

# 绘制 tau 分析图
plt.subplot(1, 3, 3)
plt.plot(t_error, '-o', color='b', linewidth=1.5, markersize=8, label='tau GLS RMSE')  # 绘制GLS tau 估计误差图
plt.plot(t_CRLB, '-d', color='g', linewidth=1.5, markersize=8, label='tau CRLB')  # 绘制 tau CRLB 图
plt.legend(loc='best')  # 添加图例
plt.title('tau RMSE 分析')  # 图表标题
plt.yscale('log')  # 设置 y 轴为对数刻度
plt.xlabel('K (10样本倍数)')  # x 轴标签
plt.ylabel('均方根误差 (RMSE)')  # y 轴标签
plt.grid(True)  # 添加网格线

plt.tight_layout()
plt.show()  # 显示图表

