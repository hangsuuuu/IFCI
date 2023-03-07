import numpy as np
import torch
from scipy.stats import f


def casual_inference(args, datas):
# 后门公式矫正，计算E(Ze|do(e))
    means = []
    n = []
    # 分别计算每个环境下的means
    for i in range(len(datas)):
        data = datas[i][0]
        labels = datas[i][1]
        n.append(len(data))
        cla, num = np.unique(labels, return_counts=True)
        class_means = []
        pro_y = []
        # 按label分组计算P(y)*E(Ze|y,e)
        for label in cla:
            pos = np.where(labels == label)
            d = data[pos, :].squeeze()
            mean = np.mean(d, axis=0)
            class_means.append(mean)
            pro_y.append(num[label]/len(labels))
        y_means = np.empty(len(class_means[0]))
        for j in range(len(class_means[0])):
            y_mean = np.array(class_means)[:, j]
            y_means[j] = np.sum(pro_y*y_mean)
        means.append(y_means)
# 方差分析
    # 计算总均值
    c = 0
    total_means = np.empty([len(means), len(means[0])])
    for e in means:
        e = e*len(datas[c][1])
        total_means[c, :] = e
        c += 1

    total_means = np.sum(total_means, axis=0) / sum(n)
    # 计算组间方差MSA
    a = np.power(means - total_means, 2)
    for i in range(len(a[0])):
        a[:, i] *= n
    msa = np.sum(a, axis=0) / (args.n_env - 1)
    # 计算组内方差MSE
    E = np.empty([len(means), len(means[0])])
    for i in range(len(datas)):
        b = np.sum(np.power(datas[i][0] - means[i], 2), axis=0)
        E[i, :] = b
    mse = np.sum(E, axis=0) / (sum(n) - args.n_env)
    # MSA/MSE与F分布比较, 显著性水平0.05
    F = f.isf(q=0.05, dfn=args.n_env-1, dfd=(sum(n)-args.n_env))
    results = []
    for i in range(len(msa)):
        aa = msa[i] / mse[i]
        if (msa[i] / mse[i]) <= F:
            results.append(i)
    return results