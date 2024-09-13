import copy
import math
import numpy as np


def get_ps(P):
    function_value = copy.deepcopy(P)
    size, objective = function_value.shape
    index = []
    for i in range(size):
        diff = np.tile(function_value[i], (size, 1)) - function_value
        less = np.sum(diff < 0, axis=1).reshape(size, )
        equal = np.sum(diff == 0, axis=1).reshape(size, )
        dominated_index = ((less == 0) & (equal != objective))
        if np.sum(dominated_index) == 0:
            index.append(i)
    return P[index]


def GD(A, P):
    size = A.shape[0]
    dis = 0
    for i in range(size):
        s = A[i]
        diff = np.tile(s, (P.shape[0], 1)) - P
        dis += np.sum(diff ** 2, axis=1).min()
    gd = math.sqrt(dis) / size
    return gd


def IGD(P, A):
    size = P.shape[0]
    dis = 0
    for i in range(size):
        s = P[i]
        diff = np.tile(s, (A.shape[0], 1)) - A
        dis += np.sum(diff ** 2, axis=1).min()
    igd = math.sqrt(dis) / size
    return igd


def SC(A: np.ndarray, B: np.ndarray):
    """
        SC测度计算， C(A,B)表示B中存在的被A中至少一个解支配的解所占百分比，且在通常情况
下C(A,B) != 1−C(B,A)。如果C(A,B)>C(B,A)，则算法A比算法B的收敛性好。
    :param A: A解集合
    :param B: B解集合
    :return: SC测度值
    """
    obj_value_a = np.array(A)
    obj_value_b = np.array(B)
    size = obj_value_b.shape[0] # B集合个数
    msize = obj_value_b.shape[1] # 目标函数个数
    count = 0 # A支配B个体的数目
    for i in range(obj_value_a.shape[0]):
        diff = np.tile(obj_value_a[i], (size, 1)) - obj_value_b
        big = np.sum(diff > 0, axis=1).reshape(size, )
        equal = np.sum(diff == 0, axis=1).reshape(size, )
        dominate_index = ((big == 0) & (equal != msize))
        if np.sum(dominate_index) != 0:
            count += 1
    return count / size


def spread(A, P):
    sizeA = A.shape[0]
    objs = A.shape[1]
    d_AA = np.zeros(sizeA)
    flag = np.ones(sizeA).astype(bool)
    for i in range(sizeA):
        s = A[i]
        flag[i] = False
        diff = np.tile(s, (sizeA - 1, 1)) - A[flag]
        flag[i] = True
        d_AA[i] = math.sqrt(np.sum(diff ** 2, axis=1).min())
    ave_d = d_AA.mean()

    dAP = 0
    for o in range(objs):
        singleA = A[:, o]
        singleP = P[:, o]
        amax = singleA.max()
        amin = singleA.min()
        pmax = singleP.max()
        pmin = singleP.min()
        dAP += max(amax - pmin, pmax - amin)

    v = np.abs((d_AA - ave_d)).sum()
    deta = (dAP + v) / (dAP + ave_d * sizeA)
    return deta