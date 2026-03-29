import numpy as np

def segment(xy):
    
    x = xy[:, 0]
    y = xy[:, 1]

  
    threshold = 0.1

    S_i = 1  # size of segment
    S_n = 1  # number of segment

    # n0ind = np.where((x != 0) | (y != 0))[0]
    # n_0 = len(n0ind)
    n0ind = np.nonzero((x != 0) | (y != 0))[0]
    n_0 = len(n0ind)

    
    Seg = []
    # 添加第一个元素
    Seg.append([n0ind[0]])
    # Seg = np.zeros((n_0, n_0))  # Initialize Seg with zeros
    # Seg[0, 0] = n0ind[0]

    for i in range(1, n_0):
        distance = np.sqrt((x[n0ind[i]] - x[n0ind[i - 1]])**2 + (y[n0ind[i]] - y[n0ind[i - 1]])**2)
        if distance < threshold:
            S_i += 1
            # 添加到当前行
            Seg[S_n-1].append(n0ind[i])

        else:
            S_n += 1
            S_i = 1
            # 添加新的一行
            Seg.append([n0ind[i]])


    # 转换为 NumPy 数组
    Seg = np.array(Seg, dtype=object)
    

    Si_n = np.zeros(S_n, dtype=int)
    for j in range(S_n):
        #k = np.count_nonzero(Seg[j] != 0)
        Si_n[j]=len(Seg[j])
        #k = np.count_nonzero(Seg[j] != 0)
        #Si_n[j] = k

    return Seg, Si_n, S_n
    # for i in range(1, n_0):
    #     if np.sqrt((x[n0ind[i]] - x[n0ind[i - 1]])**2 + (y[n0ind[i]] - y[n0ind[i - 1]])**2) < threshold:
    #         S_i += 1
    #         Seg[S_i, S_n] = n0ind[i]
    #     else:
    #         S_n += 1
    #         S_i = 1
    #         Seg[S_i, S_n] = n0ind[i]

    # Si_n = np.zeros(S_n)
    # for j in range(S_n):
    #     Si_n[j] = np.sum(Seg[:, j] != 0)

    # return Seg, Si_n, S_n

