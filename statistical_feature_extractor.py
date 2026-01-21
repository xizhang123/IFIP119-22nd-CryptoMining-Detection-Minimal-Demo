import numpy as np

class SafeArray(np.ndarray):
    def mean(self, *args, **kwargs):
        if self.size == 0:
            return 0.0
        return super().mean(*args, **kwargs)
    
    def std(self, *args, **kwargs):
        if self.size == 0:
            return 0.0
        return super().std(*args, **kwargs)

def as_safe_array(arr):
    return np.asarray(arr).view(SafeArray)

feature_names = [
    # IP特征 (27维)
    'ip_gap_mean','ip_gap_std','ip_small_gap_mean','ip_small_gap_std','ip_large_gap_mean',
    'ip_large_gap_std','cs_ip_gap_mean','cs_ip_gap_std','cs_ip_small_gap_mean','cs_ip_small_gap_std',
    'cs_ip_large_gap_mean','cs_ip_large_gap_std','sc_ip_gap_mean','sc_ip_gap_std','sc_ip_small_gap_mean',
    'sc_ip_small_gap_std','sc_ip_large_gap_mean','sc_ip_large_gap_std','ip_length_mean','ip_length_std',
    'cs_ip_length_mean','cs_ip_length_std','sc_ip_length_mean','sc_ip_length_std','ip_per_second_mean',
    'cs_ip_per_second_mean','sc_ip_per_second_mean',
    # 负载特征 (27维)
    'payload_gap_mean','payload_gap_std','payload_small_gap_mean','payload_small_gap_std','payload_large_gap_mean',
    'payload_large_gap_std','cs_payload_gap_mean','cs_payload_gap_std','cs_payload_small_gap_mean',
    'cs_payload_small_gap_std','cs_payload_large_gap_mean','cs_payload_large_gap_std','sc_payload_gap_mean',
    'sc_payload_gap_std','sc_payload_small_gap_mean','sc_payload_small_gap_std','sc_payload_large_gap_mean',
    'sc_payload_large_gap_std','payload_length_mean','payload_length_std','cs_payload_length_mean',
    'cs_payload_length_std','sc_payload_length_mean','sc_payload_length_std','payload_per_second_mean',
    'cs_payload_per_second_mean','sc_payload_per_second_mean',
    # PSH标志位特征 (3维)
    'psh_rate','cs_psh_rate','sc_psh_rate'
]

def get_ip_feature(flow):
    """提取IP包相关的统计特征
    
    Args:
        flow: 包含TCP流量数据的数组，shape为(n_packets, 4)
              第一列为时间戳，第二列为包大小，第三列为负载大小，第四列为PSH标志位
    
    Returns:
        list: 27维IP包统计特征向量
    """
    # 准备基础数据
    timestamps = as_safe_array(flow[:,0])  # 时间戳序列
    lengths = as_safe_array(flow[:,1])     # 包大小序列
    gaps = timestamps[1:] - timestamps[:-1] # 包间隔序列
    
    # 1. 计算所有包的时间间隔特征
    gap_mean = gaps.mean()
    gap_std = gaps.std()
    small_gaps = gaps[gaps <= gap_mean]
    large_gaps = gaps[gaps >= gap_mean]
    small_gap_mean = small_gaps.mean()
    small_gap_std = small_gaps.std()
    large_gap_mean = large_gaps.mean()
    large_gap_std = large_gaps.std()
    
    # 2. 计算上行包(C->S)的时间间隔特征
    cs_timestamps = timestamps[lengths > 0]
    cs_gaps = cs_timestamps[1:] - cs_timestamps[:-1] if len(cs_timestamps) > 1 else as_safe_array([])
    cs_gap_mean = cs_gaps.mean()
    cs_gap_std = cs_gaps.std()
    cs_small_gaps = cs_gaps[cs_gaps <= cs_gap_mean]
    cs_large_gaps = cs_gaps[cs_gaps >= cs_gap_mean]
    cs_small_gap_mean = cs_small_gaps.mean()
    cs_small_gap_std = cs_small_gaps.std()
    cs_large_gap_mean = cs_large_gaps.mean()
    cs_large_gap_std = cs_large_gaps.std()
    
    # 3. 计算下行包(S->C)的时间间隔特征
    sc_timestamps = timestamps[lengths < 0]
    sc_gaps = sc_timestamps[1:] - sc_timestamps[:-1] if len(sc_timestamps) > 1 else as_safe_array([])
    sc_gap_mean = sc_gaps.mean()
    sc_gap_std = sc_gaps.std()
    sc_small_gaps = sc_gaps[sc_gaps <= sc_gap_mean]
    sc_large_gaps = sc_gaps[sc_gaps >= sc_gap_mean]
    sc_small_gap_mean = sc_small_gaps.mean()
    sc_small_gap_std = sc_small_gaps.std()
    sc_large_gap_mean = sc_large_gaps.mean()
    sc_large_gap_std = sc_large_gaps.std()
    
    # 4. 计算包大小特征
    abs_lengths = abs(lengths)
    length_mean = abs_lengths.mean()
    length_std = abs_lengths.std()
    cs_lengths = lengths[lengths > 0]
    sc_lengths = abs(lengths[lengths < 0])
    cs_length_mean = cs_lengths.mean()
    cs_length_std = cs_lengths.std()
    sc_length_mean = sc_lengths.mean()
    sc_length_std = sc_lengths.std()
    
    # 5. 计算包速率特征
    duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    if duration > 0:
        ip_per_second_mean = len(lengths) / duration
        cs_per_second_mean = len(cs_lengths) / duration
        sc_per_second_mean = len(sc_lengths) / duration
    else:
        ip_per_second_mean = cs_per_second_mean = sc_per_second_mean = 0.0
    
    return [
        gap_mean, gap_std, small_gap_mean, small_gap_std, large_gap_mean, large_gap_std,
        cs_gap_mean, cs_gap_std, cs_small_gap_mean, cs_small_gap_std, cs_large_gap_mean, cs_large_gap_std,
        sc_gap_mean, sc_gap_std, sc_small_gap_mean, sc_small_gap_std, sc_large_gap_mean, sc_large_gap_std,
        length_mean, length_std, cs_length_mean, cs_length_std, sc_length_mean, sc_length_std,
        ip_per_second_mean, cs_per_second_mean, sc_per_second_mean
    ]

def get_payload_feature(flow):
    """提取负载相关的统计特征
    
    Args:
        flow: 包含TCP流量数据的数组，shape为(n_packets, 4)
              第一列为时间戳，第二列为包大小，第三列为负载大小，第四列为PSH标志位
    
    Returns:
        list: 27维负载统计特征向量
    """
    # 准备基础数据
    timestamps = as_safe_array(flow[:,0])   # 时间戳序列
    payloads = as_safe_array(flow[:,2])     # 负载大小序列
    payload_mask = payloads != 0            # 有负载的包的掩码
    payload_timestamps = timestamps[payload_mask]  # 有负载的包的时间戳
    payload_lengths = payloads[payload_mask]      # 有负载的包的负载大小
    
    # 1. 计算所有负载包的时间间隔特征
    gaps = payload_timestamps[1:] - payload_timestamps[:-1] if len(payload_timestamps) > 1 else as_safe_array([])
    gap_mean = gaps.mean()
    gap_std = gaps.std()
    small_gaps = gaps[gaps <= gap_mean]
    large_gaps = gaps[gaps >= gap_mean]
    small_gap_mean = small_gaps.mean()
    small_gap_std = small_gaps.std()
    large_gap_mean = large_gaps.mean()
    large_gap_std = large_gaps.std()
    
    # 2. 计算上行负载包(C->S)的时间间隔特征
    cs_timestamps = payload_timestamps[payload_lengths > 0]
    cs_gaps = cs_timestamps[1:] - cs_timestamps[:-1] if len(cs_timestamps) > 1 else as_safe_array([])
    cs_gap_mean = cs_gaps.mean()
    cs_gap_std = cs_gaps.std()
    cs_small_gaps = cs_gaps[cs_gaps <= cs_gap_mean]
    cs_large_gaps = cs_gaps[cs_gaps >= cs_gap_mean]
    cs_small_gap_mean = cs_small_gaps.mean()
    cs_small_gap_std = cs_small_gaps.std()
    cs_large_gap_mean = cs_large_gaps.mean()
    cs_large_gap_std = cs_large_gaps.std()
    
    # 3. 计算下行负载包(S->C)的时间间隔特征
    sc_timestamps = payload_timestamps[payload_lengths < 0]
    sc_gaps = sc_timestamps[1:] - sc_timestamps[:-1] if len(sc_timestamps) > 1 else as_safe_array([])
    sc_gap_mean = sc_gaps.mean()
    sc_gap_std = sc_gaps.std()
    sc_small_gaps = sc_gaps[sc_gaps <= sc_gap_mean]
    sc_large_gaps = sc_gaps[sc_gaps >= sc_gap_mean]
    sc_small_gap_mean = sc_small_gaps.mean()
    sc_small_gap_std = sc_small_gaps.std()
    sc_large_gap_mean = sc_large_gaps.mean()
    sc_large_gap_std = sc_large_gaps.std()
    
    # 4. 计算负载大小特征
    abs_lengths = abs(payload_lengths)
    length_mean = abs_lengths.mean()
    length_std = abs_lengths.std()
    cs_lengths = payload_lengths[payload_lengths > 0]
    sc_lengths = abs(payload_lengths[payload_lengths < 0])
    cs_length_mean = cs_lengths.mean()
    cs_length_std = cs_lengths.std()
    sc_length_mean = sc_lengths.mean()
    sc_length_std = sc_lengths.std()
    
    # 5. 计算负载包速率特征
    duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    if duration > 0:
        payload_per_second_mean = len(payload_lengths) / duration
        cs_per_second_mean = len(cs_lengths) / duration
        sc_per_second_mean = len(sc_lengths) / duration
    else:
        payload_per_second_mean = cs_per_second_mean = sc_per_second_mean = 0.0
    
    return [
        gap_mean, gap_std, small_gap_mean, small_gap_std, large_gap_mean, large_gap_std,
        cs_gap_mean, cs_gap_std, cs_small_gap_mean, cs_small_gap_std, cs_large_gap_mean, cs_large_gap_std,
        sc_gap_mean, sc_gap_std, sc_small_gap_mean, sc_small_gap_std, sc_large_gap_mean, sc_large_gap_std,
        length_mean, length_std, cs_length_mean, cs_length_std, sc_length_mean, sc_length_std,
        payload_per_second_mean, cs_per_second_mean, sc_per_second_mean
    ]

def get_psh_feature(flow):
    """提取PSH标志位相关的统计特征
    
    Args:
        flow: 包含TCP流量数据的数组，shape为(n_packets, 4)
              第一列为时间戳，第二列为包大小，第三列为负载大小，第四列为PSH标志位
    
    Returns:
        list: 3维PSH标志位统计特征向量
    """
    # 准备基础数据
    lengths = as_safe_array(flow[:,1])  # 包大小序列
    pshs = as_safe_array(flow[:,3])     # PSH标志位序列
    
    # 计算PSH标志位设置率
    psh_rate = np.abs(pshs).sum() / len(pshs) if len(pshs) > 0 else 0.0
    
    # 计算上行包的PSH标志位设置率
    cs_pshs = pshs[lengths > 0]
    cs_psh_rate = cs_pshs.sum() / len(cs_pshs) if len(cs_pshs) > 0 else 0.0
    
    # 计算下行包的PSH标志位设置率
    sc_pshs = pshs[lengths < 0]
    sc_psh_rate = np.abs(sc_pshs).sum() / len(sc_pshs) if len(sc_pshs) > 0 else 0.0
    
    return [psh_rate, cs_psh_rate, sc_psh_rate]

def flow_to_vector(flow):
    """从TCP流量数据中提取统计特征向量
    
    本函数从TCP流量数据中提取57维统计特征，包括：
    1. IP包相关特征(27维)：
       - 包间隔时间的统计特征
       - 包大小的统计特征
       - 包传输速率特征
    2. 负载相关特征(27维)：
       - 负载包间隔时间的统计特征
       - 负载大小的统计特征
       - 负载包传输速率特征
    3. PSH标志位特征(3维)：
       - 各方向PSH标志位的设置率
    
    Args:
        flow: 包含TCP流量数据的数组，shape为(n_packets, 4)
              第一列为时间戳，第二列为包大小，第三列为负载大小，第四列为PSH标志位
    
    Returns:
        numpy.ndarray: 57维统计特征向量
    """
    return np.array([
        *get_ip_feature(flow),      # 27维IP包特征
        *get_payload_feature(flow),  # 27维负载特征
        *get_psh_feature(flow)       # 3维PSH标志位特征
    ])