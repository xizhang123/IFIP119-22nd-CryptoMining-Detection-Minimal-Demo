#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>


namespace py = pybind11;

// 安全的均值计算
double safe_mean(const std::vector<double>& arr) {
    if (arr.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& val : arr) sum += val;
    return sum / arr.size();
}

// 安全的标准差计算
double safe_std(const std::vector<double>& arr) {
    if (arr.empty()) return 0.0;
    double mean = safe_mean(arr);
    double sum_sq = 0.0;
    for (const auto& val : arr) {
        double diff = val - mean;
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / arr.size());
}

// 提取IP包相关的统计特征
std::vector<double> get_ip_feature(py::array_t<double> flow) {
    auto flow_buf = flow.unchecked<2>();
    std::vector<double> timestamps, lengths;
    std::vector<double> gaps, cs_gaps, sc_gaps;
    
    // 提取基础数据
    for (size_t i = 0; i < flow_buf.shape(0); ++i) {
        timestamps.push_back(flow_buf(i, 0));
        lengths.push_back(flow_buf(i, 1));
    }
    
    // 计算时间间隔
    for (size_t i = 1; i < timestamps.size(); ++i) {
        gaps.push_back(timestamps[i] - timestamps[i-1]);
    }
    
    // 计算统计特征
    double gap_mean = safe_mean(gaps);
    double gap_std = safe_std(gaps);
    
    std::vector<double> small_gaps, large_gaps;
    for (const auto& gap : gaps) {
        if (gap <= gap_mean) small_gaps.push_back(gap);
        if (gap >= gap_mean) large_gaps.push_back(gap);
    }
    
    // 提取C->S和S->C的包
    std::vector<double> cs_timestamps, sc_timestamps;
    for (size_t i = 0; i < lengths.size(); ++i) {
        if (lengths[i] > 0) cs_timestamps.push_back(timestamps[i]);
        if (lengths[i] < 0) sc_timestamps.push_back(timestamps[i]);
    }
    
    // 计算C->S的时间间隔
    for (size_t i = 1; i < cs_timestamps.size(); ++i) {
        cs_gaps.push_back(cs_timestamps[i] - cs_timestamps[i-1]);
    }
    
    double cs_gap_mean = safe_mean(cs_gaps);
    double cs_gap_std = safe_std(cs_gaps);
    
    std::vector<double> cs_small_gaps, cs_large_gaps;
    for (const auto& gap : cs_gaps) {
        if (gap <= cs_gap_mean) cs_small_gaps.push_back(gap);
        if (gap >= cs_gap_mean) cs_large_gaps.push_back(gap);
    }
    
    // 计算S->C的时间间隔
    for (size_t i = 1; i < sc_timestamps.size(); ++i) {
        sc_gaps.push_back(sc_timestamps[i] - sc_timestamps[i-1]);
    }
    
    double sc_gap_mean = safe_mean(sc_gaps);
    double sc_gap_std = safe_std(sc_gaps);
    
    std::vector<double> sc_small_gaps, sc_large_gaps;
    for (const auto& gap : sc_gaps) {
        if (gap <= sc_gap_mean) sc_small_gaps.push_back(gap);
        if (gap >= sc_gap_mean) sc_large_gaps.push_back(gap);
    }
    
    // 计算包大小特征
    std::vector<double> abs_lengths, cs_lengths, sc_lengths;
    for (const auto& len : lengths) {
        abs_lengths.push_back(std::abs(len));
        if (len > 0) cs_lengths.push_back(len);
        if (len < 0) sc_lengths.push_back(std::abs(len));
    }
    
    // 计算包速率特征
    double duration = timestamps.size() > 1 ? timestamps.back() - timestamps.front() : 0.0;
    double ip_per_second_mean = duration > 0 ? lengths.size() / duration : 0.0;
    double cs_per_second_mean = duration > 0 ? cs_lengths.size() / duration : 0.0;
    double sc_per_second_mean = duration > 0 ? sc_lengths.size() / duration : 0.0;
    
    return {
        safe_mean(gaps), safe_std(gaps),
        safe_mean(small_gaps), safe_std(small_gaps),
        safe_mean(large_gaps), safe_std(large_gaps),
        safe_mean(cs_gaps), safe_std(cs_gaps),
        safe_mean(cs_small_gaps), safe_std(cs_small_gaps),
        safe_mean(cs_large_gaps), safe_std(cs_large_gaps),
        safe_mean(sc_gaps), safe_std(sc_gaps),
        safe_mean(sc_small_gaps), safe_std(sc_small_gaps),
        safe_mean(sc_large_gaps), safe_std(sc_large_gaps),
        safe_mean(abs_lengths), safe_std(abs_lengths),
        safe_mean(cs_lengths), safe_std(cs_lengths),
        safe_mean(sc_lengths), safe_std(sc_lengths),
        ip_per_second_mean, cs_per_second_mean, sc_per_second_mean
    };
}

// 提取负载相关的统计特征
std::vector<double> get_payload_feature(py::array_t<double> flow) {
    auto flow_buf = flow.unchecked<2>();
    std::vector<double> timestamps, payloads;
    std::vector<double> payload_timestamps, payload_lengths;
    
    // 提取基础数据
    for (size_t i = 0; i < flow_buf.shape(0); ++i) {
        timestamps.push_back(flow_buf(i, 0));
        payloads.push_back(flow_buf(i, 2));
        if (payloads.back() != 0) {
            payload_timestamps.push_back(timestamps.back());
            payload_lengths.push_back(payloads.back());
        }
    }
    
    std::vector<double> gaps;
    for (size_t i = 1; i < payload_timestamps.size(); ++i) {
        gaps.push_back(payload_timestamps[i] - payload_timestamps[i-1]);
    }
    
    double gap_mean = safe_mean(gaps);
    double gap_std = safe_std(gaps);
    
    std::vector<double> small_gaps, large_gaps;
    for (const auto& gap : gaps) {
        if (gap <= gap_mean) small_gaps.push_back(gap);
        if (gap >= gap_mean) large_gaps.push_back(gap);
    }
    
    // 提取C->S和S->C的负载包
    std::vector<double> cs_timestamps, sc_timestamps;
    std::vector<double> cs_lengths, sc_lengths;
    for (size_t i = 0; i < payload_lengths.size(); ++i) {
        if (payload_lengths[i] > 0) {
            cs_timestamps.push_back(payload_timestamps[i]);
            cs_lengths.push_back(payload_lengths[i]);
        }
        if (payload_lengths[i] < 0) {
            sc_timestamps.push_back(payload_timestamps[i]);
            sc_lengths.push_back(std::abs(payload_lengths[i]));
        }
    }
    
    std::vector<double> cs_gaps;
    for (size_t i = 1; i < cs_timestamps.size(); ++i) {
        cs_gaps.push_back(cs_timestamps[i] - cs_timestamps[i-1]);
    }
    
    double cs_gap_mean = safe_mean(cs_gaps);
    double cs_gap_std = safe_std(cs_gaps);
    
    std::vector<double> cs_small_gaps, cs_large_gaps;
    for (const auto& gap : cs_gaps) {
        if (gap <= cs_gap_mean) cs_small_gaps.push_back(gap);
        if (gap >= cs_gap_mean) cs_large_gaps.push_back(gap);
    }
    
    std::vector<double> sc_gaps;
    for (size_t i = 1; i < sc_timestamps.size(); ++i) {
        sc_gaps.push_back(sc_timestamps[i] - sc_timestamps[i-1]);
    }
    
    double sc_gap_mean = safe_mean(sc_gaps);
    double sc_gap_std = safe_std(sc_gaps);
    
    std::vector<double> sc_small_gaps, sc_large_gaps;
    for (const auto& gap : sc_gaps) {
        if (gap <= sc_gap_mean) sc_small_gaps.push_back(gap);
        if (gap >= sc_gap_mean) sc_large_gaps.push_back(gap);
    }
    
    std::vector<double> abs_lengths;
    for (const auto& len : payload_lengths) {
        abs_lengths.push_back(std::abs(len));
    }
    
    // 计算负载包速率特征
    double duration = timestamps.size() > 1 ? timestamps.back() - timestamps.front() : 0.0;
    double payload_per_second_mean = duration > 0 ? payload_lengths.size() / duration : 0.0;
    double cs_per_second_mean = duration > 0 ? cs_lengths.size() / duration : 0.0;
    double sc_per_second_mean = duration > 0 ? sc_lengths.size() / duration : 0.0;
    
    return {
        safe_mean(gaps), safe_std(gaps),
        safe_mean(small_gaps), safe_std(small_gaps),
        safe_mean(large_gaps), safe_std(large_gaps),
        safe_mean(cs_gaps), safe_std(cs_gaps),
        safe_mean(cs_small_gaps), safe_std(cs_small_gaps),
        safe_mean(cs_large_gaps), safe_std(cs_large_gaps),
        safe_mean(sc_gaps), safe_std(sc_gaps),
        safe_mean(sc_small_gaps), safe_std(sc_small_gaps),
        safe_mean(sc_large_gaps), safe_std(sc_large_gaps),
        safe_mean(abs_lengths), safe_std(abs_lengths),
        safe_mean(cs_lengths), safe_std(cs_lengths),
        safe_mean(sc_lengths), safe_std(sc_lengths),
        payload_per_second_mean, cs_per_second_mean, sc_per_second_mean
    };
}

// 提取PSH标志位相关的统计特征
std::vector<double> get_psh_feature(py::array_t<double> flow) {
    auto flow_buf = flow.unchecked<2>();
    int total_packets = 0, cs_packets = 0, sc_packets = 0;
    int total_psh = 0, cs_psh = 0, sc_psh = 0;
    
    for (size_t i = 0; i < flow_buf.shape(0); ++i) {
        double length = flow_buf(i, 1);
        bool is_psh = flow_buf(i, 3) != 0;
        
        total_packets++;
        if (is_psh) total_psh++;
        
        if (length > 0) {
            cs_packets++;
            if (is_psh) cs_psh++;
        } else if (length < 0) {
            sc_packets++;
            if (is_psh) sc_psh++;
        }
    }
    
    double psh_rate = total_packets > 0 ? static_cast<double>(total_psh) / total_packets : 0.0;
    double cs_psh_rate = cs_packets > 0 ? static_cast<double>(cs_psh) / cs_packets : 0.0;
    double sc_psh_rate = sc_packets > 0 ? static_cast<double>(sc_psh) / sc_packets : 0.0;
    
    return {psh_rate, cs_psh_rate, sc_psh_rate};
}

// 将流量数据转换为特征向量
std::vector<double> flow_to_vector(py::array_t<double> flow) {
    std::vector<double> features;
    
    auto ip_features = get_ip_feature(flow);
    auto payload_features = get_payload_feature(flow);
    auto psh_features = get_psh_feature(flow);
    
    features.insert(features.end(), ip_features.begin(), ip_features.end());
    features.insert(features.end(), payload_features.begin(), payload_features.end());
    features.insert(features.end(), psh_features.begin(), psh_features.end());
    
    return features;
}

PYBIND11_MODULE(statistical_feature_extractor_cc, m) {
    m.doc() = "C++ implementation of flow feature extraction";
    m.def("flow_to_vector", &flow_to_vector, "Convert flow data to feature vector");
}