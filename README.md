# IFIP119-22nd-CryptoMining-Detection-Minimal-Demo
<div align="center">
中文 | [English](./readme_en.md)
</div>

该项目是"Cryptomining Traffic Detection via Statistical Filtering and Deep Learning" (IFIP WG 11.9, 22nd) 中挖矿木马检测方法最小系统样例，用于促进相关研究。项目包含以下部分：
- 数据处理模块
    - 元数据提取工具
    - 流追踪模块
    - 统计特征提取模块
- 检测器模块
    - 统计过滤模型（LSSVM）
    - 精确检测模型（TELSTM）
- 挖矿木马数据
    - 真实挖矿流量样本（来自MalwareBazaar）
    - 运行挖矿样本10分钟的捕获流量
    - 挖矿样本运行期间的CPU使用记录
- 最小系统
    - 检测模型的权重文件
    - 完整的串联检测流程

## 数据处理模块
包含三个部分:元数据提取模块、流追踪模块和统计特征提取模块。
### 元数据提取模块
- 功能：从.pcap文件中提取挖矿流量检测所需的元数据，保存到.csv文件中
- 文件：pcap2meta.cpp（源码），pcap2meta.exe（预编译工具）
- 依赖: npcap-sdk-1.13，wpcap.lib，packet.lib （已经包含在项目中）
- 编译：cl /EHsc /std:c++17 pcap2meta.cpp /I".\npcap-sdk-1.13\Include" /link /LIBPATH:".\npcap-sdk-1.13\Lib" wpcap.lib packet.lib
- 使用：pcap2meta.exe packet_1.pcap packet_2.pcap ... packet_n.pcap 会在工具自身所在的目录生成元数据csv文件。
### 流追踪模块
- 功能：处理元数据csv文件，生成待检测的TCP流
- 文件：tcpmeta.py
- 依赖：requirements.txt（已经包含在项目中）
- 依赖安装：pip install -r requirements.txt
- 方法：load_meta_for_lstm_from_csv，load_meta_from_csv
- 使用：参考最小系统实现
### 统计特征提取模块
- 功能：从TCP流中提取统计过滤所需的57维特征
- 目录：extract_features（源码）
- 依赖：requirements.txt（已经包含在项目中）
- 安装：在 extract_features 执行 "pip install --use-pep517 ."
- 使用：参考最小系统实现(statistical_feature_extractor_cc)
- 替代：statistical_feature_extractor.py 等价的python实现

## 检测器模块
包含两个部分:统计过滤模型（LSSVM）和精确检测模型（TELSTM）。
### 统计过滤模型（LSSVM）
- 功能：过滤大多数非挖矿TCP流
- 文件：无，基于scikit-learn实现
- 使用：参考最小系统实现
### 精确检测模型（TELSTM）
- 功能：给出可疑TCP流的精确检测结果
- 文件：
    - time2vector.py：时间戳嵌入
    - simple_projector.py：其余源数据向量化
    - detector.py：检测器
- 使用：参考最小系统实现
## 挖矿木马数据(危险！)
- 木马样本：01_sample/，解压密码"infected"
- 样本运行期间的CPU使用记录：02_cpu_usage/
- 运行10分钟捕获流量：03_traffic/
- 更多样本请访问：https://bazaar.abuse.ch/browse/

## 最小系统
### 检测模型的权重文件
- LSSVM模型权重：parameter2.npz
- TELSTM模型权重：lstm_classifier.weight
### 完整的串联检测流程
- pcap->metedada：01_metadata_extract.ipynb
- detection: 02_mining_detection.ipynb
