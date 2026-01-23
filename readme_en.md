# IFIP119-22nd-CryptoMining-Detection-Minimal-Demo
<div align="center">
    
[中文](README.md) | English

</div>

This project is a minimal system example of the cryptomining malware detection method from "Cryptomining Traffic Detection via Statistical Filtering and Deep Learning" (IFIP WG 11.9, 22nd), intended to facilitate related research. The project includes the following parts:
- Data Processing Module
    - Metadata Extraction Tool
    - Flow Tracking Module
    - Statistical Feature Extraction Module
- Detector Module
    - Statistical Filtering Model (LSSVM)
    - Precise Detection Model (TELSTM)
- Cryptomining Malware Data
    - Real cryptomining traffic samples (from MalwareBazaar)
    - Traffic captured during 10 minutes of running mining samples
    - CPU usage records during the running of mining samples
- Minimal System
    - Detection model weight files
    - Complete serial detection process

## Data Processing Module
Contains three parts: Metadata Extraction Module, Flow Tracking Module, and Statistical Feature Extraction Module.

### Metadata Extraction Module
- Function: Extract metadata required for cryptomining traffic detection from .pcap files and save to .csv files
- File: pcap2meta.cpp (Source Code), pcap2meta.exe (Pre-compiled Tool)
- Dependencies: npcap-sdk-1.13, wpcap.lib, packet.lib (Included in the project)
- Compilation: cl /EHsc /std:c++17 pcap2meta.cpp /I".\npcap-sdk-1.13\Include" /link /LIBPATH:".\npcap-sdk-1.13\Lib" wpcap.lib packet.lib
- Usage: pcap2meta.exe packet_1.pcap packet_2.pcap ... packet_n.pcap will generate metadata csv files in the directory where the tool is located.

### Flow Tracking Module
- Function: Process metadata csv files to generate TCP flows for detection
- File: tcpmeta.py
- Dependencies: requirements.txt (Included in the project)
- Dependency Installation: pip install -r requirements.txt
- Methods: load_meta_for_lstm_from_csv, load_meta_from_csv
- Usage: Refer to the minimal system implementation

### Statistical Feature Extraction Module
- Function: Extract 57-dimensional features required for statistical filtering from TCP flows
- Directory: extract_features (Source Code)
- Dependencies: requirements.txt (Included in the project)
- Installation: Execute "pip install --use-pep517 ." in extract_features directory
- Usage: Refer to the minimal system implementation (statistical_feature_extractor_cc)
- Alternative: statistical_feature_extractor.py equivalent python implementation

## Detector Module
Contains two parts: Statistical Filtering Model (LSSVM) and Precise Detection Model (TELSTM).

### Statistical Filtering Model (LSSVM)
- Function: Filter most non-mining TCP flows
- File: None, based on scikit-learn implementation
- Usage: Refer to the minimal system implementation

### Precise Detection Model (TELSTM)
- Function: Provide precise detection results for suspicious TCP flows
- Files:
    - time2vector.py: Timestamp embedding
    - simple_projector.py: Vectorization of other source data
    - detector.py: Detector
- Usage: Refer to the minimal system implementation

## Cryptomining Malware Data (DANGER!)
- Malware Samples: 01_sample/, unzip password "infected"
- CPU usage records during sample execution: 02_cpu_usage/
- Traffic captured during 10 minutes of running: 03_traffic/
- More samples please visit: https://bazaar.abuse.ch/browse/

## Minimal System
### Detection Model Weight Files
- LSSVM Model Weights: parameter2.npz
- TELSTM Model Weights: lstm_classifier.weight

### Complete Serial Detection Process
- traffic.pcap->metedada.csv：01_metadata_extract.ipynb
- metedada.csv->pool black list: 02_mining_detection.ipynb
- faster detection: 03_random_sample.ipynb
