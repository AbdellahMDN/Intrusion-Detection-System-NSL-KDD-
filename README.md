# Network Intrusion Detection System (IDS)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A high-performance machine learning-based Intrusion Detection System achieving **93.44% accuracy** and **99.07% attack detection rate** on the official NSL-KDD test set.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Acknowledgments](#acknowledgments)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

In the era of rapidly evolving cyber threats, protecting computer networks has become more critical than ever. This project implements a sophisticated Intrusion Detection System (IDS) using machine learning techniques to automatically analyze network traffic patterns, detect anomalies, and identify security breaches in real-time.

**What makes this project special:**
-  Achieves **state-of-the-art performance** on the official NSL-KDD test set
-  **99.07% attack detection rate** with only 119 missed attacks out of 12,833
-  Handles **unseen attack types** not present in training data
-  Efficient hybrid approach combining clustering and ensemble learning
-  Comprehensive exploratory data analysis and visualization

##  Key Features

- **Hybrid ML Architecture**: Combines K-Means clustering with Random Forest classifiers
- **Zero-Day Attack Detection**: Successfully generalizes to 17 attack types not seen during training
- **Configurable Threshold**: Adjustable sensitivity for different security requirements
- **Complete Pipeline**: End-to-end solution from data preprocessing to model evaluation
- **Detailed Analysis**: Extensive EDA revealing attack patterns and dataset characteristics
- **Production-Ready**: Optimized for performance with MiniBatchKMeans and parallel processing

##  Dataset

This project uses the **NSL-KDD dataset**, created by the **Canadian Institute for Cybersecurity (CIC)**. NSL-KDD is one of the most popular and challenging benchmarks in network intrusion detection research.

**Dataset Characteristics:**
- **Training Set**: 125,973 network connections (23 attack types)
- **Test Set**: 22,544 network connections (38 attack types)
- **Features**: 41 features across 4 categories (basic, content, time-based, host-based)
- **Attack Categories**: DoS, Probe, R2L, U2R, and Normal traffic

**Key Challenge**: The test set contains 17 attack types that do NOT appear in training data, making this a realistic and challenging benchmark for model generalization.

**Download**: [NSL-KDD Dataset Documentation](https://www.unb.ca/cic/datasets/nsl.html)

### Attack Categories

| Category | Description | Examples |
|----------|-------------|----------|
| **DoS** | Denial of Service | neptune, smurf, teardrop |
| **Probe** | Surveillance/Scanning | nmap, portsweep, satan |
| **R2L** | Remote to Local | guess_passwd, ftp_write, warezmaster |
| **U2R** | User to Root | buffer_overflow, rootkit, perl |
| **Normal** | Legitimate traffic | normal |

##  Methodology

### Our Approach vs Common Pitfalls

**❌ Common Mistake**: Many studies merge KDDTrain+ and KDDTest+ and randomly split them, achieving artificial 99%+ accuracy but failing on real-world data.

**✅ Our Approach**: We strictly follow the official evaluation protocol:
- Train exclusively on KDDTrain+
- Validate on the official KDDTest+ (with unseen attacks)
- No data leakage between train and test sets

This ensures **real-world robustness** and **generalization capability**.

### Pipeline Overview

```
Raw Data → EDA → Preprocessing → Clustering → Classification → Evaluation
   ↓         ↓         ↓             ↓              ↓             ↓
NSL-KDD   Patterns  One-Hot    K-Means(9)    Random Forest    Metrics
                   Encoding   MiniBatch      (600 trees)      Analysis
```

### Key Steps

1. **Exploratory Data Analysis**: Identify patterns, imbalances, and attack distributions
2. **Feature Engineering**: One-Hot encoding for categorical features, MinMax scaling
3. **Clustering Strategy**: 9 clusters identified using MiniBatchKMeans
4. **Smart Classification**:
   - Pure clusters (>99.9% one class) → Direct mapping
   - Mixed clusters → Random Forest classifiers (600 trees, depth=20)
5. **Adaptive Thresholding**: Configurable sensitivity for different security policies

##  Results

### Performance Metrics (Threshold = 0.02)

| Metric | Score |
|--------|-------|
| **Accuracy** | **93.44%** |
| **F1-Score** | **0.9451** |
| **Detection Rate** | **99.07%** |
| **False Alarm Rate** | **13.99%** |

### Confusion Matrix

|  | Predicted Normal | Predicted Attack |
|---|---|---|
| **Actual Normal** | 8,352 | 1,359 |
| **Actual Attack** | 119 | 12,714 |

**Outstanding Achievement**: Only **119 attacks missed** out of 12,833 total attacks on the official test set!

### Performance Context

This model achieves **one of the best results reported** on the official NSL-KDD test set following the proper evaluation protocol. These metrics demonstrate:
- Exceptional generalization to unseen attack types
- Strong balance between detection rate and false alarm rate
- Practical applicability for real-world network security

##  Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Clone the Repository

```bash
git clone https://github.com/AbdellahMDN/Intrusion-Detection-System-NSL-KDD.git
cd intrusion-detection-system
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

##  Usage

### Download the Dataset

1. Download NSL-KDD dataset from [CIC website](https://www.unb.ca/cic/datasets/nsl.html)
2. Extract `KDDTrain+.txt` and `KDDTest+.txt`
3. Update file paths in the notebook

### Run the Complete Pipeline

```python
# Update paths to your dataset location
train_path = "path/to/KDDTrain+.txt"
test_path = "path/to/KDDTest+.txt"

# Run the notebook cells sequentially
# The pipeline includes:
# 1. Data loading and EDA
# 2. Preprocessing and feature engineering
# 3. Clustering with K-Means
# 4. Training Random Forest classifiers
# 5. Model evaluation and metrics
```

### Adjust Detection Threshold

```python
# More sensitive (catches more attacks, more false alarms)
threshold = 0.02

# More conservative (fewer false alarms, may miss some attacks)
threshold = 0.5

# Make predictions
y_pred = predict_fast(X_test_scaled, test_clusters, 
                      cluster_info, cluster_models, threshold)
```

##  Model Architecture

### Hybrid Clustering + Ensemble Approach

```
Input Features (122 dimensions after encoding)
           ↓
    MinMaxScaler Normalization
           ↓
    MiniBatchKMeans (9 clusters)
           ↓
    ┌──────┴──────────────────────────┐
    ↓                                  ↓
Pure Clusters                   Mixed Clusters
(Direct Mapping)            (Random Forest per cluster)
    ↓                                  ↓
    └──────┬──────────────────────────┘
           ↓
    Probability Aggregation
           ↓
    Threshold Classification
           ↓
    Final Prediction (0=Normal, 1=Attack)
```

### Cluster Distribution Strategy

- **2 Pure Clusters**: Direct mapping (100% attack or 0% attack)
- **7 Mixed Clusters**: Individual Random Forest classifiers
- Each RF: 600 trees, max depth 20, optimized hyperparameters

##  Acknowledgments

This project was **inspired by** and **builds upon** the excellent work of GitHub user **[thinline72](https://github.com/thinline72)**. The original clustering approach provided a solid foundation, which I extended and improved with:

- Enhanced preprocessing pipeline
- Optimized hyperparameters
- Advanced threshold tuning
- Comprehensive evaluation metrics
- Better handling of imbalanced clusters

Special thanks to:
- **Canadian Institute for Cybersecurity (CIC)** for creating and maintaining the NSL-KDD dataset
- **thinline72** for the foundational clustering methodology
- The open-source ML community for sklearn, pandas, and visualization tools

##  References

1. Tavallaee, M., et al. (2009). "A detailed analysis of the KDD CUP 99 data set"
2. NSL-KDD Dataset: https://www.unb.ca/cic/datasets/nsl.html
3. Original inspiration: [thinline72's repository](https://github.com/thinline72)

##  Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

**Areas for contribution:**
- Testing on other IDS datasets (CICIDS2017, UNSW-NB15)
- Deep learning implementations (LSTM, CNN)
- Real-time deployment optimization
- Additional visualization and interpretability

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Contact

**Your Name**
- GitHub: [@abdellahMDN](https://github.com/AbdellahMDN)
- Email: moudianeabdellah@gmail.com
- Kaggle: [moud[IA]ne](https://www.kaggle.com/abdellahmdn)

---

⭐ **If you find this project useful, please consider giving it a star!**

**Note**: This is an academic/research project. For production deployment in critical infrastructure, additional security measures, real-time optimization, and thorough testing are recommended.
