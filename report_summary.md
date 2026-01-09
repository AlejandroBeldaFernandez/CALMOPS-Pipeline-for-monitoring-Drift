# Data Quality & Comparison Report

**Generator:** SyntheticGenerator_Drifted_feature_drift
**Date:** 2026-01-09T10:28:19.199529

## 📜 Drift Injection History
### Injection #1
- **Method:** inject_feature_drift
- **Type:** add_value
- **Features:** age
- **Magnitude:** 0.2
- **Time:** 2026-01-08T13:12:24.266213

### Injection #2
- **Method:** inject_feature_drift
- **Type:** add_value
- **Features:** age
- **Magnitude:** 0.2
- **Time:** 2026-01-08T13:20:50.264710

### Injection #3
- **Method:** inject_feature_drift
- **Type:** add_value
- **Features:** age
- **Magnitude:** 0.2
- **Time:** 2026-01-08T13:28:04.619660

### Injection #4
- **Method:** inject_feature_drift
- **Type:** add_value
- **Features:** age
- **Magnitude:** 0.2
- **Time:** 2026-01-08T13:35:17.882588

### Injection #5
- **Method:** inject_feature_drift
- **Type:** add_value
- **Features:** age
- **Magnitude:** 0.2
- **Time:** 2026-01-08T13:42:30.292818

### Injection #6
- **Method:** inject_feature_drift
- **Type:** add_value
- **Features:** age
- **Magnitude:** 0.2
- **Time:** 2026-01-08T13:48:36.904477

### Injection #7
- **Method:** inject_feature_drift
- **Type:** add_value
- **Features:** age
- **Magnitude:** 0.2
- **Time:** 2026-01-08T14:05:43.856378

### Injection #8
- **Method:** inject_feature_drift
- **Type:** add_value
- **Features:** age
- **Magnitude:** 0.2
- **Time:** 2026-01-08T14:11:13.687653

### Injection #9
- **Method:** inject_feature_drift
- **Type:** add_value
- **Features:** age
- **Magnitude:** 0.2
- **Time:** 2026-01-09T10:23:12.887549

### Injection #10
- **Method:** inject_feature_drift
- **Type:** add_value
- **Features:** age
- **Magnitude:** 0.2
- **Time:** 2026-01-09T10:28:19.198080

## ⭐ Quality Score
- **Overall Quality:** 0.9755555555555555
- **Weighted Score:** 0.9755555555555555

## 📊 Dataset Statistics
| Metric | Real | Synthetic |
| :--- | :--- | :--- |
| Rows | 50 | 50 |
| Columns | 10 | 10 |
| Duplicates | 0 | 0 |

## 📉 Statistical Tests & Distribution Fit
| Column | Test | Statistic | P-Value | Real Dist | Synth Dist | Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| salary | Kolmogorov-Smirnov | 0.0000 | 1.0000 | norm | norm | Distributions are likely similar |
| commission | Kolmogorov-Smirnov | 0.0000 | 1.0000 | norm | norm | Distributions are likely similar |
| age | Kolmogorov-Smirnov | 0.4400 | 0.0001 | None | Poisson | Distributions are likely different |
| elevel | Kolmogorov-Smirnov | 0.0000 | 1.0000 | Poisson | Poisson | Distributions are likely similar |
| car | Kolmogorov-Smirnov | 0.0000 | 1.0000 | Poisson | Poisson | Distributions are likely similar |
| zipcode | Kolmogorov-Smirnov | 0.0000 | 1.0000 | Poisson | Poisson | Distributions are likely similar |
| hvalue | Kolmogorov-Smirnov | 0.0000 | 1.0000 | norm | norm | Distributions are likely similar |
| hyears | Kolmogorov-Smirnov | 0.0000 | 1.0000 | None | None | Distributions are likely similar |
| loan | Kolmogorov-Smirnov | 0.0000 | 1.0000 | lognorm | lognorm | Distributions are likely similar |
| target | Kolmogorov-Smirnov | 0.0000 | 1.0000 | Bernoulli | Bernoulli | Distributions are likely similar |

## 📉 Distribution Comparison
| Column | Real Dist | Synthetic Dist | Match |
| :--- | :--- | :--- | :--- |
| salary | Mean: 93074.78, Std: 41430.25 | Mean: 93074.78, Std: 41430.25 | ✅ |
| commission | Mean: 19733.44, Std: 28436.26 | Mean: 19733.44, Std: 28436.26 | ✅ |
| age | Mean: 49.28, Std: 18.14 | Mean: 69.28, Std: 18.14 | ❌ |
| elevel | Mean: 2.34, Std: 1.51 | Mean: 2.34, Std: 1.51 | ✅ |
| car | Mean: 11.16, Std: 5.38 | Mean: 11.16, Std: 5.38 | ✅ |
| zipcode | Mean: 3.76, Std: 2.65 | Mean: 3.76, Std: 2.65 | ✅ |
| hvalue | Mean: 400514.78, Std: 262356.19 | Mean: 400514.78, Std: 262356.19 | ✅ |
| hyears | Mean: 15.72, Std: 9.40 | Mean: 15.72, Std: 9.40 | ✅ |
| loan | Mean: 256686.48, Std: 159360.16 | Mean: 256686.48, Std: 159360.16 | ✅ |
| target | Mean: 0.68, Std: 0.47 | Mean: 0.68, Std: 0.47 | ✅ |

## 🖼️ Visualizations
### Dimensionality Reduction (UMAP/PCA)
![UMAP](umap_SyntheticGenerator_Drifted_feature_drift.png)

