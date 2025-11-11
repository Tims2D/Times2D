# Times2D

## 📄 Paper
Our paper describing Times2D has been accepted to **AAAI 2025** and is now available on arXiv:  
🔗 [Times2D: Multi-Period Decomposition and Derivative Mapping for General Time Series Forecasting](https://arxiv.org/abs/2504.00118)

---

## 🧠 Overview
**Times2D** is a novel framework for time series forecasting that transforms 1D time series data into a 2D representation.  
This transformation enables the model to capture intricate temporal variations — such as multi-periodicity, sharp fluctuations, and turning points — which are challenging to model using traditional 1D approaches.

The model leverages three key modules:
- **Periodic Decomposition Block (PDB):** Decomposes the time series via FFT to capture both short- and long-term periodic components.  
- **First & Second Derivative Heatmaps (FSDH):** Highlights local trends, peaks, and sharp transitions.  
- **Aggregation Forecasting Block (AFB):** Combines outputs from multiple 2D features for robust and efficient forecasting.

---

## 📂 Table of Contents
- [Architecture](#architecture)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

---

## 🏗️ Architecture
The Times2D framework integrates signal decomposition, derivative encoding, and efficient aggregation into a unified model for general-purpose forecasting.

**Key features:**
- Multi-period decomposition via FFT  
- Dynamic 2D embedding through derivative mapping  
- Shared convolutional feature extraction across time-frequency axes  
- Compatibility with diverse datasets and forecasting horizons

---

## 📊 Data
These datasets are commonly used for benchmarking time series forecasting models across domains such as temperature, electricity, transportation, weather, and health.

| Dataset      | Columns | Prediction Horizons | Train/Val/Test | Frequency | Domain |
|---------------|----------|---------------------|----------------|------------|--------|
| **ETTm1** | 7 | {96,192,336,720} | (34465,11521,11521) | 15 min | Transformer Temperature |
| **ETTm2** | 7 | {96,192,336,720} | (34465,11521,11521) | 15 min | Transformer Temperature |
| **ETTh1** | 7 | {96,192,336,720} | (8545,2881,2881) | 1 hour | Transformer Temperature |
| **ETTh2** | 7 | {96,192,336,720} | (8545,2881,2881) | 1 hour | Transformer Temperature |
| **Electricity** | 321 | {96,192,336,720} | (18317,2633,5261) | 1 hour | Load Demand |
| **Traffic** | 862 | {96,192,336,720} | (12185,1757,3509) | 1 hour | Transportation |
| **Weather** | 21 | {96,192,336,720} | (36792,5271,10540) | 10 min | Meteorological |
| **National Illness** | 7 | {24,36,48,60} | (616,77,52) | 1 week | Health |
| **Exchange Rate** | 8 | {96,192,336,720} | (7588,1517,1517) | 1 day | Finance |
| **Solar Energy** | 137 | {96,192,336,720} | (36888,5256,10512) | 10 min | Energy |

---

## ⚙️ Installation
To set up the environment and install the required packages, follow these steps:

```bash
git clone https://github.com/Tims2D/Times2D.git
cd Times2D
pip install -r requirements.txt
```

---

## 🚀 Usage
To run the models, navigate to the `scripts` folder, pick the intended `.sh` file, and execute it using the following commands:

```bash
#### 🔹 Long-Term Forecasting

# ETT Datasets
sh ./scripts/Times2D/longTerm/Times2D_ETTh1.sh
sh ./scripts/Times2D/longTerm/Times2D_ETTh2.sh
sh ./scripts/Times2D/longTerm/Times2D_ETTm1.sh
sh ./scripts/Times2D/longTerm/Times2D_ETTm2.sh

# Exchange Rate
sh ./scripts/Times2D/longTerm/Times2D_exchange_rate.sh

# Solar Energy
sh ./scripts/Times2D/longTerm/Times2D_solar.sh

# National Illness
sh ./scripts/Times2D/longTerm/Times2D_national_illness.sh

# Weather
sh ./scripts/Times2D/longTerm/Times2D_weather.sh

# Traffic
sh ./scripts/Times2D/longTerm/Times2D_traffic.sh


#### 🔹 Short-Term Forecasting

sh ./scripts/Times2D/ShortTerm/M4.sh
```

---

## 🙏 Acknowledgements
This project makes use of code from the following open-source projects:

[TimesNet](https://github.com/thuml/Time-Series-Library) - A deep learning model for time series forecasting developed by THUML @ Tsinghua University, used under the MIT License.  
[PDF](https://github.com/Hank0626/PDF) - A framework licensed under the GNU Affero General Public License Version 3 (AGPLv3). For more details, see the full [AGPLv3 License](https://www.gnu.org/licenses/agpl-3.0.html).  
[Autoformer](https://github.com/thuml/Autoformer) - A model for long-term time series forecasting.  
[PatchTST](https://github.com/yuqinie98/PatchTST) - A Transformer model for multivariate time series forecasting.  
[Informer](https://github.com/zhouhaoyi/Informer2020) - An efficient transformer model for long sequence time-series forecasting.

We are grateful to the authors for their contributions to the open-source community.
