# Times2D
## 📄 Paper

Our paper describing Times2D has been accepted to **AAAI 2025** and is now available on arXiv:  
🔗 [Times2D: Multi-Period Decomposition and Derivative Mapping for General Time Series Forecasting](https://arxiv.org/abs/2504.00118)

## Overview

Times2D is a novel framework for time series forecasting that transforms 1D time series data into a 2D representation. This transformation enables the capture of intricate temporal variations such as multi-periodicity, sharp fluctuations, and turning points, which are challenging to model using traditional 1D methods. The model leverages advanced techniques, including Periodic Decomposition Block (PDB) and First and Second Derivative Heatmaps (FSDH), to efficiently forecast time series data across various domains.

## Table of Contents
- [Times2D Architecture](#Times2D-Architecture)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)


## Times2D Architecture
Times2D comprises three core components:

Periodic Decomposition Block (PDB): Uses Fast Fourier Transform (FFT) to decompose the time series into dominant periods, capturing both short-term and long-term dependencies.
First and Second Derivative Heatmaps (FSDH): Computes first and second derivatives of the time series to highlight sharp changes and turning points in the data.
Aggregation Forecasting Block (AFB): Aggregates the outputs of the PDB and FSDH, enabling accurate forecasting of time series data.

## 📊 Data
The following datasets were used in our **Times2D** experiments, covering a wide range of real-world domains such as energy systems, weather, transportation, health, and finance.

You can access all datasets here:
- 🌍 **Google Drive:** [https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2)  
- 🇨🇳 **Baidu Cloud:** [https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy)

**Datasets used:**
- **ETT (ETTh1, ETTh2, ETTm1, ETTm2)**
- **Exchange Rate**
- **Solar Energy**
- **National Illness**
- **Weather**
- **Traffic**
- **M4**
## Installation

To set up the environment and install the required packages, follow these steps:

1. **Clone the Repository:**

   First, clone this repository to your local machine.

2. **Install Required Packages:**

   It is recommended to create your own virtual environment and install the necessary packages in Python 3.10 as follows:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

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

## Acknowledgements

This project makes use of code from the following open-source projects:

[TimesNet](https://github.com/thuml/Time-Series-Library) - A deep learning model for time series forecasting developed by THUML @ Tsinghua University, used under the MIT License.  
[PDF](https://github.com/Hank0626/PDF) - A framework licensed under the GNU Affero General Public License Version 3 (AGPLv3). For more details, see the full [AGPLv3 License](https://www.gnu.org/licenses/agpl-3.0.html).  
[Autoformer](https://github.com/thuml/Autoformer) - A model for long-term time series forecasting.  
[PatchTST](https://github.com/yuqinie98/PatchTST) - A Transformer model for multivariate time series forecasting.  
[Informer](https://github.com/zhouhaoyi/Informer2020) - An efficient transformer model for long sequence time-series forecasting.

We are grateful to the authors for their contributions to the open-source community.


