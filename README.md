# Times2D

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
## Data
These datasets are commonly used for benchmarking time series forecasting models in academic research and competitions, covering domains like temperature, electricity, transportation, weather, and health. See the table below for details.

| Dataset      | Number of columns | Prediction Horizon  | Train/Validation/Test Size | Frequency | Domain       |
|--------------|-------------------|---------------------|----------------------------|-----------|--------------|
| ETTm1        | 7                 | {96, 192, 336, 720} | (34465, 11521, 11521)      | 15 min    | Electricity Transformer Temperature  |
| ETTm2        | 7                 | {96, 192, 336, 720} | (34465, 11521, 11521)      | 15 min    | Electricity Transformer Temperature  |
| ETTh1        | 7                 | {96, 192, 336, 720} | (8545, 2881, 2881)         | 1 hour    | Electricity Transformer Temperature  |
| ETTh2        | 7                 | {96, 192, 336, 720} | (8545, 2881, 2881)         | 1 hour    | Electricity Transformer Temperature  |
| Electricity  | 321               | {96, 192, 336, 720} | (18317, 2633, 5261)        | 1 hour    | Electricity Load Demand  |
| Traffic      | 862               | {96, 192, 336, 720} | (12185, 1757, 3509)        | 1 hour    | Transportation|
| Weather      | 21                | {96, 192, 336, 720} | (36792, 5271, 10540)       | 10 min    | Weather      |


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
sh ./scripts/Tims2D/longTerm/etth1.sh
sh ./scripts/Tims2D/ShortTerm/M4.sh
```

## Acknowledgements

This project makes use of code from the following open-source projects:

[TimesNet](https://github.com/thuml/Time-Series-Library) - A deep learning model for time series forecasting developed by THUML @ Tsinghua University, used under the MIT License.  
[PDF](https://github.com/Hank0626/PDF) - A framework licensed under the GNU Affero General Public License Version 3 (AGPLv3). For more details, see the full [AGPLv3 License](https://www.gnu.org/licenses/agpl-3.0.html).  
[Autoformer](https://github.com/thuml/Autoformer) - A model for long-term time series forecasting.  
[PatchTST](https://github.com/yuqinie98/PatchTST) - A Transformer model for multivariate time series forecasting.  
[Informer](https://github.com/zhouhaoyi/Informer2020) - An efficient transformer model for long sequence time-series forecasting.

We are grateful to the authors for their contributions to the open-source community.


