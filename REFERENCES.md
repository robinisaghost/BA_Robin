# References — devBranch-offset-loss-bounded-lag

All sources referenced in the source code of this branch, listed in
Vancouver style as required by PD Dr. Kaspar Riesen (Pattern Recognition
Group, University of Bern).

---

## Models

[1] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
    *Neural Computation*, 9(8), 1735–1780.
    https://doi.org/10.1162/NECO.1997.9.8.1735
    → Used in: `src/ba_baseline/models/lstm.py`, `scripts/train_lstm_*.py`

[2] Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time
    series is worth 64 words: Long-term forecasting with transformers. In
    *Proceedings of the Eleventh International Conference on Learning
    Representations (ICLR 2023)*.
    https://openreview.net/forum?id=Jbdc0vTOcol
    → Used in: `src/ba_baseline/models/patchtst.py`, `scripts/train_patchtst_*.py`

[3] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
    Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you
    need. In *Advances in Neural Information Processing Systems (NeurIPS 2017)*.
    → Used in: `src/ba_baseline/models/patchtst.py`
    (Transformer encoder architecture, via `nn.TransformerEncoder`)

[4] Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., & Choo, J. (2022).
    Reversible instance normalization for accurate time-series forecasting
    against distribution shift. In *Proceedings of the Tenth International
    Conference on Learning Representations (ICLR 2022)*.
    https://openreview.net/forum?id=cGDAkQo1C0p
    → Used in: `src/ba_baseline/models/patchtst.py`
    (RevIN per-window normalisation before encoder, denormalisation after head)

---

## Preprocessing and Data Pipeline

[5] Lim, B., & Zohren, S. (2020). Time series forecasting with deep
    learning: A survey. *Philosophical Transactions of the Royal Society A*,
    379(2194).
    https://doi.org/10.1098/rsta.2020.0209
    → Used in: `src/ba_baseline/data/multi_patient_dataset.py`
    (sliding-window / rolling prediction strategy)

[6] Nemat, H., Khadem, H., Elliott, J., & Benaissa, M. (2024).
    Data-driven blood glucose level prediction in type 1 diabetes:
    a comprehensive comparative analysis. *Scientific Reports*, 14(1), 21863.
    https://doi.org/10.1038/s41598-024-70277-x
    → Used in: `src/ba_baseline/data/multi_patient_dataset.py`
    (z-score normalisation in blood glucose forecasting)

---

## Evaluation Metrics

[7] van den Hoek, R. (2026). Mitigating Time-Shift Errors in CGM-based
    Glucose Forecasting and Hypoglycemia Event Prediction. Bachelor Thesis,
    University of Bern, Faculty of Science (INF).
    Supervisor: PD Dr. Kaspar Riesen.
    → Used in: `src/ba_baseline/metrics/metrics.py`, `src/ba_baseline/data/split.py`,
               `src/ba_baseline/losses/bounded_lag_loss.py`,
               `scripts/train_lstm_bounded_lag.py`,
               `scripts/train_patchtst_bounded_lag.py`

[8] Hüni, F. (2023). Predicting events of hypoglycemia: A comparison of long
    short-term memory and graph attention network based approaches. Bachelor
    Thesis, University of Bern, Faculty of Science (INF).
    Supervisor: PD Dr. Kaspar Riesen.
    → Used in: `src/ba_baseline/metrics/metrics.py`, `src/ba_baseline/data/split.py`,
               `src/ba_baseline/models/lstm.py`, `scripts/train_lstm_*.py`,
               `scripts/train_patchtst_*.py`

---

## Dataset

[9] Garcia-Tirado, J., Colmegna, P., Villard, O., Diaz, J. L.,
    Esquivel-Zuniga, R., Koravi, C. L. K., Barnett, C. L., Oliveri, M. C.,
    Fuller, M., Brown, S. A., DeBoer, M. D., & Breton, M. D. (2023).
    Assessment of meal anticipation for improving fully automated insulin
    delivery in adults with type 1 diabetes. *Diabetes Care*, 46(9), 1652–1658.
    https://doi.org/10.2337/dc23-0119
    → Used in: `src/ba_baseline/data/patient_loader.py`,
               `scripts/train_lstm_60min.py`, `scripts/train_patchtst_60min.py`

---

## Hyperparameter Optimisation

[10] Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019).
    Optuna: A next-generation hyperparameter optimization framework. In
    *Proceedings of the 25th ACM SIGKDD International Conference on
    Knowledge Discovery & Data Mining* (pp. 2623–2631).
    https://doi.org/10.1145/3292500.3330701
    → Used in: `scripts/train_lstm_60min.py`, `scripts/train_patchtst_60min.py`
    (Bayesian hyperparameter search via TPE sampler)

---

## Internal

[11] Pattern Recognition Group, University of Bern. Glucose Prediction
    Proposal. Internal unpublished manuscript.
    → Used in: `src/ba_baseline/models/lstm.py`,
               `src/ba_baseline/models/patchtst.py`,
               `src/ba_baseline/data/multi_patient_dataset.py`,
               `scripts/train_lstm_60min.py`,
               `scripts/train_patchtst_60min.py`,
               `scripts/train_lstm_bounded_lag.py`,
               `scripts/train_patchtst_bounded_lag.py`

---

## Alignment Loss (Bounded-Lag Objective)

[12] Le Guen, V., & Thome, N. (2019). Shape and time distortion loss for
    training deep time series forecasting models. In *Advances in Neural
    Information Processing Systems 32 (NeurIPS 2019)*.
    https://proceedings.neurips.cc/paper/2019/hash/466accbac9a66b805ba50e42ad715740-Abstract.html
    → Used in: `src/ba_baseline/losses/bounded_lag_loss.py`
    (motivation for temporal-alignment losses; DILATE is the closest prior
    work to bounded_lag_mse)

[13] Cuturi, M., & Blondel, M. (2017). Soft-DTW: a differentiable loss
    function for time-series. In *Proceedings of the 34th International
    Conference on Machine Learning (ICML 2017)*, vol. 70, pp. 894–903. PMLR.
    https://proceedings.mlr.press/v70/cuturi17a.html
    → Used in: `src/ba_baseline/losses/bounded_lag_loss.py`
    (differentiable alignment via DTW relaxation; bounded_lag_mse is a
    discrete, winner-takes-all variant of this idea)

[14] Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm
    optimization for spoken word recognition. *IEEE Transactions on
    Acoustics, Speech, and Signal Processing*, 26(1), 43–49.
    https://doi.org/10.1109/TASSP.1978.1163055
    → Used in: `src/ba_baseline/losses/bounded_lag_loss.py`
    (Sakoe-Chiba band constrains DTW to a bounded diagonal strip, which
    directly motivates the ±max_lag window in bounded_lag_mse)
