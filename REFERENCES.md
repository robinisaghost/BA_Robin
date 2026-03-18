# References — devBranch-offset-loss-dtw

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
    → Used in: `src/ba_baseline/models/patchtst.py` (positional encoding)

---

## Loss Functions

[4] Cuturi, M., & Blondel, M. (2017). Soft-DTW: a differentiable loss
    function for time-series. In *Proceedings of the 34th International
    Conference on Machine Learning (ICML 2017)*, pp. 894–903.
    http://proceedings.mlr.press/v70/cuturi17a.html
    → Used in: `src/ba_baseline/losses/soft_dtw_loss.py`,
               `scripts/train_lstm_dtw.py`, `scripts/train_patchtst_dtw.py`

[5] Berndt, D. J., & Clifford, J. (1994). Using dynamic time warping to find
    patterns in time series. In *Proceedings of the AAAI Workshop on Knowledge
    Discovery in Databases (KDD 1994)*, pp. 359–370.
    → Used in: `src/ba_baseline/losses/soft_dtw_loss.py` (original DTW)

---

## Evaluation Metrics

[6] van den Hoek, R. (2026). Mitigating Time-Shift Errors in CGM-based
    Glucose Forecasting and Hypoglycemia Event Prediction. Bachelor Thesis,
    University of Bern, Faculty of Science (INF).
    Supervisor: PD Dr. Kaspar Riesen.
    → Used in: `src/ba_baseline/metrics/metrics.py`, `src/ba_baseline/data/split.py`

[7] Hüni, F. (2023). Predicting events of hypoglycemia: A comparison of long
    short-term memory and graph attention network based approaches. Bachelor
    Thesis, University of Bern, Faculty of Science (INF).
    Supervisor: PD Dr. Kaspar Riesen.
    → Used in: `src/ba_baseline/metrics/metrics.py`, `src/ba_baseline/data/split.py`

---

## Dataset

[8] Garcia-Tirado, J., Colmegna, P., Villard, O., Diaz, J. L.,
    Esquivel-Zuniga, R., Koravi, C. L. K., Barnett, C. L., Oliveri, M. C.,
    Fuller, M., Brown, S. A., DeBoer, M. D., & Breton, M. D. (2023).
    Assessment of meal anticipation for improving fully automated insulin
    delivery in adults with type 1 diabetes. *Diabetes Care*, 46(9), 1652–1658.
    https://doi.org/10.2337/dc23-0119
    → Used in: `src/ba_baseline/data/patient_loader.py`

---

## Internal

[9] Pattern Recognition Group, University of Bern. Glucose Prediction
    Proposal. Internal unpublished manuscript.
    → Used in: `src/ba_baseline/models/lstm.py`,
               `src/ba_baseline/models/patchtst.py`,
               `scripts/train_lstm_dtw.py`, `scripts/train_patchtst_dtw.py`
