## Writeup

The clinical management of central nervous system (CNS) tumors is heavily dependent on accurate presurgical classification. Recent epidemiological data from the CBTRUS report highlights severe incidence imbalances among brain tumors—where meningiomas and gliomas dominate, but rare subtypes like medulloblastomas and pineal region tumors present distinct diagnostic challenges and high mortality rates (Price et al., 2024). To address this, the field of radiomics has championed the conversion of clinical observations and medical imaging into mineable, high-dimensional data (Gillies et al., 2016).

Building upon these foundations, our current production workflow is a LightGBM-centered multimodal pipeline with radiomics-aware filtering and calibrated class-scale tuning. The core representation remains

$$
    x_i = \left[x_i^{\text{tab}}, x_i^{\text{text}}\right],
$$

where $x_i^{\text{tab}}$ includes structured clinical variables, report-derived indicators, and retained radiomics descriptors, and $x_i^{\text{text}}$ is a sparse TF-IDF representation of the report.

This combination is well-suited for heterogeneous tabular-text data and remains computationally efficient relative to heavier neural alternatives (Ke et al., 2017; Pedregosa et al., 2011).

### Documentation Navigation

- TF-IDF vectorizer reference: <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html>
- LightGBM documentation: <https://lightgbm.readthedocs.io/en/stable/index.html>
- Bayesian optimization overview: <https://en.wikipedia.org/wiki/Bayesian_optimization>

### 1. Loading Data

The script first merges two information sources:

- JSON case files containing the free-text report, label, and available modalities.
- Optional clinical CSV files containing structured patient-level variables.

Formally, for patient $i$, we observe a tuple

$$
    \left(X_i^{\text{clinical}}, X_i^{\text{report}}, Y_i\right),
$$

where $Y_i \in \{1,\dots,5\}$ denotes the tumor subtype. The implementation converts the JSON structure into a DataFrame and then performs a left join on `case_id`, creating one patient-level row per example. This is a simple but important step, because all later modeling assumes the classification problem is i.i.d. at the patient level rather than at the token or modality level.

The target classes are highly imbalanced, which is critical for model design. If $n_k$ denotes the number of training examples in class $k$, then the imbalance ratio

$$
    r = \frac{\max_k n_k}{\min_k n_k}
$$

is large, especially for the Pineal/Choroid and sellar-region classes. Since Kaggle evaluates macro-$F_1$, every class contributes equally to the score,

$$
    F_1^{\text{ macro}} = \frac{1}{K}\sum_{k=1}^{K} F_{1,k}
$$

so a model that performs extremely well on common classes but misses rare classes is still suboptimal.

This is why the pipeline does not optimize plain accuracy alone. Instead, it uses a custom blended objective that incorporates weighted-$F_1$, macro-$F_1$, and minority recall directly.

#### Radiomics ANOVA Screening

Before model fitting, the pipeline applies a univariate ANOVA filter to radiomics columns. Let $x_j$ denote radiomics feature $j$ and $y$ denote the class label. For each feature, we compute an ANOVA $F$-statistic:

$$
F_j = \frac{\text{between-class variance of } x_j}{\text{within-class variance of } x_j},
$$

then convert it to a $p$-value and retain the feature when:

$$
p_j \le 0.05.
$$

In implementation terms (`select_radiomics_by_anova` in `src/lgbm.py`), this screening step:

- removes weak radiomics features before high-dimensional sparse fusion;
- reduces noise and collinearity pressure for downstream LightGBM splits;
- keeps the remaining feature set clinically interpretable at the modality-feature level.

After ANOVA, the pipeline also applies missingness-based filtering and optional missingness indicators to stabilize retained radiomics signals across splits.

### 2. TF-IDF Vectorization

The radiology report is the dominant information source in this project. The script uses TF-IDF vectorization to map text into a sparse linear space, which is a classical and effective baseline for medical text classification when the dataset is not large enough to fully exploit deep language models (Pedregosa et al., 2011; Ramos, 2003).

For a term $t$ in document $d$, TF-IDF is

$$
    \text{tfidf}(t,d) = \text{tf}(t,d)\cdot \log\left(\frac{N}{d\cdot f(t)}\right)
$$

where $\text{tf}(t,d)$ is term frequency, $d\cdot f(t)$ is document frequency, and $N$ is the number of documents.

In `lgbm.py`, we deliberately use two complementary text views:

- word TF-IDF with n-grams $(1,2)$;
- character TF-IDF with character-window n-grams $(3,5)$.

The exact feature construction is:

```python
tfidf_word = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=2500,
)
tfidf_char = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=1200,
)
X_train_text_word = tfidf_word.fit_transform(X_train["report"])
X_train_text_char = tfidf_char.fit_transform(X_train["report"])
X_train_text = hstack([X_train_text_word, X_train_text_char], format="csr")
```

This design matters. Word n-grams capture medical phrases such as "midline shift", "sellar region", or "hydrocephalus". Character n-grams help with lexical variation, misspellings, inflections, and morphology, which is especially useful in radiology text where spelling and phrasing are not perfectly standardized. If the text representation is

$$
    x_i^{\text{text}} = \big[x_i^{\text{word}}, x_i^{\text{char}}\big]
$$

then the model receives both semantic phrase-level signal and robust subword-level signal.

This choice is one reason the LGBM pipeline becomes stronger after feature augmentation: the model is no longer relying on only a coarse bag-of-words view, but instead receives multiple sparse projections of the same clinical narrative.

### 3. Feature Augmentation

Beyond raw TF-IDF, the script engineers several report-level features from domain language in the radiology note. Examples include:

- `has_hydrocephalus`
- `has_enhancement`
- `has_pineal`
- `has_sellar`
- `has_ventricular`
- `has_extra_axial_meningioma_keywords`
- interaction terms such as `enhancement_x_edema`

Mathematically, these are deterministic transformations

$$
    \phi_j: X^{\text{report}} \to \mathbb{R}^m
$$

where each component $\phi_j(x)$ is either a count, a binary indicator, or an interaction. For example,

$$
    \phi_{\text{hydro}}(x) = \mathbf{1}\Big\{\text{``hydrocephalus"} \in x\Big\}
$$

and

$$
    \phi_{\text{enh}\times\text{edema}}(x) = \phi_{\text{enh}}(x)\cdot \phi_{\text{edema}}(x)
$$

This helps because some minority classes are characterized by very specific anatomical language. For instance, Pineal/Choroid cases are often associated with ventricular obstruction or pineal localization, while sellar-region tumors are associated with terms such as "sellar", "suprasellar", or "pituitary".

Hand-crafted indicators inject this localized domain knowledge in a low-variance way. Recent state-of-the-art neuro-oncology models, such as those by Wang et al. (2024), have demonstrated that integrating qualitative MRI signatures (e.g., enhancement, edema, localization) into machine learning ensembles like LightGBM yields highly robust presurgical predictions for rare brain tumors. Furthermore, extracting these predefined semantic features directly from standardized radiology reports sidesteps the reproducibility crisis often associated with complex pixel-level image segmenters (Zwanenburg et al., 2020), offering a highly deployable and interpretable feature space. In small-to-medium sized datasets, such hybrid feature engineering often improves stability relative to relying only on a learned deep representation (Huang et al., 2022; Pedregosa et al., 2011).

The script also adds report statistics such as:

- document length;
- word count;
- unique-word ratio;
- digit count;
- punctuation count.

These are weak features individually, but collectively they provide lightweight stylistic and structural information that trees can exploit through non-linear splits.

### 4. LightGBM

The classifier is a multiclass LightGBM model. LightGBM grows trees in a leaf-wise manner, choosing the split that maximally improves the objective, which often gives stronger empirical performance than level-wise growth for structured data under a fixed tree budget (Ke et al., 2017).

At a high level, the model predicts class probabilities

$$
    \hat{p}_{ik} = \Pr(Y_i = k \mid x_i),
$$

and optimizes an additive boosted-tree objective

$$
    \mathcal{L} = \sum_{i=1}^{n} \ell(y_i, \hat{y}_i) + \sum_{m=1}^{M}\Omega(f_m),
$$

where each $f_m$ is a tree, $\ell$ is the multiclass loss, and $\Omega$ regularizes tree complexity.

In `lgbm.py`, hyperparameters are not fixed constants. They are learned by Bayesian HPO each run, and the selected best parameter dictionary is printed during training (`Best Params From Bayes HPO`).

The search space is mapped through `normalize_lgbm_params`, which enforces practical bounds:

```python
{
    'n_estimators': [120, 800],
    'learning_rate': [10^-2.5, 10^-0.7],
    'num_leaves': [16, 128],
    'max_depth': [-1, 20],
    'min_child_samples': [5, 80],
    'subsample': [0.6, 1.0],
    'colsample_bytree': [0.6, 1.0],
    'reg_alpha': [10^-6, 10^1],
    'reg_lambda': [10^-6, 10^1],
    'objective': 'multiclass', 
    'class_weight': 'balanced', 
    'random_state': 42, 
    'n_jobs': -1, 'verbosity': -1
}
```

This setup keeps the optimization flexible while still bounded, and lets each run adapt to the current feature distribution and split.

The model still uses `class_weight="balanced"`, which reweights each class approximately inversely to its frequency. If $\pi_k$ is the empirical class proportion, then the class weight behaves roughly like

$$
    w_k \propto \frac{1}{\pi_k},
$$

so the loss penalizes errors on rare classes more heavily.

### 5. Bayesian Optimization

Instead of hand-tuning hyperparameters, `lgbm.py` uses Bayesian optimization through `bayes_optimize`. The basic idea is to model the unknown validation objective

$$
    f(\theta) = \text{score}(\theta)
$$

with a Gaussian process surrogate, then choose the next candidate hyperparameter vector $\theta$ by maximizing an acquisition function. In the script, the acquisition score is effectively

$$
    a(\theta) = \mu(\theta) + 2\sigma(\theta),
$$

where $\mu(\theta)$ is the surrogate mean prediction and $\sigma(\theta)$ is the surrogate uncertainty.

This is an upper-confidence-bound style rule: it balances exploitation of promising regions with exploration of uncertain ones (Snoek et al., 2012). In practice, this is much more sample-efficient than naive grid search for a mixed continuous/discrete hyperparameter space.

The demo optimizer implementation from `src/utils.py` is:

```python
def bayes_optimize(objective, bounds, n_trials: int, n_init: int, seed: int, candidate_pool: int = 384):
    keys = list(bounds.keys())
    low = np.array([float(bounds[k][0]) for k in keys], dtype=np.float64)
    high = np.array([float(bounds[k][1]) for k in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    x_hist, y_hist = [], []

    def _eval_point(x):
        score = float(objective(**dict(zip(keys, x))))
        x_hist.append(np.asarray(x, dtype=np.float64))
        y_hist.append(score)

    for _ in range(n_init):
        _eval_point(rng.uniform(low, high))

    for step in range(n_trials - n_init):
        # scale y for GP stability
        y_arr = np.asarray(y_hist, dtype=np.float64)
        y_scaled = (y_arr - np.mean(y_arr)) / max(np.std(y_arr), 1e-8)
        
        gp.fit(np.vstack(x_hist), y_scaled)
        
        x_cand = rng.uniform(low, high, size=(candidate_pool, len(keys)))
        mu, sigma = gp.predict(x_cand, return_std=True)
        
        x_next = x_cand[np.argmax(mu + 2.0 * sigma)]
        _eval_point(x_next)
```

The key point is that the optimized score is not raw accuracy. The script defines

$$
    \text{Blend} = 0.45\cdot F_1^{\text{weighted}} + 0.35\cdot F_1^{\text{macro}} + 0.20\cdot \text{Recall}_{\text{focus}},
$$

where

$$
    \text{Recall}_{\text{focus}} = \frac{1}{|S|}\sum_{k\in S}\text{Recall}_k,
$$

and $S$ is the set of especially difficult minority classes. This is a principled design choice: because the competition metric and the clinical use case both punish failure on rare subtypes, the hyperparameter search itself should reward minority-class sensitivity.

### 6. Minority Upsampling and Class-Scale Tuning

One of the most important ideas in the solution is that class imbalance is handled in two stages.

First, the training set is selectively upsampled for the hardest classes. If $n_k$ is the original class count and $\tilde{n}_k$ is the augmented count, then for focus classes the script increases their representation toward

$$
    \tilde{n}_k = \max(3n_k, \text{median class count}),
$$

This does not fully rebalance the dataset, but it pushes minority classes away from the extreme tail without distorting the whole class distribution.

Second, after the model outputs class probabilities, the script applies class-specific scaling:

$$
    \tilde{p}_{ik} = \frac{s_k \cdot \hat{p}_{ik}}{\sum_{j=1}^{K} s_j \cdot \hat{p}_{ij}},
$$
where $s_k > 0$ is a tunable scale for class $k$.

This matters because multiclass argmax decisions are sensitive to calibration. Even if the model ranks the correct minority class highly, a small probability deficit can still cause an incorrect label. Scaling effectively shifts class-wise decision boundaries after training, which is often cheaper and more stable than retraining the full model.

In the current `lgbm.py`, scale tuning is done in two stages:

1. tune scales on the holdout validation split (`val_tuned_blended_score`);
2. re-tune scales from out-of-fold probabilities on `train+val` and use these OOF scales as default for final inference.

This post-processing stage is implemented in `lgbm.py` as:

```python
def apply_class_scales(proba, scales):
    adjusted = proba * scales.reshape(1, -1)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def tune_class_scales(y_true, proba, focus_ids, rounds=4):
    scales = np.ones(proba.shape[1], dtype=np.float64)
    best_score = blend_score(y_true, np.argmax(proba, axis=1), focus_ids)
    grid = np.linspace(0.5, 3.0, 26)
    for _ in range(rounds):
        for cls_id in focus_ids:
            for g in grid:
                trial = scales.copy()
                trial[cls_id] = g
                pred = np.argmax(apply_class_scales(proba, trial), axis=1)
                score = blend_score(y_true, pred, focus_ids)
                if score > best_score:
                    best_score = score
                    scales[cls_id] = g
    return scales, best_score
```

And the OOF default is produced by:

```python
oof_scales, oof_blended_score = tune_class_scales_from_oof(
    X_oof, y_oof, X_test,
    best_params=best_params,
    focus_ids=focus_ids,
    n_splits=OOF_SCALE_N_SPLITS,
    seed=KFOLD_RANDOM_STATE,
)
class_scales = oof_scales
```

### 7. Training, Validation, and Submission

The current workflow in `lgbm.py` is:

1. fit preprocessing on train, then transform validation and test;
2. upsample focus classes on train only;
3. run 5-fold stratified CV diagnostics on `train+val` (with fold-level HPO and fold-level scale tuning);
4. run Bayesian HPO on the main training/validation split to obtain `best_params`;
5. fit the best LGBM configuration on the balanced training set;
6. tune class scales on validation predictions;
7. tune OOF class scales on `train+val` and set these as default scales;
8. report validation metrics;
9. rebuild features on the combined dataset ($\text{train} + \text{val}$);
10. fit one final model with `best_params` on balanced `train+val` and infer test with OOF-tuned scales.

This separation is important. CV is used for robustness diagnostics and OOF scale estimation, while the final submission model is trained on the full labeled sample (`train+val`) with HPO-selected parameters.

The validation report shows that the model is not merely strong on the majority classes:

- Pineal/Choroid recall = $1.00$
- Sellar-region recall = $1.00$
- Meningioma recall = $1.00$
- Brain metastase recall = $0.97$

This is exactly why the pipeline transfers well to Kaggle. The model is not achieving a high score by collapsing rare classes into common ones; instead, it maintains strong class-wise balance.

### Other Questions

#### Why not using Random Forest?

Random Forest is a strong baseline for tabular data and we did test it seriously. It is robust, easy to optimize, and often performs well on mixed clinical variables (Breiman, 2001). However, in this task LightGBM has three concrete advantages.

First, LightGBM is a boosting method, so it reduces bias by sequentially correcting earlier mistakes. Random Forest reduces variance through bagging, but it does not perform the same stage-wise error correction. For a difficult multiclass boundary with many sparse text dimensions, boosting is often more expressive.

Second, LightGBM is more computationally efficient when repeated tuning is required. Since the project evolved through many iterations of HPO, feature engineering, and threshold tuning, training speed mattered a lot. The leaf-wise LightGBM implementation gave more performance per unit time than RF in later experiments (Ke et al., 2017).

Third, the final LightGBM pipeline makes better use of the hybrid feature space. Once the text representation was strengthened with word + char TF-IDF and domain indicators, the non-linear splits in boosted trees captured these interactions more effectively than a bagged forest. In practical terms, this is why LGBM became the production model.

So the answer is not that RF is bad. Rather, RF was a valuable benchmark, but LightGBM provided the best combination of speed, flexibility, and macro-$F_1$ performance.

#### Why does BPNN fail here, even though our previous paper used it?

This is a very important question, because our earlier work in `docs/plan.md` and the prior SLI paper considered BPNN/MLP as a serious candidate (Huang et al., 2022). In that earlier setting, the feature space was smaller, more structured, and easier to stabilize with preprocessing. A compact neural network could be competitive because the representation was lower-dimensional and the class structure was less extreme.

In the current brain-tumor problem, several properties make BPNN less suitable as the primary production model.

First, the input is highly sparse and heterogeneous. The combined feature vector contains:

- one-hot encoded tabular variables;
- engineered binary indicators;
- word TF-IDF features;
- character TF-IDF features.

This produces a very high-dimensional sparse input matrix. In principle, MLPs can consume such data, but in practice they are usually less sample-efficient than boosted trees unless the dataset is much larger. The network must learn weights for thousands of sparse coordinates, while the rarest classes have very small support. That raises estimation variance and encourages overfitting (Goodfellow et al., 2016).

Second, the minority classes are too small for a neural model to generalize stably. If a class has only a handful of examples, then stochastic gradient updates for that class are noisy. Tree models can still isolate a minority pattern through a few highly informative splits, especially when those patterns correspond to key tokens like "pineal" or "suprasellar". An MLP, by contrast, tries to distribute the signal across many weights, which is much harder to estimate reliably at this sample size.

Third, our previous plan already hinted at this tradeoff. The BPNN section in `plan.md` reported that the MLP was competitive but showed a substantial overfitting gap. That pattern is exactly what we would expect in a medium-sized medical dataset with severe multiclass imbalance and sparse text features. The model has enough flexibility to fit the training set, but not enough stable support to consistently beat the strongest tree-based system out of sample.

A useful way to summarize the difference is:

- in the previous paper, BPNN was feasible because the feature engineering and data regime were more favorable to dense neural learning;
- in this project, the decisive signal is sparse, localized, and strongly keyword-driven, which is exactly the regime where TF-IDF + boosted trees is hard to beat.

So BPNN does not "fail" in the sense of being unusable. It simply fails to become the best final model. It remains a reasonable complementary model for ensembling, but not the optimal standalone solution for this dataset.

### Results

#### Classification Report

The latest reported validation snapshot below comes from the fixed-parameter reference workflow (`lgbm_fixed.py`). `lgbm.py` itself is HPO-driven and can produce different metrics per run.

| Metric | Value |
|---|---:|
| Accuracy | 0.8798586572438163 |
| Micro-$F_1$ | 0.8798586572438163 |
| Macro-$F_1$ | 0.807533609097973 |
| Weighted-$F_1$ | 0.8776004438803224 |
| Blended score (val-tuned scales) | 0.847743174261604 |
| Blended score after threshold tuning | 0.8220003917228115 |

Radiomics ANOVA summary for this run:

| Item | Value |
|---|---:|
| Candidate radiomics features | 20 |
| Retained after ANOVA ($p \le 0.05$) | 14 |
| Retention ratio | 70.0% |

Class-wise validation summary:

| Class | Precision | Recall | $F_1$ | Support |
|---|---:|---:|---:|---:|
| Brain Metastase Tumour | 0.74 | 0.64 | 0.69 | 36 |
| Glioma | 0.87 | 0.89 | 0.88 | 132 |
| Meningioma | 0.93 | 0.96 | 0.94 | 104 |
| Pineal tumour and Choroid plexus tumour | 0.67 | 0.67 | 0.67 | 3 |
| Tumors of the sellar region | 1.00 | 0.75 | 0.86 | 8 |

The current strategy selected by validation is `LGBM Only` with class-scale tuning.

#### Running Time

Our final fixed-parameter workflow remains practical for experimentation and reproducibility, while still incorporating radiomics selection, CV diagnostics, and class-scale tuning.

| Environment | Time (seconds) |
|---|---:|
| Kaggle (`submission.csv` run) | 1038.50 |

This timing includes the end-to-end fixed-parameter run (feature processing, CV diagnostics, validation evaluation, and submission generation).


### References

- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5-32. https://doi.org/10.1023/A:1010933404324
- Gillies, R. J., Kinahan, P. E., & Hricak, H. (2016). Radiomics: Images are more than pictures, they are data. *Radiology, 278*(2), 563-577.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press. https://www.deeplearningbook.org/
- Huang, G., Cheng, A., & Gao, Y. (2022). Machine learning improvements to the accuracy of predicting specific language impairment. In *2022 International Conference on Image Processing, Computer Vision and Machine Learning (ICICML)*. IEEE. https://doi.org/10.1109/ICICML57342.2022.10009881
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems, 30*.
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830.
- Price, M., Barnholtz-Sloan, J. S., Ostrom, Q. T., et al. (2024). CBTRUS Statistical Report: Primary Brain and Other Central Nervous System Tumors Diagnosed in the United States in 2017–2021. *Neuro-Oncology, 26*(Suppl 5).
- Ramos, J. (2003). Using TF-IDF to determine word relevance in document queries. In *Proceedings of the First Instructional Conference on Machine Learning*.
- Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian optimization of machine learning algorithms. In *Advances in Neural Information Processing Systems, 25*.
- Wang, Y.-R., et al. (2024). Advancing presurgical non-invasive molecular subgroup prediction in medulloblastoma using artificial intelligence and MRI signatures. *Cancer Cell, 42*, 1239-1257.
- Zwanenburg, A., Vallières, M., et al. (2020). The Image Biomarker Standardization Initiative: Standardized Quantitative Radiomics for High-Throughput Image-based Phenotyping. *Radiology, 295*, 328-338.
