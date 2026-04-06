# Group Project Call for Proposal: Multimodal Presurgical Brain Tumor Classification

**STAT3612: Statistical Machine Learning (Spring 2026)** — due: April 5, 2026, Sunday, 11:59 PM

---

## 1 Objective

In this group project, your task is to develop statistical machine learning models for presurgical brain tumor classification using multimodal data (including medical imaging, text information, demography information). Please note that you have the flexibility to choose one or some of these modalities for tumor classification in your project. You have the freedom to choose feature engineering techniques and machine learning algorithms without any restrictions. It's important to note that additional training data is not permitted for this project. The performance of your algorithm will be evaluated through a Kaggle competition. Please participate in the competition and submit your results on:

https://www.kaggle.com/competitions/2026-spring-sdst-stat-3612-group-project

---

## 2 Background

Accurate presurgical diagnosis of brain tumors is essential for treatment planning and patient management. Brain tumors vary substantially in their biological behavior, prognosis, and recommended treatment strategies [1]. For example, some tumors are typically managed with maximal safe surgical resection, whereas others may be treated primarily with chemotherapy and/or radiotherapy. Therefore, a timely and reliable preoperative tumor classification can play an important role in guiding clinical decision-making.

Magnetic Resonance Imaging (MRI) is the primary imaging modality used for the diagnosis and monitoring of brain tumors [2]. In routine clinical practice, radiologists interpret imaging findings together with patient information and summarize their assessment in radiology reports. However, manual interpretation can be challenging due to overlapping imaging characteristics across tumor types, inter-observer variability, and the complexity of multimodal clinical data. In this project, students will develop statistical machine learning models for presurgical brain tumor classification using multimodal data, including imaging features, radiology reports, and clinical/demographic information.

---

## 3 Dataset Description

### 3.1 Overall

In this project, your task is to develop statistical machine learning models for presurgical brain tumor classification using multimodal data. We consider a five-class tumor classification task, including brain metastase tumour, choroid plexus tumour, glioma, meningioma, a combination class consisting of pineal tumour and tumors of the sellar region.

The dataset used in this project is a curated dataset that includes both non-public/private data and data collected from publicly available online sources. Overall, the dataset consists of 2,838 patients. For each patient, we provide corresponding image features extracted from MRI scans, tumor class labels, radiology reports, and clinical information.

For the course project, the released dataset is divided into training, validation, and test sets, containing 1,983, 283, and 572 patients, respectively. More details about the released files can be found on the Kaggle competition page.

### 3.2 Available Data Modalities

More specifically, the provided data include the following modalities:

**1) Magnetic Resonance Imaging (MRI)**

For each patient, we provide multiple MRI modalities, including T1, T1-contrast, T2, and T2-FLAIR. Originally, the MRI data are volumetric 3D scans. To simplify the project, reduce computational burden, and address privacy concerns, the original MRI scans are not released. Instead, representative tumor-containing 2D slices were selected for each available modality and used to extract imaging features. Only the extracted features are provided in the released dataset.

The released MRI-derived features include:

- **a. Radiomic features:** For each MRI slice, we extract 5 types of radiomics features using PyRadiomics (https://pyradiomics.readthedocs.io/en/latest/). The data is stored at the folder `radiomics_info/`.
- **b. Deep image features:** Deep image features extracted from ResNet [3]. The extracted image features are stored at the folder `image_features/`.

**2)** Patient demographics (age and gender), which are saved at the folder `clinical_information/`.

**3)** Radiology report which describes the key findings from the MRI. We provide the radiology report associated with each patient. Importantly, only the findings section is included, while the impression section is excluded. This helps reduce direct diagnostic label leakage and encourage students to extract useful information from descriptive clinical text. These raw radiology reports are stored at the folder `original_raw_report/`.

To facilitate the use of radiology reports, we have additionally provided five types of clinical information extracted from the radiology report. This includes tumor location, signal intensity description from T1, T1c, T2, and FLAIR, respectively. Students are also encouraged to explore other approaches for extracting clinically meaningful information from the radiology reports.

These extracted report-derived features, together with patient demographic information, are stored in folder `clinical_information/`.

---

## 4 File Organization on Kaggle

Overall, the details of provided data are organized as follows (Please find more description of the dataset in the Kaggle):

1. `train.json`: Contains training data consisting of ID, image path, and tumor classification labels, etc.
2. `valid.json`: Contains validation data consisting of ID, image path, and tumor classification labels, etc.
3. `test.json`: Contains test data consisting of ID, image path, etc.
4. `image_features`: This archive holds features extracted from ResNet.
5. `original_raw_report`: Contains the corresponding radiology report.
6. `clinical_information`: Contains the patient demographics and extracted clinical information from radiology report.
7. `radiomics_info`: The extracted radiomics imaging features and the clinical information.
8. `sample_submission.csv`: This file serves as the template for your predictions.

**Evaluation Metrics.** We use F1 metric to evaluate the performance of your algorithms in the Kaggle. In your final report, you are strongly encouraged to include additional evaluation metrics such as AUC, precision, recall, and other relevant measures. You are also encouraged to report class-wise performance across different tumor types and analyze the contribution of different features or modalities to your method.

---

## 5 Grading Criteria

The grade of your group project will be based on these four aspects:

1. **Proposal (10%):** The soundness and difficulty of proposed methods.
2. **Performance (30%):** Test performance of your best algorithms.
3. **Workload and model analysis (40%):** The analysis part includes but is not limited to the comparison of different models, interpretability analysis of proposed models, the influence of different modalities, and class imbalance in the dataset.
4. **Story-telling (20%):** The story-telling of your oral presentation (on-site) and written report.

---

## 6 Proposal Format

There is no specific format for the proposal. It is recommended to be 1–2 pages and should outline what you propose to do and a rough plan for how you will pursue the project. For example, you can describe your plan for adopted methods and model analysis, the time schedule, the work distribution, and the potential resources you will use. You can also cite references or useful resources.

---

## 7 Submission

All submissions are in groups as a unit. Your total submission includes these four aspects:

1. A PDF file of your project proposal (on Moodle) **(DUE: April 5, 2026, Sunday, 11:59 PM)**.
2. Submit your predictions (on Kaggle) **(DUE: The oral presentation day on April 24, 2026, 8:00 AM)**.
3. Your slides of oral presentation (on Moodle) **(DUE: The oral presentation day on April 24, 2026, 8:00 AM)**.
4. Your code and written report (on Moodle). Please submit your best model in Jupyter Notebook format with adequate description, so that your Kaggle submission results can be reproduced by tutors.

### Instructions of Submission on Kaggle

- Register a Kaggle account on https://www.kaggle.com/.
- Join the competition. The links are in Section Objective.
- Submit your results on Kaggle. Your submission file should have the same format as the `sample_submission.csv` in Kaggle.

---

## 8 Frequently Asked Questions

**Can I use Deep Neural Network?**
Yes. It can be used. However, it's important to note that their performance may not always be optimal.

**Should I use both modality data?**
No. You have the flexibility to choose one or both modalities to use. You can use pure radiomics imaging features, or deep imaging features or clinical information. You can also experience the power of multimodal data by using all the available information.

**Can I use additional data beyond the released dataset?**
No. You may only use the data provided through the Kaggle competition. Any external data, including additional public datasets, pretrained task-specific datasets, or privately collected data outside the released project files, is not permitted.

**Where can I find GPU resources?**
We recommend utilizing Colab (https://colab.google.com/) for deep learning tasks.

---

## References

[1] Price M, Ballard C, Benedetti J, et al. CBTRUS statistical report: primary brain and other central nervous system tumors diagnosed in the United States in 2017–2021. *Neuro-oncology*, 2024, 26(Suppl 6): vi1.

[2] Wang Y R J, Wang P, Yan Z, et al. Advancing presurgical non-invasive molecular subgroup prediction in medulloblastoma using artificial intelligence and MRI signatures. *Cancer Cell*, 2024, 42(7): 1239–1257. e7.

[3] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition. *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016: 770–778.