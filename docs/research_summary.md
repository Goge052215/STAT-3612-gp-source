# Summary of Four Interconnected Papers: Radiomics Foundations, Standardization, Brain Tumor Epidemiology, and AI-Driven Non-Invasive Medulloblastoma Subgroup Prediction

**Compiled Summary**  
**Date:** April 2026  
**Purpose:** This Markdown document provides detailed, self-contained summaries of the four provided papers, including key methods, findings, limitations, and clinical/scientific impact. It also explicitly highlights **connections across fields** (radiomics, standardization initiatives, neuro-oncology epidemiology, and AI/ML applications in pediatric brain tumors). All summaries are derived directly from the provided document excerpts.

---

## 1. Paper: *Advancing presurgical non-invasive molecular subgroup prediction in medulloblastoma using artificial intelligence and MRI signatures*  
**Authors:** Yan-Ran (Joyce) Wang et al. (2024)  
**Journal:** *Cancer Cell* 42, 1239–1257 (July 8, 2024)  
**DOI:** https://doi.org/10.1016/j.ccell.2024.06.002

### Key Background & Motivation
- Medulloblastoma (MB) is the most common malignant CNS tumor in children/adolescents and requires **molecular subgrouping** (WNT, SHH, Group 3, Group 4) per WHO CNS5 for risk stratification and therapy.
- Current gold standard (RNA-seq or DNA methylation on post-surgical tissue) is costly, invasive, and inaccessible in resource-limited settings.
- Goal: Develop **presurgical, non-invasive, low-cost** AI model using routine MRI.

### Methods & Dataset
- **Largest international molecularly characterized MB cohort**: 934 patients from **13 centers** (11 China, 2 USA; 803 Chinese, 131 US patients).
- **689 patients** with high-resolution presurgical MRI (axial contrast-enhanced T1-weighted + T2-weighted) used for model development.
- **Deep learning tumor segmentation** (verified manually) → quantitative **radiomic features** (intra-tumoral, peri-tumoral) + **manual qualitative MRI signatures** (location, enhancement, metastases, margins, edema).
- **Two-stage ML classifiers** (LightGBM/SVM/RF/MLP ensemble):
  - 3-class: WNT vs. SHH vs. non-WNT/non-SHH (G3/G4).
  - Binary: G3 vs. G4 within non-WNT/non-SHH.
- **Robust validation**:
  - Internal 3-fold cross-validation (primary set: Beijing Tiantan Hospital).
  - External validation (cross-continental, racially diverse).
  - **Consecutive prospective validation** (n=40 fresh 2023 patients with DNA methylation ground truth).

### Major Findings
- **3-class classifier AUCs**:
  - Internal CV: WNT 0.924, SHH 0.819, G3/G4 0.810.
  - External: WNT 0.852, SHH 0.806, G3/G4 0.766.
  - Consecutive: class-weighted average AUC **0.900**.
- **Binary G3 vs. G4 AUCs**:
  - Internal 0.822, External 0.859, Consecutive **0.852**.
- **Feature importance (Shapley)**: Intra-tumoral features dominate (68–79%); machine-generated radiomics ~75–80% importance vs. human qualitative ~20–25%. T1E and T2 contribute similarly.
- Strong radiomics–NanoString gene signature correlations (heatmaps).
- **Public dataset released**: MRI signatures + clinicopathology + survival → global MB research accelerator.
- East Asia vs. North America subset differences noted for management implications.

### Impact & Limitations
- First large-scale demonstration that **AI-enabled MRI** can replace or augment molecular testing presurgically.
- Consecutive validation shows real-world clinical utility and generalizability.

---

## 2. Paper: *Radiomics: Images Are More than Pictures, They Are Data*  
**Authors:** Robert J. Gillies, Paul E. Kinahan, Hedvig Hricak (2016)  
**Journal:** *Radiology* 278(2):563–577

### Core Concept (Radiomics Definition)
- **Radiomics** = high-throughput extraction of **quantitative** features from medical images (CT, MRI, PET) → conversion of images into **mineable high-dimensional data**.
- Features include:
  - **First-order**: intensity statistics (mean, skewness, kurtosis).
  - **Second-order & higher**: texture (GLCM, GLRLM, etc.), shape, volume, habitat (physiologically distinct subregions).
- Combined with clinical/genomic data for **diagnostic, prognostic, predictive models**.

### Key Advantages (Table 1 in paper)
- Uses **standard-of-care images** (no extra scans).
- Interrogates **entire tumor** (vs. biopsy sampling error).
- Captures **intra-tumoral heterogeneity** and **stroma**.
- Enables **longitudinal monitoring**.

### Radiogenomics
- Radiomic features can **predict** gene expression/mutation status (cross-validation of biopsy genomics).
- Independent features add orthogonal information → improved decision support.

### Challenges & Vision
- Lack of standardization in image acquisition, segmentation, and feature definitions → poor reproducibility.
- Calls for **benchmarks**, shared databases, and integration into precision medicine.
- Explicitly positions radiomics as a **translational technology** for oncology (and beyond).

---

## 3. Paper: *CBTRUS Statistical Report: Primary Brain and Other Central Nervous System Tumors Diagnosed in the United States in 2017–2021*  
**Authors:** Mackenzie Price, Jill S. Barnholtz-Sloan, Quinn T. Ostrom et al. (2024)  
**Journal:** *Neuro-Oncology* 26(Suppl 5)

### Key Epidemiologic Findings (2017–2021, entire US population)
- **Overall AAAIR**: 25.34 per 100,000 (malignant 6.89; non-malignant 18.46).
- Females > males (28.77 vs. 21.78); non-Hispanic Black slightly highest incidence.
- **Malignant histopathology**: Glioblastoma = 13.9% of all tumors, **51.5% of malignant**.
- **Non-malignant**: Meningioma = 41.7% of all tumors, **56.8% of non-malignant**.
- **Pediatric (0–19 y)**: AAAIR 6.02 per 100,000; **medulloblastoma** is a major malignant contributor.
  - Most common MB molecular subtypes: SHH-activated & TP53-wildtype + non-WNT/non-SHH (G3/G4).
- **Mortality**: 87,053 deaths (avg. 4.41 per 100,000; 17,411/year). Brain/CNS cancer = leading cancer death cause in children 0–14 y.
- **5-year relative survival**:
  - Malignant: **35.7%** overall (75.3% in 0–14 y).
  - Non-malignant: **92.0%**.

### New in This Report
- First inclusion of **MB molecular subtype proportions** by age.

---

## 4. Paper: *The Image Biomarker Standardization Initiative: Standardized Quantitative Radiomics for High-Throughput Image-based Phenotyping*  
**Authors:** Alex Zwanenburg, Martin Vallières et al. (2020)  
**Journal:** *Radiology* 295:328–338 (IBSI)

### Objective & Phases
- Address **reproducibility crisis** in radiomics (different software → different values for same image).
- **Standardize 174 radiomics features** + general image-processing scheme.

### Methodology
- **Phase I**: Digital phantom (80 voxels) → reference values without image processing (25 teams).
- **Phase II**: Lung cancer CT + 5 predefined processing configurations.
- **Phase III (validation)**: Multimodality sarcoma cohort (51 patients; CT, ¹⁸F-FDG PET, T1w MRI).

### Results
- **169 of 174 features standardized** with strong/very strong consensus on reference values.
- Reproducibility (Phase III):
  - CT: 166/169 excellent.
  - PET: 164/169 excellent.
  - MRI: 164/169 excellent.
- Only 2 features (oriented minimum bounding box densities) could not be standardized.

### Impact
- Provides **verifiable reference values** and open datasets for software calibration.
- Enables **reproducible, comparable radiomics studies** worldwide.

---

## Connections Across Fields

| Field / Paper | Radiomics Foundations (Gillies 2016) | IBSI Standardization (2020) | CBTRUS Epidemiology (2024) | MB AI-MRI Prediction (Wang 2024) |
|---------------|--------------------------------------|-----------------------------|-----------------------------|----------------------------------|
| **Conceptual Link** | Introduces radiomics as “images → mineable data” for precision oncology | Provides the **standardized toolbox** (169 features + processing pipeline) | Quantifies disease burden & survival gaps | **Direct application** of radiomics + AI on MRI for presurgical MB subgrouping |
| **Methodological Link** | Texture, shape, intensity, habitats | IBSI reference values & ICC validation | N/A (descriptive) | Uses **radiomic + qualitative MRI signatures** exactly as envisioned by Gillies; benefits from IBSI standardization |
| **Clinical Link** | Calls for non-invasive biomarkers | Enables multi-center, reproducible studies | Shows MB is leading pediatric cancer death; survival still poor (35.7% 5-yr malignant) | Solves accessibility issue highlighted by CBTRUS; presurgical prediction could reduce global disparities |
| **Data Link** | Advocates shared databases | Supplies phantom & reference datasets | Population-level incidence/survival | Releases **public 934-patient MRI + molecular + survival dataset** (13 centers) |
| **Future Synergy** | Radiogenomics hypothesis | Reproducible radiogenomics possible | Molecular subtype epidemiology now trackable | AI model can be externally validated on CBTRUS-linked cohorts; standardized features ensure generalizability |

### Overarching Narrative
- **Gillies (2016)** lays the conceptual foundation: “Images are data.”
- **IBSI (2020)** solves the reproducibility crisis, making radiomics clinically deployable.
- **CBTRUS (2024)** quantifies the **urgent clinical need**—especially for pediatric MB where molecular subgrouping is mandatory but tissue access is limited.
- **Wang et al. (2024)** delivers the **first large-scale, validated proof-of-concept** that standardized radiomics + AI on routine MRI can provide presurgical molecular diagnosis, directly addressing the gaps identified in the other three papers.

**Collectively**, these works form a complete translational pipeline: **epidemiologic need → radiomics theory → standardization → AI implementation → open data for global impact**.

**Suggested Next Steps**: Integrate the Wang et al. public dataset with IBSI-compliant pipelines and CBTRUS-linked survival outcomes for multi-continental radiogenomic models.