# 2026 Vivli AMR Surveillance Data Challenge

This repository contains the code, data processing pipeline, and analytical workflow developed for the **2026 Vivli AMR Surveillance Data Challenge**.

---

## Data Availability Note

> [!IMPORTANT]
> **The raw input dataset is not uploaded to this repository.**  
> The raw data was provided directly by **Vivli** for the purpose of this challenge. Because this repository is public, the input database has been omitted. However, one can recreate aour input database by following the next section.

---

## Data Preprocessing

Starting from the **ATLAS (Pfizer)** surveillance dataset (as of May 25, 2026), our preprocessing workflow was structured as follows:

* We filtered ATLAS (Pfizer) data for *Escherichia coli* (*E. coli*) to the period **2018–2024**.
* We restricted the dataset to a dense ($\ge 95\%$ populated) and non-degenerate (non-susceptible isolates $\ge 2\%$) 15-drug panel containing **CLSI** breakpoints.
* For each drug, we used the laboratory interpretation (**S/I/R**), which applies the clinical breakpoint and sidesteps interval-censoring of raw MICs (Minimum Inhibitory Concentrations).
* Consistent with international surveillance convention, **Intermediate (I)** was grouped with **Resistant (R)** as **non-susceptible (NS)**. *(Note: The structural results are unchanged under a strict R-only rule)*.
* **Colistin** was excluded from structural analysis as a known broth-microdilution artefact (100% of isolates were NS).

---

## Repository Structure

```text
.
├── code/          # Scripts for exploratory data analysis and figures
├── outputs/       # Figures, outputs from the model and rank frequency plots for the most abundant species
└── README.md      # Project documentation