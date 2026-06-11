# LoanIQ — AI Loan Approval Expert System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Yes-orange?logo=streamlit)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-green?logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Unspecified-lightgrey)](#license)


## Overview

LoanIQ is an explainable loan approval dashboard built in Streamlit. It blends four intelligent decision layers into one hybrid system:

- **Rule-Based Expert System** for deterministic underwriting logic
- **Bayesian Probabilistic Reasoning** for evidence-driven credit scoring
- **Random Forest Machine Learning** for predictive approval probability
- **BFS Search Path Scoring** for path-based decision transparency

This repository is ideal for demoing explainable AI, financial risk analytics, and hybrid decision systems.

---

## Visual Preview

![LoanIQ dashboard preview](https://via.placeholder.com/900x320.png?text=LoanIQ+Dashboard+Preview)

> Replace the placeholder with an actual app screenshot once you capture the Streamlit UI.

---

## Table of contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Project structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Dataset](#dataset)
7. [How it works](#how-it-works)
8. [Ethics & bias analysis](#ethics--bias-analysis)
9. [Roadmap](#roadmap)
10. [License](#license)

---

## Features

- **Interactive user interface** for applicant data entry and instant scoring
- **Hybrid fusion engine** combining ML, rules, Bayesian inference, and search
- **Explainability** via:
  - Fired positive/negative underwriting rules
  - Bayesian evidence table
  - Random Forest feature importance
  - BFS decision path visualization
- **Model performance dashboard** with accuracy, precision, recall, F1 score, and confusion matrix
- **Bias analysis** for education, employment, and income groups
- **Self-contained pipeline** inside `app.py` with no hidden external service dependencies

### What makes LoanIQ different?

- Combines multiple decision methodologies into one transparent scoring system
- Designed to support audit-friendly review of every loan decision
- Uses plain-language explanations for both approval and rejection outcomes
- Includes a built-in ethics section to help identify potential bias patterns

---

## Architecture

<img width="2125" height="2031" alt="loan_approval_system_architecture" src="https://github.com/user-attachments/assets/439e5d41-d58f-460a-9600-a699a95d75aa" />

flowchart LR
  A[Applicant Input Form] --> B[Hybrid Decision Engine]
  B --> C[Rule-Based System]
  B --> D[Bayesian Reasoner]
  B --> E[Random Forest Model]
  B --> F[BFS Search Path]
  C --> G[Score Fusion]
  D --> G
  E --> G
  F --> G
  G --> H[Final Approval / Rejection]
  H --> I[Explainable Analytics & Reporting]
```

---

## Project structure

| File | Description |
|---|---|
| `app.py` | Main Streamlit application and hybrid intelligence engine |
| `loan_approval_dataset.csv` | Training dataset used by the app |
| `Dataset_Link.txt` | Original Kaggle dataset source URL |
| `loan-approval-prediction-random-forest-model.ipynb` | Notebook for model exploration and analysis |
| `requirements.txt` | Python dependencies for this project |

---

## Installation

### Prerequisites

- Python 3.10 or newer
- Git (optional, for cloning the repository)

### Install dependencies

```bash
pip install -r requirements.txt
```

If you prefer not to use `requirements.txt`, install directly:

```bash
pip install streamlit pandas numpy plotly scikit-learn
```

---

## Usage

### Run the web app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, typically:

```text
http://localhost:8501
```

### App workflow

1. Enter applicant profile details in the sidebar
2. Set income, loan request, credit score, and asset values
3. Click **Analyse Application** to run the decision engine
4. Review the verdict, layer breakdown, and explanation cards

---

## Dataset

This repository uses the Loan Approval dataset from Kaggle:

- https://www.kaggle.com/datasets/rohitgrewal/loan-approval-dataset

Make sure `loan_approval_dataset.csv` remains in the project root. The app will fail to load if the CSV is missing.

---

## How it works

### 1. Data preprocessing

`app.py` loads the dataset and performs preprocessing, including:

- label encoding for categorical fields
- feature engineering for loan-to-income, total assets, asset-to-loan, and income per dependent
- standard scaling for model input

### 2. Model training

The app trains three classifiers on launch:

- Random Forest
- Logistic Regression
- Decision Tree

The trained models are cached by Streamlit so repeated reloads are faster.

### 3. Decision fusion

LoanIQ computes four separate scores and fuses them into a single verdict:

- **ML approval probability** from Random Forest
- **Bayesian approval probability** from conditional evidence
- **Rule-based credit score** using expert underwriting rules
- **BFS search confidence** from a decision path graph

The final verdict uses a weighted combination of these scores and a configurable threshold.

---

## Ethics & bias analysis

The app includes a dedicated ethics dashboard section that:

- compares approval rates by education (`Graduate` vs `Not Graduate`)
- compares approval rates by self-employed status
- compares approval trends across income quintiles
- displays the confusion matrix for the Random Forest classifier

These metrics make it easier to identify and discuss bias in model behavior.

---

## Roadmap

- [ ] Add real screenshot and demo GIF to the README
- [ ] Add a persisted model export option (`joblib` or `pickle`)
- [ ] Add user authentication for a production-ready demo
- [ ] Extend the dataset with alternate credit features and fairness-aware training
- [ ] Add unit tests for decision components and data preprocessing

---

## Contributing

Contributions are welcome. If you want to improve LoanIQ:

1. Fork the repository
2. Create a new branch for your changes
3. Submit a pull request with a clear description

---

## License

This project does not include a license file yet. Add a license if you want to open source it publicly.
