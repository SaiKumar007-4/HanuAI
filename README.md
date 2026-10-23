# HanuAi – Machine Learning & Data Analytics Assessment

This repository contains the complete implementation, datasets, notebooks, and executive reports for the **HanuAi ML Assessment**.

---

## 📂 Repository Structure & Key Deliverables

```
├── task1_web_scraping_sentiment_analysis.ipynb   # Task 1 Notebook: BestBuy Canada Scraper & VADER Sentiment
├── task1_bestbuy_reviews_sentiment.csv           # Task 1 Dataset: Extracted Reviews & Aspect Tags (CSV)
├── task1_bestbuy_reviews_sentiment.xlsx          # Task 1 Dataset: Extracted Reviews & Aspect Tags (Excel)
├── task2_advanced_eda_text_mining.ipynb          # Task 2 Notebook: Advanced EDA, Text Mining & Clustering
├── task2_processed_text_mining_tags.csv          # Task 2 Dataset: 1,000 Mined Records & Categories (CSV)
├── task2_processed_text_mining_tags.xlsx         # Task 2 Dataset: 1,000 Mined Records & Categories (Excel)
├── Hanuai_task2_assignment.csv                   # Original Input Warranty Dataset
├── HANUAI_ML_ASSESSMENT_FINAL_REPORT.md          # Comprehensive Technical & Stakeholder Report
├── HANUAI_ML_ASSESSMENT_REPORT.pdf               # Executive Presentation PDF Report
├── generate_task1_dataset.py                     # Pipeline Script for Task 1
├── process_task2_complete.py                     # Pipeline Script for Task 2
├── execute_notebooks.py                          # Automated Headless Notebook Runner
└── generate_pdf_report.py                        # ReportLab PDF Generation Script
```

---

## 🚀 Tasks Overview

### Task 1: Web Scraping and Aspect-Based Sentiment Analysis
- **Target Platform:** BestBuy Canada (`www.bestbuy.ca/en-ca`)
- **Product:** *Evenflo EveryStage DLX All-in-One Convertible Car Seat* (SKU: 13862215)
- **Volume:** 69 reviews mined across all 5 ratings (1 to 5 stars) and all 5 filters (`Most Helpful`, `Newest`, `Highest Rating`, `Lowest Rating`, `Most Relevant`).
- **Fields Extracted:** `Primary Key`, `Title`, `Review Text`, `Date` (YYYY-MM-DD), `Rating` (out of 5), `Source`, `Reviewer Name`, `Filter Applied`, `VADER Compound`, `Sentiment Polarity`, and `SENTIMENT` tag (e.g., `['Good Design & Quality (Pos)']`).
- **Anti-Scraping Solutions:** Architectural solutions for Akamai Bot Manager, Canadian geolocation restrictions, rotating residential proxies, headless fingerprint masking (`playwright-stealth`), and OneTrust cookie consent interception.

### Task 2: Advanced EDA, Text Mining and Unsupervised Clustering
- **Dataset:** 1,000 real-world automotive warranty records (Model Year 2020 vehicles across ThunderVolt, NovaSprint, NebulaCruiser, TurboFlux).
- **Statistical Analysis:** Time-to-Failure (TTF) modeling ($\text{Mean} = 223.3\text{ days}$, $\text{Median} = 202.5\text{ days}$), factory delivery lag, and plant-level defect concentration (Fort Wayne: 21.7%, Flint: 20.8%, Silao: 19.9%).
- **NLP Text Mining:** Decomposed free-text narrative logs (`CUSTOMER_VERBATIM`, `CAUSAL_VERBATIM`, `CORRECTION_VERBATIM`) into the 6 structured operational entities across all 1,000 records:
  - `Trigger`
  - `Failure Component`
  - `Failure Condition`
  - `Additional Context`
  - `Fix Component`
  - `Fix Condition`
- **Issue Categorization & Clustering:** Grouped records into 7 macro issue categories (Network & Bus Communication: 50.0%, Display Defect: 16.1%, Hardware Failure: 12.9%, Electrical & Wiring: 7.8%) and applied K-Means ($K=5$) and LDA topic modeling to identify latent engineering root causes.

---

## 🛠️ How to Run

### Run Notebooks in Browser
```bash
python -m notebook --port 8888
```

### Re-execute All Notebooks Headlessly
```bash
python execute_notebooks.py
```

### Re-generate Datasets & Reports
```bash
python generate_task1_dataset.py
python process_task2_complete.py
python generate_pdf_report.py
```
