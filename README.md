# 🚀 Loan Portfolio Report Automation


---

## 📊 Overview

A scalable and modular **Loan Portfolio Reporting System** built using Python and Pandas.

This project automates:

* 📈 AUM (Assets Under Management) reporting
* 📉 Full loan portfolio analytics
* ⚙️ Complex financial transformations
* 📂 Multi-source data ingestion

---

## 🏗️ Architecture

```
                +----------------------+
                |   Input CSV Files    |
                | (Multiple Sources)   |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |  Data Ingestion      |
                | (Pandas Loaders)     |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Data Transformation  |
                |  (Business Logic)    |
                +----------+-----------+
                           |
           +---------------+----------------+
           |                                |
           v                                v
+----------------------+        +------------------------+
|   AUM Report Logic   |        |  Full Report Logic     |
|  (Core Metrics)      |        | (Overdue, Interest,    |
|                      |        |  Provisioning etc.)    |
+----------+-----------+        +-----------+------------+
           |                                |
           +---------------+----------------+
                           |
                           v
                +----------------------+
                |   Output Reports     |
                |  (CSV Generation)    |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |       Logging        |
                |   (File + Console)   |
                +----------------------+
```

---

## 📁 Project Structure

```
code/
│
├── data/                        # Input datasets
├── Generated_Report/            # Output reports
├── logs/                        # Logs
│
├── Reporting/
│   └── LoanPortv2.py            # Main script
│
└── README.md
```

---

## ⚙️ Features

* Modular pipeline design
* CLI execution using argparse
* Dynamic file paths (no hardcoding)
* Logging system (file + console)
* Handles missing files gracefully
* Supports AUM and Full report modes

---

## 📥 Input Files

Place these files in `data/`:

```
aum_month_report.csv
sharing_ratio_master.csv
bookdbet_tagging.csv
bookdbet_tagging_product.csv
write_off_lans.csv
morat_data.csv
foreclosure_tagging.csv
CAL_MANAGED_DA_PTC_Master.csv
overdue_report.csv
PORTFOLIO_REPORT_OPENING.csv
Repo_Stock_report.csv
co_lending_rate_master_report.csv
```

---

## 🚀 How to Run

### Run AUM Report

```
python Reporting/LoanPortv2.py --report-type aum
```

### Run Full Report

```
python Reporting/LoanPortv2.py --report-type full
```

---

## 📤 Output

Generated in:

```
Generated_Report/
```

Example:

```
loan_portfolio_report_14_April_2026.csv
```

---

## 📝 Logs

Stored in:

```
logs/loan_portfolio_<timestamp>.log
```

---

## 🔧 Custom Run

```
python Reporting/LoanPortv2.py \
  --report-type aum \
  --data-dir /your/data/path \
  --output /your/output/report.csv
```

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Logging

---

## 📈 Future Improvements

* Airflow integration
* Docker support
* Cloud storage (S3)
* PySpark migration

---

## 👨‍💻 Author

Sairaj Sawant

---

## 📌 Note

* AUM report → Monthly MIS
* Full report → Internal analysis

---
