# MP1 Regression Analytics Laboratory

A Mountain Path Academy–styled Streamlit application for:

- Simple Linear Regression (SLR)
- Multiple Linear Regression (MLR)
- Binary Logistic Regression

## Features

- Excel/CSV upload and worksheet selection
- User-selected Y and X variables
- Robust numeric parsing for commas, currency symbols, percentages and parenthesized negatives
- OLS inference, ANOVA-style model statistics, residual diagnostics and VIF
- Logistic odds ratios, ROC/AUC, precision–recall, confusion matrix and Hosmer–Lemeshow calibration
- Interactive Plotly charts
- Formatted Excel report downloads
- MP1 navy-and-gold design and Prof. V. Ravichandran profile

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Upload this folder to a GitHub repository.
2. In Streamlit Community Cloud, create an app from the repository.
3. Set the main file path to `app.py`.
4. Deploy.

Educational use only. Regression outputs should be validated before operational decision-making.

