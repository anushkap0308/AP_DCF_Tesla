# AP_DCF_Tesla
A dynamic Discounted Cash Flow (DCF) valuation model with an interactive Streamlit dashboard, enabling users to estimate intrinsic stock value by adjusting key inputs such as WACC, terminal growth rate, and operating margins. Includes sensitivity analysis for deeper financial insights.

## Overview

This project is an interactive **Discounted Cash Flow (DCF) Valuation Model** built using Python and Streamlit. It allows users to estimate the **intrinsic value of a company** by adjusting key financial assumptions such as growth rate, WACC, operating margins, and reinvestment rate.

The model is designed to simulate real-world financial analysis and provide insights into how different assumptions impact valuation.

---

## What is DCF?

Discounted Cash Flow (DCF) is a valuation method used to estimate the value of an investment based on its expected future cash flows. These cash flows are discounted back to their present value using a discount rate (WACC).

---

## Features

* Interactive dashboard using Streamlit
* Adjustable inputs:

  * WACC (Discount Rate)
  * Terminal Growth Rate
  * Operating Margin
  * Tax Rate
  * Reinvestment Rate
* Intrinsic Share Price Calculation
* Sensitivity Analysis
* Comparison with Market Price

---

## Tech Stack

* Python
* Streamlit
* Pandas
* NumPy
* Matplotlib

---

## Project Structure

```
dcf-valuation-model
│-- AP_Tesla_DCF.xlsx
│-- AP_Tesla_dcf_dashboard.py
│-- README.md
```

---

## How to Run the Project

1. Clone the repository:

```
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
```

2. Navigate to the folder:

```
cd dcf-valuation-model
```

3. Run the Streamlit app:

```
streamlit run dcf_dashboard.py
```

---

## Key Insights

* The valuation is highly sensitive to **WACC** and **terminal growth rate**
* Small changes in assumptions can significantly impact intrinsic value
* Helps understand why market price may differ from calculated value

---

## Dashboard Preview

<img width="1910" height="1026" alt="image" src="https://github.com/user-attachments/assets/35c2aba8-28d1-427d-8f1d-1b8230ccbd88" />
<img width="1540" height="793" alt="image" src="https://github.com/user-attachments/assets/f7284980-bd87-4974-98c5-8431295d7cb9" />
<img width="1512" height="903" alt="image" src="https://github.com/user-attachments/assets/674e3c7e-1959-49fc-b12d-fcdaaed1cc4e" />
<img width="1525" height="893" alt="image" src="https://github.com/user-attachments/assets/f59c73b6-6140-4591-82e9-b52bc78b693c" />

---

## Use Case

This model can be used by:

* Finance students for learning valuation
* Investors for basic intrinsic value estimation
* Analysts for scenario and sensitivity analysis

---

## Disclaimer

This model is for educational purposes only and should not be considered financial advice.

---

## Author - **Anushka Pandey**
Feel free to connect and share feedback!

