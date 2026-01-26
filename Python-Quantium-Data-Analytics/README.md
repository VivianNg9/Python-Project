<h1 align="center">Quantium-Retail Analytics</h1>

## Quantium Retail Analytics — Chips Category Performance  
🔗 [Report](https://github.com/VivianNg9/Python-Project/blob/main/Python-Quantium-Data-Analytics/Report.pdf)

*Analysis of customer behaviour, trial-store performance, and commercial recommendations*

## 📘 Overview  
This project is part of the **Forage x Quantium Data Analytics Virtual Experience**, where I analysed supermarket chips purchases to uncover customer behaviour patterns, assess a store-layout trial, and translate analytics into commercial recommendations.

The program consists of three components:

1. **Data Preparation & Customer Analytics**  
2. **Experimentation & Uplift Testing**  
3. **Commercial Insight & Strategic Recommendation**

---

## Task 1 — Data Preparation & Customer Analytics  
**Files:** [`QVI_task1.ipynb`](https://github.com/VivianNg9/Python-Project/blob/main/Python-Quantium-Data-Analytics/Customer%20Analysis/QVI_customer_analysis.ipynb)
**Datasets:** [`QVI_purchase_behaviour.csv`](https://github.com/VivianNg9/Python-Project/blob/main/Python-Quantium-Data-Analytics/Customer%20Analysis/QVI_purchase_behaviour.csv), [`QVI_transaction_data.xlsx`](https://github.com/VivianNg9/Python-Project/blob/main/Python-Quantium-Data-Analytics/Customer%20Analysis/QVI_transaction_data.csv)

### What I Did
- Cleaned and prepared transaction data (date formatting, outlier removal, filtering for chip products).  
- Profiled customers across **Lifestage** and **Member Type**.  
- Conducted a deep dive into the **Mainstream Young Singles/Couples** segment to understand brand and pack-size preference.

### 🔍 Key Insights
- **Top value-driving segments:** Budget Older Families, Mainstream Young Singles/Couples, and Mainstream Retirees.  
- **Older Families purchase more per shop**, while **Mainstream Young Singles/Couples drive category volume** due to population size.  
- **Kettles 175g remains the top-performing SKU** across most customer groups.  
- Mainstream Young Singles/Couples are **28% more likely to buy Tyrells** and show a **strong preference for larger 270g packs**.

---

## Task 2 — Experimentation & Uplift Testing  
**Files:** [`QVI_task2.ipynb`](https://github.com/VivianNg9/Python-Project/blob/main/Python-Quantium-Data-Analytics/Trial%20Store%20Analysis/QVI_trial_store_analysis.ipynb)  
**Dataset:** [`QVI_data.csv`](https://github.com/VivianNg9/Python-Project/blob/main/Python-Quantium-Data-Analytics/Trial%20Store%20Analysis/QVI_data.csv)

### What I Did
- Selected control stores using Pearson correlations and magnitude distance.  
- Compared trial vs control performance for stores **77, 86, 88** on sales and customer counts.  
- Conducted significance testing to determine whether observed uplift exceeded normal variation.

### 🔍 Key Insights
- **Control matches identified:**  
  - Store 77 → 233  
  - Store 86 → 155  
  - Store 88 → 40  
- **Store 77 shows the strongest uplift**, with significant increases in sales and customer count.  
- **Store 86 shows meaningful customer growth**, though sales uplift is less consistent.  
- **Store 88 shows no significant impact**, suggesting demographic or execution misalignment.

Overall, the trial demonstrates **clear success in two locations**, supporting a **targeted rollout** rather than a full-scale implementation.

---

## Task 3 — Commercial Application  
Developed a strategic PowerPoint report using the **Pyramid Principle**, highlighting:

- Who the highest-value customer segments are  
- Where the trial delivered measurable uplift  
- Which stores present the strongest rollout opportunity  
- Clear, evidence-backed recommendations for the Chips Category Manager  

---

## 🛠️ Tech Stack  
- **Python 3.10**  
- **Libraries:** pandas, numpy, matplotlib, seaborn, datetime, sklearn, scipy, mlxtend

---

## 📈 What This Project Demonstrates  
- End-to-end retail analytics capability  
- Statistical experiment design & uplift measurement  
- Customer segmentation and behaviour analysis  
- Commercial storytelling for business stakeholders  
