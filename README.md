# 📊 SocialPulse – Multi-Agent Social Listening & Trend Prediction Dashboard

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge\&logo=python\&logoColor=white)
![Mistral AI](https://img.shields.io/badge/Mistral_AI-API-orange?style=for-the-badge)

SocialPulse is a **social media analytics dashboard** built with **Streamlit**. It ingests CSV data, performs sentiment analysis, detects trending topics using clustering and keyword extraction, and uses the **Mistral AI API** to generate executive summaries and actionable brand insights.

---

## ✨ Features

* **📂 CSV Upload & Validation**
  Upload social media datasets with columns like `text`, `date`, `username`, `likes`, and `platform`. Missing optional fields are automatically handled.

* **😊 Sentiment Analysis**
  Classifies posts/comments as **Positive**, **Negative**, or **Neutral**.

* **📈 Trend Detection**

  * Top keywords and hashtags
  * K-Means clustering of similar posts
  * Emerging topic identification
  * Trend ranking based on frequency and sentiment

* **🤖 AI Insights with Mistral**

  * Conversation summary
  * Trend explanation
  * 3–5 actionable brand recommendations
  * Executive summary for presentation

* **📊 Interactive Dashboard**
  Beautiful charts and analytics using Plotly / Matplotlib, plus word clouds and top author insights.

* **💾 Export Options**
  Download cleaned CSV data and analysis reports in TXT format.

---

## 🛠️ Tech Stack

| Area               | Technology                           |
| ------------------ | ------------------------------------ |
| Frontend           | Streamlit                            |
| Data Processing    | Pandas, NumPy                        |
| Machine Learning   | Scikit-learn, K-Means, TF-IDF        |
| Sentiment Analysis | TextBlob / fallback rule-based logic |
| Visualization      | Plotly, Matplotlib, WordCloud        |
| AI Integration     | Mistral AI API                       |

---

## 📁 Project Structure

```bash
SocialPulse/
├── app.py
├── requirements.txt
├── .env.example
├── utils/
│   ├── data_processing.py
│   └── visualization.py
├── services/
│   ├── sentiment.py
│   ├── trends.py
│   └── mistral_ai.py
├── sample_data/
│   └── sample_social_data.csv
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/socialpulse-dashboard.git
cd socialpulse-dashboard
```

### 2. Create a Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Mistral API Key

Create a `.env` file in the root folder:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

If the API key is not available, the app will use a safe fallback summary and still run properly.

### 5. Run the App

```bash
streamlit run app.py
```

Open the app in your browser at:

```bash
http://localhost:8501
```

---

## 📋 Sample CSV Format

Your CSV file should contain at least a `text` column.

```csv
text,date,username,likes,platform
"Loving the new update! #design",2026-04-01,user123,45,Twitter
"This feature is confusing",2026-04-01,user456,12,Instagram
```

If `date`, `username`, `likes`, or `platform` are missing, the app adds default values automatically.

---

## 🔒 Environment Variables

| Variable          | Description             | Required |
| ----------------- | ----------------------- | -------- |
| `MISTRAL_API_KEY` | Your Mistral AI API key | Optional |

---

## 🧠 How It Works

1. User uploads a CSV file.
2. Data is validated and cleaned.
3. Sentiment analysis is performed on each text entry.
4. Keywords and trends are extracted.
5. Mistral AI generates summary insights.
6. Results are displayed on an interactive dashboard.
7. Reports can be exported as CSV or TXT.

---

## 🎯 Use Case

This project can be used by:

* brands tracking customer feedback
* creators analyzing audience comments
* marketers identifying emerging trends
* students learning social media analytics and NLP

---

## 🧪 Example Output

* Sentiment distribution chart
* Top trending keywords
* Cluster-based conversation themes
* AI-generated executive summary
* Exportable report

---

## 📌 Future Enhancements

* Real-time social media API integration
* Multi-language sentiment analysis
* Advanced topic modeling using LDA / BERTopic
* Database support for historical analytics
* User login and personalized dashboards

---

## 👨‍💻 Author

**Your Name**
Semester 8 – Social Media Analytics
Computer Science and Engineering

---

## 📄 License

This project is for academic and educational use.
You may add an MIT License file if required.
