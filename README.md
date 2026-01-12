**An ML + LLM powered travel planning system using multi-agent reasoning, explainable ML, and interactive analytics**

---

## 📌 Overview

This project implements a **personalized, explainable trip planning system** that combines:

* **LLM-based agentic reasoning** (CrewAI + LangChain)
* **Classical Machine Learning models** for recommendation and optimization
* **Explainability tools (SHAP)** to justify decisions
* **Interactive Streamlit UI** for end users

Unlike simple chatbot planners, this system **separates reasoning, recommendation, optimization, and explanation into modular agents and ML pipelines**, making it transparent, extensible, and research-ready.

---

## Highlights

* 🔗 **Hybrid AI System**: LLM agents + ML models working together
* 🤖 **Multi-Agent Architecture**: Specialized agents for planning, filtering, and optimization
* 📊 **Explainable AI**: SHAP-based explanations for ML-driven recommendations
* 🧠 **Personalization**: Uses user profiles and historical behavior
* 🗂️ **Custom Datasets**: Self-curated POIs + synthetic & Kaggle-based user data
* 🖥️ **Interactive UI**: Streamlit dashboard with analytics and explanations

---

## 🧠 System Architecture

```text
User Input
   ↓
ML Models (classification, scoring, recommendation)
   ↓
Explainability Layer (SHAP)
   ↓
CrewAI Agents (reasoning & planning)
   ↓
Optimized Itinerary
   ↓
Streamlit UI + Analytics Dashboard
```

---

## 🧩 Core Components

### 1. Agentic AI (LLM Layer)

* **CrewAI** for multi-agent orchestration
* **LangChain** for prompt and tool management
* Agents collaborate to:

  * Interpret user intent
  * Plan itineraries
  * Justify recommendations

### 2. Machine Learning Layer

| Task                     | Model                                  |
| ------------------------ | -------------------------------------- |
| Trip Type Classification | Decision Tree, Random Forest           |
| POI Preference Scoring   | Naive Bayes                            |
| Recommendation System    | Hybrid (Collaborative + Content-based) |
| Itinerary Optimization   | Ranking + Constraint-based selection   |

### 3. Explainability

* **SHAP** used to explain:

  * Why destinations were chosen
  * Which user features influenced decisions
* Example explanation:

  > *“This destination was selected due to your preference for cultural activities, mid-range budget, and prior visit history.”*

---

## 📁 Datasets Used

### 🔹 Points of Interest (POIs)

* **Custom-built dataset**
* Curated manually and programmatically
* Contains:

  * Location
  * Category
  * Estimated cost
  * Popularity
  * Activity type

### 🔹 Users Dataset

* **Kaggle-based + synthetic data**
* User demographics, budgets, preferences

### 🔹 User History

* Generated using Python (fake but realistic values)
* Past trips, interactions, and feedback
* Enables collaborative filtering & analytics

> ⚠️ No real personal data is used. All user data is anonymized or synthetic.

---

## 🗂️ Project Structure

```text
agentic_trip_planner/
│── main.py                      # Entry point (agents + ML pipeline)
│── trip_agents.py               # CrewAI agent definitions
│── trip_tasks.py                # Agent task prompts
│
├── ml_models/
│   ├── preprocess.py
│   ├── destination_classifier.py
│   ├── preference_scorer.py
│   ├── recommender.py
│   ├── optimizer.py
│   └── explainability.py
│
├── data/
│   ├── users.csv
│   ├── user_history.csv
│   └── pois.csv
│
├── ui/
│   ├── app.py                   # Streamlit app
│   └── components/
│
├── models/                      # Saved ML models (joblib)
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

* **Windows**

```bash
venv\Scripts\activate
```

* **Linux/Mac**

```bash
source venv/bin/activate
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Core libraries include:

* `crewai`, `langchain`
* `scikit-learn`, `pandas`, `numpy`
* `shap`, `matplotlib`, `seaborn`
* `streamlit`
* `transformers`, `accelerate`

---

## ▶️ Running the Project

### Run the Streamlit App

```bash
streamlit run ui/app.py
```

### Run the Agentic Pipeline (CLI)

```bash
python main.py
```

---

## 📊 Analytics & Dashboard

The Streamlit UI provides:

* Popular destination trends
* Budget distribution
* User preference heatmaps
* Trip history visualization
* SHAP explanation plots

---

## 🧪 Research & Novelty

### Why this project is novel:

* Combines **agentic LLM reasoning with classical ML**
* Adds **explainability** to recommender systems
* Uses **synthetic + real-world datasets** responsibly
* Modular design suitable for **conference or journal publication**

### Suitable For:

* Pattern Recognition / ML coursework
* Agentic AI research
* Explainable AI studies
* Recommender systems research

---

Future Enhancements

* Deep learning–based preference modeling
* Graph-based POI relationships
* Reinforcement learning for itinerary optimization
* Online learning from real user feedback
* API-based deployment (FastAPI)
