# 📊 Developer Metrics Analytics Predictor

An elegant, production-ready web application that utilizes an **AdaBoost Classifier** to analyze developer productivity, stress, and workload metrics to generate intelligent predictions. Built with **Flask** and styled with a high-end, responsive **Tailwind CSS** interface.

🔗 **Live Production URL:** https://newproject-mauve-two.vercel.app/

---

## ✨ Features

* **Advanced Machine Learning:** Powered by a 50-estimator AdaBoost Classifier built using `scikit-learn 1.6.1`.
* **Futuristic UI (Glassmorphism):** An attractive, dark-themed dashboard built entirely using Tailwind CSS, featuring smooth focus states and dynamic status banners.
* **Fully Responsive:** Perfectly optimized for desktops, tablets, and mobile displays.
* **Serverless Optimized:** Tailored architecture designed for instant deployment on platforms like Vercel.

---

## 🛠️ Monitored Metrics (Features)

The model evaluates **9 distinct inputs** to formulate its prediction:

1. **Hours Coding:** Daily hours dedicated strictly to programming.
2. **AI Usage Hours:** Time spent leveraging AI assistance tools (Copilot, ChatGPT, etc.).
3. **Lines of Code:** Volume of source code written.
4. **Commits:** Total code repository check-ins/commits.
5. **Bugs Reported:** Number of software defects logged.
6. **Sleep Hours:** Rest duration (crucial for cognitive correlation).
7. **Distractions:** Frequency or duration of workflow interruptions.
8. **Cognitive Load:** Subjective scale measuring mental strain.
9. **Stress Level:** Scale evaluating current workplace stress.

---

## 📂 Project Architecture

This project is streamlined for single-file serverless execution by integrating the front-end layout directly into the app router.

```text
your-project/
│
├── adaboostmodel.pkl     # Pre-trained Scikit-Learn Model
├── app.py                # Core Flask Router & Embedded HTML UI
├── requirements.txt      # Production dependencies
└── vercel.json           # Vercel Configuration Settings
