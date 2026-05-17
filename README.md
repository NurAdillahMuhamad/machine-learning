# ⚙️ Predictive Maintenance AI Dashboard

> Real-time machine failure prediction dashboard powered by Random Forest ML model — built for industrial IoT monitoring at PT Industri Maju, Semarang Plant.

🔗 **Live Demo:** [predictive-maintenance-semarang.streamlit.app](https://predictive-maintenance-semarang.streamlit.app)

---

## 📌 Problem Statement

Unplanned machine downtime is one of the costliest problems in manufacturing. A single failure on a CNC line can cost **Rp 42.000.000+** and cause **6–8 hours** of production loss.

This dashboard predicts machine failure **before it happens** using real-time sensor data, enabling maintenance teams to act proactively instead of reactively.

---

## 🖥️ Dashboard Features

| Feature | Description |
|---|---|
| 🎯 Failure Probability Gauge | Real-time probability score (0–100%) with color-coded risk level |
| ⚠️ Dynamic Alert Banner | Contextual warnings: Normal / Peringatan / Kritis |
| 🔧 Failure Type Detection | Automatic classification: Tool Wear, Heat Dissipation, Power, Random |
| 💰 Business Impact Estimator | Estimated downtime cost and duration |
| 📊 Feature Importance | Live bar chart from actual model weights |
| 🌡️ Sensor Status Monitor | 5-parameter live tracking with threshold indicators |
| 🎛️ Interactive Sidebar | Real-time slider input for all sensor values |

---

## 🤖 Machine Learning Model

| Property | Details |
|---|---|
| Algorithm | Random Forest Classifier |
| Dataset | AI4I 2020 Predictive Maintenance (UCI ML Repository) |
| Target | Binary classification — `machine failure` (0/1) |
| Recall | **78%** |
| Imbalanced Data Handling | SMOTE (Synthetic Minority Oversampling Technique) |

### Input Features

| Feature | Unit | Description |
|---|---|---|
| `suhu_udara` | Kelvin | Ambient air temperature |
| `suhu_proses` | Kelvin | Process temperature |
| `kecepatan` | RPM | Rotational speed |
| `torsi` | Nm | Torque |
| `keausan_alat` | minutes | Tool wear time |

### Why Recall Matters

In predictive maintenance, **missing a failure (false negative) is far more costly** than a false alarm. The model is optimized for recall — catching as many real failures as possible.

---

## 🗂️ Project Structure

```
machine-learning/
├── app.py                              # Streamlit dashboard (main app)
├── predictive_maintenance_model.pkl    # Trained Random Forest model
├── predictive_maintenance_features.pkl # Feature names
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

---

## 🚀 Run Locally

### Prerequisites
- Python 3.8+
- Anaconda (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/nuradillahmuhamad/machine-learning.git
cd machine-learning

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## 📦 Dependencies

```
streamlit
pandas
scikit-learn
joblib
plotly
imbalanced-learn
numpy
```

---

## 📈 Dataset

**AI4I 2020 Predictive Maintenance Dataset**
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)
- 10,000 data points from a simulated milling machine
- Features: air temperature, process temperature, rotational speed, torque, tool wear
- Failure types: Tool Wear Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, Random Failure

---

## 🧠 Model Training Highlights

- Applied **SMOTE** to address class imbalance (failures are rare ~3.4% of data)
- Tuned Random Forest hyperparameters for high recall
- `predict_proba()` used to output continuous failure probability instead of binary prediction
- Feature importance extracted directly from `model.feature_importances_`

---

## 👤 Author

**Nuradillah Muhamad**
Semarang, Central Java — Industrial AI & Machine Learning Portfolio

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
