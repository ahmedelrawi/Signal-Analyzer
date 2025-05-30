
#  Signal Analyzer 

This project uses deep learning and digital signal processing to detect and classify signal types from their FFT representations. It combines an **Autoencoder** for anomaly detection and a **CNN classifier** for signal type classification.

---

##  Features

- Detects **anomalous signals** using Autoencoder
- Classifies signal types: `sine`, `square`, `sawtooth`, `triangle`, `AM`, `FM`
- Visual signal analysis using FFT
- Provides a **Streamlit UI** 
- Ready for **Docker deployment**

---

##  Technologies Used

- **Python 3.10**
- **NumPy**, **Matplotlib**, **Scikit-learn**
- **TensorFlow / Keras** for DL models
- **Streamlit** for interactive UI
- **Docker** for containerization

---

##  Project Structure

```
 Signal Analyzer

├── models/
│   ├── autoencoder_model.keras
│   └── classifier_model.h5
├── signals_images/               # FFT images of signals
├── app.py                        # Streamlit app
├── deploy.py                     # Flask API
├── config.py                     # Threshold, labels, etc.
├── ML pipeline.py                # Training pipeline and utilities
├── requirements.txt              # All dependencies
└── README.md
```

---

##  How to Run

### 1. Clone the repository

```bash
git clone https://github.com/ahmedelrawy/Signal-Analyzer.git
cd Signal-Analyzer
```

### 2. Create virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

---

##  Models Used

- **Autoencoder**:
  - Learns reconstruction of normal signals
  - Calculates reconstruction error for anomaly detection
  
- **CNN Classifier**:
  - Predicts one of 6 signal types
  - Trained on FFT images

---

##  Example Results

| Signal Type | Sample FFT Image |
|-------------|------------------|
| Sine        | ![sine](examples/sine.png) |
| Triangle    | ![triangle](examples/triangle.png) |

---

## ⚠️ Notes

- `classifier_model.h5` is a large file (>70MB). Consider replacing with a Google Drive download or use Git LFS.
- Project supports Docker; just ensure `Dockerfile` and `requirements.txt` are present.

---

## 👤 Author

**Ahmed Elrawy**  
AI & DSP Engineer  
[LinkedIn](https://www.linkedin.com/in/ahmed-elrawy-7b01081a7/)

---

##  License

Open source for educational and non-commercial use.
