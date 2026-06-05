\# 🧠 StressSense AI – Multi-Model Stress Detection System



StressSense AI is a full-stack machine learning project that detects human stress levels using physiological sensor data. The system explores and compares multiple deep learning architectures including CNN-LSTM, BiLSTM, and Attention-based models to achieve improved prediction performance.



\---



\## 🚀 Project Highlights



\- Real-time stress detection from physiological signals

\- Multiple deep learning models implemented and compared

\- Model evolution: CNN-LSTM → BiLSTM → Attention-based model

\- REST APIs for prediction and integration with frontend

\- Interactive dashboard for visualization and results

\- End-to-end ML pipeline (data → training → prediction → UI)



\---



\## 🧠 Machine Learning Pipeline



\### 📌 Models Implemented



1\. \*\*CNN + LSTM Model\*\*

&#x20;  - Extracts spatial + temporal features

&#x20;  - Used as baseline model



2\. \*\*BiLSTM Model\*\*

&#x20;  - Captures bidirectional temporal dependencies

&#x20;  - Improves sequence understanding



3\. \*\*Attention-Based Model\*\*

&#x20;  - Focuses on important time-steps in signals

&#x20;  - Best-performing model



\---



\## 📊 Input Features



\- EDA (Electrodermal Activity)

\- HR (Heart Rate)

\- BVP (Blood Volume Pulse)

\- Temperature



\---



\## 🎯 Output



\- Stress classification:

&#x20; -  Low Stress

&#x20; - Medium Stress

&#x20; -High Stress

\- Confidence score from model prediction



\---



\## 🛠️ Tech Stack



\### Backend

\- Python

\- Flask / FastAPI

\- NumPy, Pandas

\- TensorFlow / PyTorch

\- Scikit-learn



\### Frontend

\- React.js / Angular

\- TypeScript

\- Tailwind CSS



\### Tools

\- Git \& GitHub

\- VS Code



\---



\## 📁 Project Structure

StressSense/

│

├── BACKEND/

│ ├── models/ # CNN, BiLSTM, Attention models

│ ├── api.py # REST APIs

│ ├── train\_cnn\_lstm.py

│ ├── train\_bilstm.py

│ ├── train\_attention.py

│

├── FRONTEND/

│ ├── src/

│ ├── components/

│

├── utils/

├── requirements.txt

└── README.md





\---



\## 📡 API Endpoints



| Method | Endpoint   | Description          |

|--------|------------|----------------------|

| POST   | /predict   | Predict stress level |

| GET    | /health    | API health check     |



\---



\## 📈 Model Comparison



| Model      | Performance | Notes |

|------------|------------|------|

| CNN-LSTM   | Baseline   | Good feature extraction |

| BiLSTM     | Better     | Captures both directions |

| Attention  | Best       | Focuses on important signals |



\---



\## 🧠 Key Learning Outcome



\- Built and compared multiple deep learning models

\- Learned sequence modeling (CNN, LSTM, BiLSTM)

\- Implemented Attention mechanism for better performance

\- Full-stack integration of ML + Web application



\---



\## 👩‍💻 Author



\*\*Harshini V\*\*  

MCA Graduate | Full Stack Developer | 



\---



\## 📌 Note



This project demonstrates end-to-end machine learning workflow from data preprocessing to model deployment.

