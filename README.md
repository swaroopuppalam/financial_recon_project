# Financial Reconciliation Anomaly Detection

This project is an end-to-end anomaly detection system for financial reconciliation. It consists of:

- **Backend API**: FastAPI for real-time anomaly detection.
- **ML Model**: Isolation Forest trained using Jupyter.
- **Frontend UI**: Streamlit for anomaly visualization.
- **Dockerized Deployment**: Docker Compose to run the entire stack.

## Getting Started

1. **Train the ML Model**
   ```bash
   jupyter lab
   ```
   - Open `ml/train_model.ipynb` and run all cells.
   - This will generate `model.pkl`.

2. **Run the Project using Docker**
   ```bash
   docker-compose up --build
   ```

3. **Access the API & UI**
   - FastAPI Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - Streamlit UI: [http://127.0.0.1:8501](http://127.0.0.1:8501)

## Technologies Used
- **FastAPI**: Backend API for anomaly detection
- **Scikit-Learn**: Machine Learning model (Isolation Forest)
- **Streamlit**: Interactive UI for anomaly detection
- **Docker**: Containerized deployment

Enjoy! 🚀
