## Telco Churn Predictor – End-to-End ML Project for Customer Churn Prediction

**Author:** Ankit Kashyap  
**Project Type:** Production-grade ML system  
**Deployment:** Railway (Docker + CI/CD)

---

### Overview

An end-to-end machine learning system for predicting customer churn in a telecom setting.
The project covers the complete lifecycle—from data preparation and model training to a production-ready API and web UI deployed on **Railway** using Docker and built-in CI/CD.

---

### Problem & Value

* **Proactive churn prevention**: Identifies customers likely to churn so teams can intervene early.
* **Production ML**: Model is accessible via a REST API and a web UI—no notebooks required.
* **Reliable delivery**: Dockerized application with CI/CD ensures consistent builds and deployments.
* **Reproducibility**: MLflow tracks experiments, metrics, and model artifacts.

---

### System Design & Implementation

The system is designed with a **strict separation of concerns** to mirror real-world ML production workflows.

* **Modeling & Training**
  * Feature engineering and XGBoost classifier
  * Experiments, metrics, and artifacts logged to MLflow

* **Inference Service**
  * FastAPI application exposing:
    * `POST /predict` for churn predictions
    * `GET /` for health checks

* **Web Interface**
  * Gradio-based UI for interactive predictions

* **Containerization**
  * Dockerized Python 3.11 application
  * Service listens on port **7860** (configured dynamically via environment variable)

* **CI/CD**
  * GitHub-integrated CI/CD provided by Railway
  * Automatic build and redeployment on every push to the `main` branch

* **Cloud Deployment**
  * **Railway (fully managed container platform)**
  * Docker image built directly from the GitHub repository
  * Automatic redeployment on new commits
  * Scales down on inactivity to minimize resource usage (free tier behavior)

* **Observability**
  * Railway-provided build and runtime logs
  * Built-in health checks via FastAPI `GET /`

---

### Deployment Flow

1. Code pushed to the `main` branch on GitHub
2. Railway CI/CD pipeline automatically triggers
3. Docker image is built from the repository
4. A new container revision is deployed
5. Application becomes live at the Railway-generated service URL
6. Users access:
   * `POST /predict` for API predictions
   * `/docs` for FastAPI OpenAPI documentation
   * `/ui` for the Gradio web interface

---

### CI/CD Summary

* **Source control**: GitHub  
* **CI/CD**: Railway (GitHub OAuth integration)  
* **Artifact**: Docker image  
* **Registry**: Managed internally by Railway  
* **Runtime**: Railway container infrastructure  

**Secrets management:**
* Environment variables securely managed in Railway
* Injected at runtime (not stored in repository)

---

### Key Engineering Challenges

* **Python dependency conflicts**
  * Resolved by standardizing on **Python 3.11** for ML ecosystem compatibility

* **Container not reachable**
  * Fixed by binding the application to `0.0.0.0` and using Railway’s dynamic `PORT` configuration

* **Gradio dependency mismatch**
  * Resolved by pinning compatible `gradio` and `gradio-client` versions

* **Local vs production model paths**
  * Solved by packaging the trained model at build time and using consistent loading logic

* **Safe redeployment**
  * Achieved via Railway’s container-based redeployment model

---

### Tech Stack

* **Language**: Python 3.11
* **ML**: XGBoost, Pandas, NumPy
* **Experiment Tracking**: MLflow
* **API**: FastAPI
* **UI**: Gradio
* **Containerization**: Docker
* **CI/CD**: Railway (GitHub-integrated)
* **Cloud**: Railway

---
