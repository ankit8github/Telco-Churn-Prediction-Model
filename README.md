
---

## Telco Churn – End-to-End ML Project (Google Cloud Run)

**Author:** Ankit Kashyap
**Project Type:** Production-grade ML system
**Deployment:** Google Cloud Run (Docker + CI/CD)

---

### Overview

An end-to-end machine learning system for predicting customer churn in a telecom setting.
The project covers the complete lifecycle—from data preparation and model training to a production-ready API and web UI deployed on **Google Cloud Run** using Docker and CI/CD.

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
  * Service listens on port **7860**

* **CI/CD**

  * GitHub Actions pipeline implemented from scratch
  * Automatically builds and pushes Docker images to Docker Hub on every `main` branch push

* **Cloud Deployment**

  * **Google Cloud Run (fully managed)**
  * Pulls container images directly from Docker Hub
  * Automatic redeployment on new image versions
  * Scales to zero when idle to minimize cost

* **Observability**

  * Google Cloud Logs for container stdout/stderr
  * Built-in health checks via Cloud Run

---

### Deployment Flow

1. Code pushed to `main`
2. GitHub Actions:

   * Builds Docker image
   * Pushes image to Docker Hub
3. Google Cloud Run:

   * Pulls the latest image
   * Deploys a new revision automatically
4. Application becomes live at the Cloud Run service URL
5. Users access:

   * `POST /predict` for API predictions
   * Gradio UI via `/ui`

---

### CI/CD Summary

* **Source control**: GitHub
* **CI/CD**: GitHub Actions
* **Artifact**: Docker image
* **Registry**: Docker Hub
* **Runtime**: Google Cloud Run

**Secrets used:**

* `DOCKER_USERNAME`
* `DOCKER_PASSWORD`
* `GCP_PROJECT_ID`
* `GCP_SA_KEY`

---

### Key Engineering Challenges

* **Python dependency conflicts**

  * Resolved by standardizing on **Python 3.11** for ML ecosystem compatibility

* **Container not reachable**

  * Fixed by binding the app to `0.0.0.0` and correctly configuring Cloud Run port `7860`

* **Gradio dependency mismatch**

  * Resolved by pinning compatible `gradio` and `gradio-client` versions

* **Local vs production model paths**

  * Solved by packaging the trained model at build time and using consistent loading logic

* **Zero-downtime redeployment**

  * Achieved via Cloud Run’s revision-based deployment model

---

### Tech Stack

* **Language**: Python 3.11
* **ML**: XGBoost, Pandas, NumPy
* **Experiment Tracking**: MLflow
* **API**: FastAPI
* **UI**: Gradio
* **Containerization**: Docker
* **CI/CD**: GitHub Actions
* **Cloud**: Google Cloud Run

---