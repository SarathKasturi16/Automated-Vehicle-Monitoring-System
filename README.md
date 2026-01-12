# 🚦 Automated Vehicle Monitoring System

**Vehicle Detection • Tracking • Speed Estimation • Traffic Analytics Dashboard**

An end-to-end, production-style traffic analytics system built using **YOLOv8**, **ByteTrack**, and **Streamlit**.

This project performs **vehicle-centric traffic analysis** on uploaded traffic videos, including detection, tracking, speed estimation, direction analysis (IN / OUT), and presents the results through an interactive web dashboard.

---

## 📌 Key Features

- 🚗 Vehicle detection and segmentation using **YOLOv8**
- 🔁 Multi-object tracking using **ByteTrack**
- 🆔 Stable vehicle ID assignment with re-identification logic
- ⚡ Real-time vehicle speed estimation (km/h)
- ↕️ Direction detection (IN / OUT) using virtual line crossing
- 🗄️ Vehicle-centric analytics stored in **SQLite**
- 📊 Interactive dashboard with tables and metrics
- 🎥 Annotated output video with bounding boxes, IDs, and speed
- 🌐 Web interface built using **Streamlit**

---

## 🏗️ System Architecture

Input Video → YOLOv8 Segmentation (Detection) → ByteTrack (Tracking) → Stable Vehicle ID Layer (Re-Identification) → Speed & Direction Estimation → SQLite Vehicle Analytics Database → Streamlit Dashboard (Video & Metrics)

This system is **vehicle-centric**, meaning:

- Each vehicle is tracked across its full lifecycle
- Analytics are computed per vehicle (not per frame)

---

## 📁 Project Structure
```text
Automated-Vehicle-Monitoring-System/
│
├── core_pipeline.py # Core computer vision pipeline
├── streamlit_app.py # Streamlit application
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── .gitignore # Git ignore rules
```


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/SarathKasturi16/Automated-Vehicle-Monitoring-System.git
cd Automated-Vehicle-Monitoring-System
```
### 2️⃣ Install Python Dependencies
- All required libraries are listed in requirements.txt.
```bash
pip install -r requirements.txt
```
### 3️⃣ Install FFmpeg (Required)
- FFmpeg is required to convert output videos into a Streamlit-compatible format.
- For Windows
Download from https://ffmpeg.org/download.html
Add FFmpeg to PATH
```bash
ffmpeg -version
```
### Running the Application
```bash
python -m streamlit run streamlit_app.py
```


