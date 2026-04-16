# 🎵 Music AI Mood Classifier: Global 1,000 Dataset Edition
### *B.Tech AI & DS Project *

[![Live Website](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](https://anshu705-music-ai-project-app.streamlit.app)

---

## 🛠️ Achievements: What We Built
We have successfully transitioned from a local script to a cloud-based AI system.

### **1. The "Brain" (Deep Learning Model)**
* **Neural Network:** Built a Multi-Layer Perceptron (ANN) using **TensorFlow/Keras**.
* **Feature Extraction:** Advanced math extraction using **Librosa** (BPM, MFCC, Spectral Centroid, and Chroma).
* **Initial Dataset:** A curated 104-song prototype that achieved **80% accuracy** on short audio clips (ringtones).

### **2. The Web Ecosystem (Streamlit Cloud)**
* **Deployment:** Live hosting on Streamlit Community Cloud.
* **Linux Integration:** Solved complex Debian system dependencies using `packages.txt` for audio processing (`libsndfile1`).
* **Clean Git:** Implemented a robust `.gitignore` to keep the 1GB+ virtual environments and heavy binaries out of the repository.

### **3. Mobile Readiness (Android Phones)**
* **TFLite Conversion:** Optimized the model for mobile CPUs using **TensorFlow Lite**.
* **Android Studio Setup:** Initialized a Java-based native app structure designed for the **Low end chip.

---

## 📊 The "1,000 Song" Expansion (Current Phase)
We are currently scaling the dataset to **1,000 songs** to achieve production-level accuracy (95%+).
* **GTZAN Benchmark:** Integrating 1,000 scientifically categorized tracks across 10 genres.
* **Active Learning:** Adding a feedback loop to the website so user uploads are "centered" and saved for retraining.
* **Global Diversity:** Moving away from just ringtones to real-world songs from multiple countries and cultural origins.

---

## 🚀 Future Roadmap: Phase 3 & 4

### **Phase 3: The "Honour" Website Upgrade**
- [ ] **Google Sheets Integration:** Connect the website to a live datasheet for real-time curation.
- [ ] **Multi-Mood Classifier:** Expand from Happy/Sad to **Romantic, High-Energy, and Chill**.
- [ ] **Audio Visualizer:** Add real-time Waveform and Spectrogram displays during analysis.

### **Phase 4: The Native Android Application**
- [ ] **On-Device Inference:** Run the 1,000-song TFLite model directly on the Oneplus.
- [ ] **Native Music Player:** Build a Java-based UI that changes colors based on the detected mood of the song.
- [ ] **Hardware Acceleration:** Utilize the Adreno 720 GPU for faster feature extraction.

---

## 📂 Project Structure
* `app.py`: The live web engine.
* `train_ann.py`: The high-accuracy training logic.
* `extract.py`: The "Math Teacher" that turns sound into data.
* `requirements.txt`: The cloud environment blueprint.
* `packages.txt`: The Linux system audio engine.

---

**Developed by:** Manan Bansal  
