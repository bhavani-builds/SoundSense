# 🎧 SoundSense — Audio Signal Analyzer

SoundSense is a Python-based audio signal processing application that allows users to upload an audio file and analyze, visualize, add noise, and filter the audio signal.

The project demonstrates practical Digital Signal Processing (DSP) concepts using Python and provides an interactive interface through Streamlit.

---

## 🚀 Features

### 📊 Audio Signal Analysis
- Upload WAV and MP3 audio files
- Play uploaded audio
- Display audio waveform
- Calculate FFT (Fast Fourier Transform)
- Display frequency spectrum
- Detect dominant frequency
- Generate and display spectrogram

### 🧹 Noise & Filtering
- Add Gaussian noise to an audio signal
- Apply different digital filters:
  - Low-Pass Filter
  - High-Pass Filter
  - Band-Pass Filter
  - Band-Stop Filter
- Adjust filter cutoff frequencies
- Visualize filter frequency response

### 📈 Signal Quality Analysis
- Calculate Signal-to-Noise Ratio (SNR)
- Compare SNR before and after filtering
- Display SNR improvement
- Compare noisy and filtered waveforms

### 🎧 Output
- Listen to the filtered audio
- Download the processed audio as a WAV file

---

## 🛠️ Technologies Used

- **Python**
- **NumPy**
- **SciPy**
- **Librosa**
- **Matplotlib**
- **SoundFile**
- **Streamlit**

---

## 📂 Project Structure

```text
SoundSense/
│
├── app.py
│
├── signal_processing/
│   ├── __init__.py
│   ├── analysis.py
│   ├── filters.py
│   └── noise.py
│
├── requirements.txt
├── README.md
└── .gitignore
