import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import tempfile
import os

from scipy.signal import freqz

from signal_processing.analysis import (
    calculate_fft,
    find_dominant_frequency,
    calculate_spectrogram,
    calculate_snr
)

from signal_processing.noise import add_noise

from signal_processing.filters import apply_filter


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SoundSense",
    page_icon="🎵",
    layout="wide"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎵 SoundSense")

    st.write("Audio Signal Processing")

    st.markdown("---")

    st.subheader("📌 Features")

    st.write("📈 Waveform Analysis")
    st.write("📊 FFT Analysis")
    st.write("🎯 Dominant Frequency")
    st.write("🌈 Spectrogram")
    st.write("🧹 Noise Generation")
    st.write("🎛️ Digital Filters")
    st.write("📐 Filter Response")
    st.write("📊 SNR Analysis")
    st.write("🎧 Audio Processing")

    st.markdown("---")

    st.caption(
        "Built with Python, NumPy, SciPy, "
        "Librosa and Streamlit"
    )


# ============================================================
# TITLE
# ============================================================

st.title("🎵 SoundSense")

st.subheader("Audio Signal Analyzer")

st.write(
    "Analyze audio signals using digital signal "
    "processing techniques."
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3"]
)


if uploaded_file is not None:

    # ========================================================
    # LOAD AUDIO
    # ========================================================

    try:

        audio_data, sample_rate = librosa.load(
            uploaded_file,
            sr=None,
            mono=True
        )

    except Exception as e:

        st.error(
            f"Could not read audio file: {e}"
        )

        st.stop()


    # ========================================================
    # AUDIO INFORMATION
    # ========================================================

    duration = (
        len(audio_data) / sample_rate
    )


    st.success(
        "Audio uploaded successfully! ✅"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Sample Rate",
            f"{sample_rate} Hz"
        )


    with col2:

        st.metric(
            "Duration",
            f"{duration:.2f} seconds"
        )


    with col3:

        st.metric(
            "Samples",
            f"{len(audio_data):,}"
        )


    # ========================================================
    # TABS
    # ========================================================

    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Signal Analysis",
            "🧹 Filtering",
            "🎧 Output"
        ]
    )


    # ========================================================
    # TAB 1 - SIGNAL ANALYSIS
    # ========================================================

    with tab1:

        # ----------------------------------------------------
        # ORIGINAL AUDIO
        # ----------------------------------------------------

        st.subheader(
            "🔊 Original Audio"
        )

        st.audio(
            uploaded_file
        )


        # ----------------------------------------------------
        # WAVEFORM
        # ----------------------------------------------------

        st.subheader(
            "📈 Waveform"
        )


        time = (
            np.arange(
                len(audio_data)
            ) / sample_rate
        )


        fig1, ax1 = plt.subplots(
            figsize=(12, 4)
        )


        ax1.plot(
            time,
            audio_data
        )


        ax1.set_xlabel(
            "Time (seconds)"
        )


        ax1.set_ylabel(
            "Amplitude"
        )


        ax1.set_title(
            "Audio Waveform"
        )


        ax1.grid(
            True,
            alpha=0.3
        )


        st.pyplot(
            fig1
        )


        plt.close(fig1)


        # ----------------------------------------------------
        # FFT
        # ----------------------------------------------------

        st.subheader(
            "📊 Frequency Spectrum"
        )


        frequencies, magnitude = calculate_fft(
            audio_data,
            sample_rate
        )


        dominant_frequency = (
            find_dominant_frequency(
                audio_data,
                sample_rate
            )
        )


        st.metric(
            "🎯 Dominant Frequency",
            f"{dominant_frequency:.2f} Hz"
        )


        fig2, ax2 = plt.subplots(
            figsize=(12, 4)
        )


        ax2.plot(
            frequencies,
            magnitude
        )


        ax2.set_xlabel(
            "Frequency (Hz)"
        )


        ax2.set_ylabel(
            "Magnitude"
        )


        ax2.set_title(
            "Frequency Spectrum (FFT)"
        )


        ax2.set_xlim(
            0,
            min(
                5000,
                sample_rate / 2
            )
        )


        ax2.grid(
            True,
            alpha=0.3
        )


        st.pyplot(
            fig2
        )


        plt.close(fig2)


        # ----------------------------------------------------
        # SPECTROGRAM
        # ----------------------------------------------------

        st.subheader(
            "🌈 Spectrogram"
        )


        spectrogram = (
            calculate_spectrogram(
                audio_data
            )
        )


        fig3, ax3 = plt.subplots(
            figsize=(12, 5)
        )


        img = librosa.display.specshow(
            spectrogram,
            sr=sample_rate,
            x_axis="time",
            y_axis="hz",
            ax=ax3
        )


        ax3.set_title(
            "Audio Spectrogram"
        )


        fig3.colorbar(
            img,
            ax=ax3,
            format="%+2.0f dB"
        )


        st.pyplot(
            fig3
        )


        plt.close(fig3)


    # ========================================================
    # TAB 2 - FILTERING
    # ========================================================

    with tab2:

        st.subheader(
            "🧹 Noise & Digital Filtering"
        )


        # ----------------------------------------------------
        # NOISE
        # ----------------------------------------------------

        noise_level = st.slider(
            "Add Noise",
            min_value=0.0,
            max_value=0.5,
            value=0.05,
            step=0.01
        )


        noisy_signal = add_noise(
            audio_data,
            noise_level
        )


        # ----------------------------------------------------
        # FILTER TYPE
        # ----------------------------------------------------

        filter_type = st.selectbox(
            "Select Filter Type",
            [
                "Low-Pass",
                "High-Pass",
                "Band-Pass",
                "Band-Stop"
            ]
        )


        # ----------------------------------------------------
        # FILTER PARAMETERS
        # ----------------------------------------------------

        nyquist = sample_rate / 2


        max_frequency = int(
            min(
                10000,
                nyquist - 100
            )
        )


        max_frequency = max(
            max_frequency,
            500
        )


        if filter_type in [
            "Low-Pass",
            "High-Pass"
        ]:

            cutoff_frequency = st.slider(
                "Cutoff Frequency (Hz)",
                min_value=100,
                max_value=max_frequency,
                value=min(
                    3000,
                    max_frequency
                ),
                step=100
            )

            filtered_signal = apply_filter(
                noisy_signal,
                sample_rate,
                filter_type,
                cutoff_frequency=cutoff_frequency
            )


        else:

            col1, col2 = st.columns(2)


            with col1:

                low_cutoff = st.slider(
                    "Lower Cutoff (Hz)",
                    min_value=100,
                    max_value=max_frequency - 100,
                    value=min(
                        1000,
                        max_frequency - 100
                    ),
                    step=100
                )


            with col2:

                high_cutoff = st.slider(
                    "Upper Cutoff (Hz)",
                    min_value=200,
                    max_value=max_frequency,
                    value=min(
                        3000,
                        max_frequency
                    ),
                    step=100
                )


            if high_cutoff <= low_cutoff:

                st.warning(
                    "Upper cutoff must be greater "
                    "than lower cutoff."
                )

                st.stop()


            filtered_signal = apply_filter(
                noisy_signal,
                sample_rate,
                filter_type,
                low_cutoff=low_cutoff,
                high_cutoff=high_cutoff
            )


        # ----------------------------------------------------
        # FILTER RESPONSE
        # ----------------------------------------------------

        st.subheader(
            "📐 Filter Frequency Response"
        )


        if filter_type in [
            "Low-Pass",
            "High-Pass"
        ]:

            if filter_type == "Low-Pass":

                from scipy.signal import butter

                b, a = butter(
                    5,
                    cutoff_frequency / nyquist,
                    btype="low"
                )

            else:

                from scipy.signal import butter

                b, a = butter(
                    5,
                    cutoff_frequency / nyquist,
                    btype="high"
                )

        elif filter_type == "Band-Pass":

            from scipy.signal import butter

            b, a = butter(
                5,
                [
                    low_cutoff / nyquist,
                    high_cutoff / nyquist
                ],
                btype="band"
            )

        else:

            from scipy.signal import butter

            b, a = butter(
                5,
                [
                    low_cutoff / nyquist,
                    high_cutoff / nyquist
                ],
                btype="bandstop"
            )


        w, h = freqz(
            b,
            a,
            worN=2000,
            fs=sample_rate
        )


        response_db = (
            20 *
            np.log10(
                np.maximum(
                    np.abs(h),
                    1e-10
                )
            )
        )


        fig_response, ax_response = plt.subplots(
            figsize=(12, 4)
        )


        ax_response.plot(
            w,
            response_db
        )


        ax_response.set_xlabel(
            "Frequency (Hz)"
        )


        ax_response.set_ylabel(
            "Magnitude (dB)"
        )


        ax_response.set_title(
            f"{filter_type} Filter Response"
        )


        ax_response.set_xlim(
            0,
            min(
                10000,
                nyquist
            )
        )


        ax_response.set_ylim(
            -80,
            5
        )


        ax_response.grid(
            True,
            alpha=0.3
        )


        st.pyplot(
            fig_response
        )


        plt.close(fig_response)


        # ----------------------------------------------------
        # SNR
        # ----------------------------------------------------

        st.subheader(
            "📊 Signal Quality Analysis"
        )


        snr_before = calculate_snr(
            audio_data,
            noisy_signal
        )


        snr_after = calculate_snr(
            audio_data,
            filtered_signal
        )


        snr_improvement = (
            snr_after - snr_before
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "SNR Before",
                f"{snr_before:.2f} dB"
            )


        with col2:

            st.metric(
                "SNR After",
                f"{snr_after:.2f} dB"
            )


        with col3:

            st.metric(
                "Improvement",
                f"{snr_improvement:+.2f} dB"
            )


        # ----------------------------------------------------
        # SIGNAL COMPARISON
        # ----------------------------------------------------

        st.subheader(
            "📉 Noisy vs Filtered"
        )


        display_duration = min(
            5,
            duration
        )


        display_samples = int(
            display_duration *
            sample_rate
        )


        fig4, ax4 = plt.subplots(
            figsize=(12, 4)
        )


        ax4.plot(
            time[:display_samples],
            noisy_signal[:display_samples],
            label="Noisy Signal",
            alpha=0.7
        )


        ax4.plot(
            time[:display_samples],
            filtered_signal[:display_samples],
            label="Filtered Signal"
        )


        ax4.set_xlabel(
            "Time (seconds)"
        )


        ax4.set_ylabel(
            "Amplitude"
        )


        ax4.set_title(
            f"{filter_type} Filter Result"
        )


        ax4.legend()


        ax4.grid(
            True,
            alpha=0.3
        )


        st.pyplot(
            fig4
        )


        plt.close(fig4)


    # ========================================================
    # TAB 3 - OUTPUT
    # ========================================================

    with tab3:

        st.subheader(
            "🎧 Audio Output"
        )


        # ----------------------------------------------------
        # ORIGINAL
        # ----------------------------------------------------

        st.write(
            "### Original Audio"
        )


        st.audio(
            uploaded_file
        )


        # ----------------------------------------------------
        # FILTERED AUDIO
        # ----------------------------------------------------

        st.write(
            "### Filtered Audio"
        )


        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        )


        temp_file.close()


        sf.write(
            temp_file.name,
            filtered_signal,
            sample_rate
        )


        with open(
            temp_file.name,
            "rb"
        ) as audio_file:

            filtered_audio_bytes = (
                audio_file.read()
            )


        st.audio(
            filtered_audio_bytes,
            format="audio/wav"
        )


        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        st.download_button(
            label="⬇️ Download Filtered Audio",
            data=filtered_audio_bytes,
            file_name="filtered_audio.wav",
            mime="audio/wav"
        )


        # ----------------------------------------------------
        # CLEANUP
        # ----------------------------------------------------

        try:

            os.remove(
                temp_file.name
            )

        except:

            pass


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "SoundSense | Audio Signal Processing & Analysis"
)