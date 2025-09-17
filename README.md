# Trotski - Real-Time AI Interview Assistant

This project provides a high-performance, real-time audio transcription and AI-powered answering server. It uses faster-whisper for low-latency STT (Speech-to-Text) and an LLM (like GPT models) to intelligently detect questions from the transcript and generate relevant, in-character answers on the fly.

The system is composed of three main parts:

- **The Server** (`optimized_stt_server_v3.py`): A WebSocket server that receives raw audio, transcribes it, analyzes the text for questions, and generates answers.
- **The Client** (`stable_audio_client_multi_os.py`): A robust, multi-platform audio streaming client that captures microphone input using FFmpeg and streams it to the server.
- **The UI** (`index.html`): A standalone, zero-dependency web interface that connects to the server to display the live transcript and Q&A panel.

## ✨ Features

- **Real-Time Transcription**: Low-latency audio transcription using faster-whisper
- **Intelligent Question Detection**: An LLM-powered analyzer detects questions from the live transcript
- **AI-Powered Answer Generation**: Generates context-aware, in-character answers for detected questions
- **Standalone Web UI**: A feature-rich, single-file `index.html` dashboard to monitor the interview
- **Multi-Platform Support**: The server and client run on Windows, macOS, and Linux
- **Robust & Stable**: Includes automatic reconnection, backpressure handling, and stable connection parameters
- **Highly Configurable**: Nearly every aspect can be configured via environment variables

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

### Python 3.9+

### FFmpeg
Required by the audio client to capture microphone audio.

- **Windows**: Download from the [official website](https://ffmpeg.org/download.html) and add to PATH, or use Chocolatey (`choco install ffmpeg`)
- **macOS**: Install via Homebrew: `brew install ffmpeg`
- **Linux**: Install via your package manager: `sudo apt-get install ffmpeg` (Debian/Ubuntu)

### NVIDIA GPU with CUDA (Recommended)
For significant performance gains with the Whisper model.

- Install the latest [NVIDIA Driver](https://www.nvidia.com/drivers/)
- Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (v11.x is compatible)
- Install [cuDNN](https://developer.nvidia.com/cudnn)

### OpenAI API Key
Required for question detection and answer generation.

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/iluxu/Trotski.git
cd Trotski
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# On Windows
.\venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Python Dependencies

Create a `requirements.txt` file with the content specified below and run:

```bash
pip install -r requirements.txt
```

**CPU-Only Note**: If you don't have an NVIDIA GPU, first install the CPU version of PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
Then run `pip install -r requirements.txt`.

### 4. Set Up Environment Variables

Create a `.env` file by copying the example:

```bash
# On Windows
copy .env.example .env

# On macOS/Linux
cp .env.example .env
```

Now, edit the `.env` file and add your `OPENAI_API_KEY`. See the `.env.example` section for all options.

## ⚙️ Usage

The process involves three steps: starting the server, opening the UI, and starting the audio client.

### 1. Run the Server

Start the server in a terminal. It will download the Whisper model on its first run.

```bash
python optimized_stt_server_v3.py
```

You should see output indicating the server is ready:
```
🎤 Server ready on ws://127.0.0.1:8123/
```

### 2. Open the Web UI

Simply open the `index.html` file in your web browser (e.g., Chrome, Firefox, Safari). No web server is needed. The page will automatically try to connect to the WebSocket server running on your local machine.

### 3. Run the Audio Client

The client needs to know which microphone to use.

#### Step A: Find Your Audio Device

Open a new terminal and run the client with the `--list-devices` flag:

```bash
python stable_audio_client_multi_os.py --list-devices
```

This will show you a list of available microphones and the correct name to use for your operating system.

#### Step B: Start Streaming

Now, run the client with the device name you found.

```bash
# Example for Windows
python stable_audio_client_multi_os.py --device "Mixage stéréo (Realtek(R) Audio)"

# Example for macOS
python stable_audio_client_multi_os.py --device ":0"

# Example for Linux
python stable_audio_client_multi_os.py --device "hw:0,0"
```

The client will connect to the server. Start speaking, and you will see the live transcript and Q&A appear in the `index.html` UI in your browser.

## 🖥️ Web UI Features (index.html)

The web UI is a powerful dashboard for monitoring the interview in real-time.

<!-- It's a good idea to add a screenshot of your UI here -->
<!-- <img src="https://i.imgur.com/your-screenshot-url.png" width="800" alt="UI Screenshot"> -->

### Three-Panel Layout

- **Transcript Panel (Left)**: Displays the live, timestamped transcription of the audio stream
- **Answer Detail (Center)**: Shows the full text of the selected question and its generated answer
- **Q&A List (Right)**: A table of all questions detected during the session. Click any question to view it in the center panel

### Status Indicators
At the top, you can see the WebSocket connection status, auto-scroll state, and a count of detected questions.

### Interactive Controls

- **📜 Auto**: Toggles auto-scrolling on the transcript panel
- **👁️ Follow**: Toggles automatically selecting the latest detected question
- **❓ Ask**: (UI-only feature) Manually type and submit a question to simulate an answer
- **🔄 Reset**: Clears the entire session state on the server and UI
- **💾 Save**: Exports the full transcript and Q&A log as a Markdown (.md) file

### Keyboard Shortcuts
The UI is fully navigable with keyboard shortcuts (e.g., j/k to navigate questions, p to toggle auto-scroll, s to save).

## ❤️ Support the Project

If you find this tool useful, please consider supporting its development. Your support helps cover API costs, encourages further development, and allows me to dedicate more time to improving it. Thank you!

<p align="center">
<a href="https://github.com/sponsors/iluxu" target="_blank">
<img src="https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86" alt="Sponsor on GitHub">
</a>
&nbsp;&nbsp;
<a href="https://www.buymeacoffee.com/iluxu" target="_blank">
<img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="28">
</a>
</p>

## 🔧 OS-Specific Tweaks & Performance

### Windows
- **Audio Configuration**: 
  - Open Sound Settings (Right-click speaker icon → Sounds → Recording tab)
  - Right-click in empty space → "Show Disabled Devices" 
  - Enable "Stereo Mix" or "What U Hear" to capture system audio
  - Set your microphone as default recording device
- Ensure Windows Defender real-time protection doesn't block audio processing
- Consider using Windows Terminal for better Unicode character display
- Set audio client to High Priority in Task Manager for reduced latency

### macOS
- Grant microphone permissions when prompted
- Use Activity Monitor to check CPU/GPU usage during transcription

### Linux
- Ensure your user is in the `audio` group: `sudo usermod -a -G audio $USER`
- For better performance, consider using `pipewire` instead of `pulseaudio`

## 📄 .env.example

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5-nano
OPENAI_MAX_TOKENS=150
OPENAI_TEMPERATURE=0.7

# Server Configuration
SERVER_HOST=127.0.0.1
SERVER_PORT=8123

# Whisper Configuration
WHISPER_MODEL=base
WHISPER_DEVICE=auto
WHISPER_COMPUTE_TYPE=float16

# Audio Configuration
SAMPLE_RATE=16000
CHANNELS=1
CHUNK_DURATION_MS=1000

# Question Detection
MIN_QUESTION_LENGTH=10
QUESTION_DETECTION_ENABLED=true

# Logging
LOG_LEVEL=INFO
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI](https://openai.com/) for their powerful language models
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for efficient speech recognition
- [FFmpeg](https://ffmpeg.org/) for robust audio processing
