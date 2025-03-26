# Finnish Voice Assistant – Aurora

This is a Finnish-language voice assistant example that demonstrates:

- Text-to-speech (TTS) via ElevenLabs
- Speech-to-text (STT) via Google Cloud Speech-to-Text or a fallback to keyboard input
- OpenAI for short, targeted intent solving (yes/no, name extraction, address extraction)
- Basic conversation flow with three questions:
  - Whether to open Notepad
  - The user's name
  - The user's address

Upon successful responses, the assistant writes the name and address into a running Notepad instance (Windows) or, as a fallback, a local text file. For demonstration purposes, it uses the `subprocess` module to open Notepad on Windows. If you are running another operating system, you may adapt the notepad opening or disable that part of the code.

## How It Works (Flow)

1. The assistant asks if the user wants to open Notepad. (Yes/No)
2. If "Yes", it opens Notepad and asks for the user's name.
3. It writes the name into Notepad (or fallback text file).
4. Then it asks for the user's address.
5. It writes the address into Notepad (or fallback text file).
6. The assistant stops after the third question or if the user says "Exit" at any point.

All speech is in Finnish, and the code tries to parse user speech with specialized prompts sent to OpenAI for short, targeted classification (yes/no, name extraction, address extraction).

## Requirements & Installation

- Python 3.7+ recommended.
- Make sure you have PortAudio installed if you intend to use pyaudio on macOS/Linux. On Windows, pyaudio usually installs without additional system steps, but can also require installing [VC++ Build Tools].
- Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies Overview

- **python-dotenv** – for loading API keys from .env
- **openai** – for GPT-based classification
- **elevenlabs** – for text-to-speech
- **google-cloud-speech** – for speech-to-text using Google Cloud
- **pyaudio** – for capturing microphone input in real-time (PortAudio wrapper)
- **audioop-lts**
- **pyautogui** (optional) – for simulating keystrokes in Notepad
- **pywin32** (alternative optional) – for sending keystrokes in Windows environments
- **logging, re, subprocess, queue, time, etc.** – standard library modules

## Environment Variables

Before running, create a `.env` file in the same folder as your script (see example below) or set environment variables directly. The following keys are needed:

- `ELEVENLABS_API_KEY` – Your ElevenLabs API key
- `OPENAI_API_KEY` – Your OpenAI API key
- `GOOGLE_APPLICATION_CREDENTIALS` – Path to your Google Cloud service account JSON file (optional, only if you want speech recognition via Google). If you don't set this, the program will fall back to keyboard input.

## Running the Program

1. Create your `.env` file (see the sample below).
2. Run the script:

```bash
python main.py
```

3. Follow the voice or text prompts. When using Google Cloud STT, speak clearly in Finnish while your microphone is active. If no valid credentials are found, the assistant falls back to a simple keyboard prompt.

## Notes and Considerations

- **Windows-Specific**: The script calls `subprocess.Popen("notepad.exe")` to open Notepad. If you're on macOS/Linux, modify or comment out those lines or replace them with something like `subprocess.Popen(["gedit"])` (for Linux) or a relevant text editor.
- **Microphone Permissions**: Ensure your OS microphone permissions allow Python to capture audio.
- **Potential Issues**:
  - pyaudio needs system-level libraries (PortAudio) installed.
  - Using TTS from ElevenLabs or GPT-based classification from OpenAI consumes credits or tokens. Remember to set up billing on these services if needed.
  - If speech recognition fails or credentials are missing, it falls back to a keyboard input method.

## Example .env File

Create a file named `.env` in your project directory with the contents:

```ini
ELEVENLABS_API_KEY=your-elevenlabs-api-key
OPENAI_API_KEY=your-openai-api-key
GOOGLE_APPLICATION_CREDENTIALS=path/to/your-google-cloud-credentials.json
```

If you do not have `GOOGLE_APPLICATION_CREDENTIALS`, just remove or comment out that line.
The assistant will then revert to the keyboard-based fallback for user input.

## Contributing

Feel free to modify the code to adapt it for your locale, language, or platform:

- Replace `subprocess.Popen("notepad.exe")` if you're not on Windows.
- Change the speech recognition or TTS service.
- Add more sophisticated conversation flow or error handling.

## License

This project is provided "as is" without any warranty. You are free to use and modify it for your own purposes. If you plan on distributing, consider the licenses of the third-party libraries involved (OpenAI, ElevenLabs, Google Cloud, etc.)
