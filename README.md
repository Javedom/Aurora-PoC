# Aurora - Finnish Taxi Booking Voice Assistant

This project implements a conversational AI assistant in Python that allows users to book a taxi using voice commands (with keyboard fallback) primarily in Finnish. It leverages OpenAI for natural language understanding, ElevenLabs for text-to-speech, and Google Cloud Speech-to-Text for voice recognition.

## Features

* **Conversational Interface:** Engages the user in a step-by-step dialogue to gather booking details.
* **Voice Input:** Utilizes Google Cloud Speech-to-Text for recognizing Finnish speech commands via the microphone.
* **Keyboard Fallback:** Provides an option to use keyboard input if voice recognition fails or is explicitly chosen (`--keyboard` flag).
* **Natural Language Understanding (NLU):** Employs OpenAI's GPT models (specifically `gpt-4o-mini` in the code) to analyze user input for addresses, time, requirements, name, and confirmation.
* **Realistic Text-to-Speech (TTS):** Uses ElevenLabs API to generate natural-sounding Finnish voice responses.
* **State Management:** Follows a defined workflow to collect pickup/destination info, time, special requirements, and passenger name.
* **Information Extraction:** Extracts structured data (addresses, time formats, requirement codes) from unstructured user input.
* **Confirmation:** Summarizes the booking details and asks for user confirmation before finalizing.
* **Simulated Booking:** On confirmation, generates a JSON payload representing the booking details and saves it to a local file (does not actually connect to a real booking API).
* **Logging:** Records conversation flow and potential errors to `assistant.log`.

## Technologies Used

* **Python 3.x**
* **OpenAI API:** For NLU and intent analysis.
* **ElevenLabs API:** For Text-to-Speech (TTS).
* **Google Cloud Speech-to-Text API:** For Speech-to-Text (STT).
* **PyAudio:** For accessing the microphone audio stream.
* **python-dotenv:** For managing API keys and environment variables.
* **Standard Libraries:** `os`, `sys`, `time`, `logging`, `re`, `json`, `datetime`, `typing`.

## Setup

1.  **Prerequisites:**
    * Python 3.7+ installed.
    * Microphone connected and configured (for voice input).
    * Speakers/headphones (for voice output).
    * Potentially system dependencies for PyAudio (like `portaudio`). On Debian/Ubuntu: `sudo apt-get install portaudio19-dev python3-pyaudio`. On macOS: `brew install portaudio; pip install pyaudio`. Windows users usually don't need extra steps if using pip.

2.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure API Keys and Credentials:**
    * **Create a `.env` file** in the project root directory.
    * Add your API keys to the `.env` file:
        ```dotenv
        OPENAI_API_KEY="your_openai_api_key_here"
        ELEVENLABS_API_KEY="your_elevenlabs_api_key_here"
        # Optional: Specify a specific ElevenLabs voice ID if desired
        # ELEVENLABS_VOICE_ID="your_preferred_voice_id" 
        
        # Path to your Google Cloud service account key file
        GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/google_cloud_key.json" 
        ```
    * **Google Cloud Credentials:**
        * You need a Google Cloud Platform project with the Speech-to-Text API enabled.
        * Create a service account and download its JSON key file.
        * Update the `GOOGLE_APPLICATION_CREDENTIALS` path in your `.env` file to point to this downloaded JSON key file. Alternatively, you can set this as a system environment variable. The script checks both the `.env` file variable and the system environment variable.

## Usage

Run the main script from your terminal:

```bash
python AuroraVGoogle.py

The assistant will start, greet you, and ask for the pickup address.  
Speak clearly into your microphone when prompted (🎤 Kuunnellaan...).  
Follow the assistant's prompts to provide booking details.

**Using Keyboard Input:**  
To force the assistant to use keyboard input instead of the microphone:

```bash
python AuroraVGoogle.py --keyboard

## Workflow

**Start:** Assistant greets the user.

**Ask Pickup:** Prompts for the pickup address (street, number, city). Handles cases where the city is initially missing.

**Ask Destination:** Prompts for the destination address. Allows the user to specify “no destination.”

**Ask Time:** Asks for the desired pickup time (accepts specific times like “14:30”, relative times like “vartin päästä”, or immediate requests like “heti”).

**Ask Additional Info:** Inquires about special requirements (pets, wheelchair) or any other notes.

**Ask Name:** Asks for the passenger’s name for the booking confirmation.

**Build Payload:** Internally constructs the JSON payload based on collected information.

**Final Confirmation:** Reads back the summarized booking details (pickup, destination, time, name, requirements, notes) and asks for confirmation (“Onko tämä kaikki oikein?”).

**End:**
- If confirmed (“yes”): Simulates booking success, saves the JSON payload to a timestamped file (e.g., `booking_YYYYMMDD_HHMMSS.json`), and ends.  
- If denied (“no” or “exit”): Cancels the booking and ends.  
- If timeout during confirmation: Cancels the booking for safety.  
- If an error occurs: Logs the error, informs the user, and ends.

---

## Logging

Detailed information about the conversation flow, API calls, intent analysis results, and errors are logged to the `assistant.log` file in the project directory.

---

## Limitations

- **Simulated Booking:** This script does not actually book a taxi. It only generates and saves a JSON representation of the booking request.  
- **Finnish Language Focus:** The prompts and NLU analysis are heavily tuned for Finnish.  
- **API Costs:** Uses paid APIs (OpenAI, ElevenLabs, Google Cloud STT). Be mindful of usage costs.  
- **Error Handling:** While basic error handling is included, complex conversational failures or API issues might require further refinement.  
- **No Address Validation/Search:** The script relies on OpenAI to parse addresses but doesn’t validate them against a real map service or perform coordinate lookups (coordinates are explicitly set to `None` in the payload).

