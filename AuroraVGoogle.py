# -*- coding: utf-8 -*-
import os
import sys
import time
import logging
import re
import subprocess
import json
import datetime
from typing import Optional, Tuple, Dict, Any, List, Union

# Third-party imports
from dotenv import load_dotenv
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import play
# Google Speech-to-Text imports
from google.cloud import speech


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('assistant.log', encoding='utf-8') # Ensure UTF-8 for logs
    ]
)
logger = logging.getLogger()

# Load environment variables
load_dotenv()

# Constants
DEFAULT_PHONE_NUMBER = "0401234567"
DEFAULT_TRIP_TYPE = "Kuluttaja"
NOW_OFFSET_MINUTES = 5
REQUIREMENTS_MAP = {
    "lemmikki": 1,
    "pets allowed": 1,
    "koira": 1, # Add synonyms
    "kissa": 1, # Add synonyms
    "pyörätuoli": 2,
    "wheelchair": 2,
}


# ----------------------------------------------------------------
# ElevenLabs Text-to-Speech (Robust initialization)
# ----------------------------------------------------------------
class ElevenLabsTTS:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = None
        # Using a known stable voice ID Aurora Fin, replace if needed
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", "YSabzCJMvEHDduIDMdwV")
        self.model_id = "eleven_flash_v2_5" # Or other preferred model
        try:
            if not api_key:
                raise ValueError("ElevenLabs API key not provided.")
            self.client = ElevenLabs(api_key=api_key)
            # Optionally test connection here if API allows (e.g., list voices)
            logger.info("ElevenLabs TTS initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ElevenLabs client: {e}")
            # Client remains None

    def speak(self, text):
        print(f"Assistant: {text}")
        if not self.client:
            logger.error("ElevenLabs client not initialized. Cannot speak.")
            # Simulate speech time based on text length if TTS fails
            time.sleep(len(text) * 0.05)
            return False
        try:
            # Generate audio stream
            audio_stream = self.client.text_to_speech.convert(
                voice_id=self.voice_id,
                optimize_streaming_latency="0",
                output_format="mp3_44100_128",
                text=text,
                model_id=self.model_id,
                # stability=0.5, # Optional parameters
                # similarity_boost=0.8,
                # style=0.0,
                # use_speaker_boost=True,
            )

            # Play the audio stream
            # Note: elevenlabs.play handles streaming playback
            play(audio_stream)

            # Optional short pause after speech
            time.sleep(0.1)
            return True
        except Exception as e:
            logger.error(f"ElevenLabs TTS error: {e}")
            return False


# ----------------------------------------------------------------
# Google Speech Recognition
# ----------------------------------------------------------------
class GoogleSpeechRecognizer:
    def __init__(self, credentials_path=None):
        self.credentials_path = credentials_path
        # Explicitly check if GOOGLE_APPLICATION_CREDENTIALS is set in the environment
        google_creds_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if credentials_path and os.path.exists(credentials_path):
            # If a path is provided via argument and exists, use it
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            logger.info(f"Using Google credentials from argument: {credentials_path}")
        elif google_creds_env and os.path.exists(google_creds_env):
            # If env var is set and file exists, use that
             logger.info(f"Using Google credentials from environment variable: {google_creds_env}")
        else:
            # If neither is valid, raise an error
            raise FileNotFoundError("Google credentials path not found or invalid via arg or GOOGLE_APPLICATION_CREDENTIALS env var.")

        try:
            self.client = speech.SpeechClient()
            logger.info("Google Speech Recognizer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Google Speech client: {e}")
            raise # Reraise the exception to halt execution if client fails

    def listen_once(self, timeout=20):
        """Records audio from the microphone and transcribes using Google Speech API."""
        print("🎤 Kuunnellaan... (Puhu nyt)")
        try:
            import pyaudio
            import audioop
        except ImportError:
            logger.error("PyAudio or audioop library not found. Cannot record audio. Please install using 'pip install pyaudio'.")
            print("Virhe: Tarvittavia äänikirjastoja (PyAudio) ei löydy.")
            return None

        # Audio recording parameters
        RATE = 16000
        CHUNK = 1024 # Buffer size
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        MAX_SECONDS = min(timeout, 60) # Max recording time
        # Silence detection parameters
        SILENCE_THRESHOLD = 500 # RMS value below which is considered silence
        SILENCE_SECONDS = 2.5 # How many seconds of silence ends recording after speech
        silence_chunks = int(SILENCE_SECONDS * RATE / CHUNK)

        p = pyaudio.PyAudio()
        stream = None
        try:
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
        except OSError as e:
            logger.error(f"Error opening audio stream: {e}. Is a microphone connected and available?")
            print("Virhe: Äänilaitteen avaaminen epäonnistui. Onko mikrofoni kytketty?")
            p.terminate()
            return None

        frames = []
        silence_count = 0
        has_speech = False
        start_time = time.time()
        print(f"Odotetaan puhetta enintään {MAX_SECONDS} sekuntia...")

        while True:
            elapsed = time.time() - start_time
            if elapsed > MAX_SECONDS:
                logger.info(f"Recording timed out after {elapsed:.1f} seconds")
                break

            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except IOError as e:
                # This usually means buffer overflow, log it but continue
                logger.warning(f"Audio input overflow/error: {e}")
                continue

            # Simple silence detection using RMS
            volume = audioop.rms(data, 2) # 2 is format width (paInt16)
            is_silence = volume < SILENCE_THRESHOLD

            if not is_silence:
                # If speech detected, reset silence counter
                has_speech = True
                silence_count = 0
            elif has_speech:
                # If silence detected *after* speech started, increment counter
                silence_count += 1

            # If enough consecutive silence chunks detected after speech, stop recording
            if has_speech and silence_count >= silence_chunks:
                logger.info(f"Detected {SILENCE_SECONDS}s silence after speech, stopping recording.")
                break

        # Stop and close the audio stream
        try:
            stream.stop_stream()
            stream.close()
        except Exception as stream_err:
             logger.error(f"Error closing audio stream: {stream_err}")
             # Continue processing if frames were captured
        finally:
            p.terminate() # Terminate PyAudio instance

        if not has_speech:
            logger.info("No speech detected during recording.")
            print("Puhetta ei havaittu.")
            return None

        print("Käsitellään puhetta...")
        audio_data = b''.join(frames)

        # Prepare audio and config for Google Speech API
        audio = speech.RecognitionAudio(content=audio_data)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code="fi-FI", # Finnish language code
            enable_automatic_punctuation=True, # Improve readability
            # model="telephony", # Optional: Specify model if needed
            # use_enhanced=True, # Optional: Use enhanced model if available/enabled
        )

        try:
            # Send audio to Google Speech API for transcription
            response = self.client.recognize(config=config, audio=audio)

            # Extract the most likely transcript
            if response.results:
                transcript = response.results[0].alternatives[0].transcript.strip()
                confidence = response.results[0].alternatives[0].confidence
                logger.info(f"Google STT Result: '{transcript}' (Confidence: {confidence:.2f})")
                if transcript:
                    print(f"Käyttäjä: {transcript}")
                    return transcript
                else:
                    print("Puhetta ei tunnistettu selvästi.")
                    return None
            else:
                print("Puhetta ei tunnistettu.")
                return None
        except Exception as e:
            logger.error(f"Error during Google Speech recognition API call: {e}")
            print("Virhe puheentunnistuksessa.")
            return None


# ----------------------------------------------------------------
# Basic Keyboard Input Recognition (Fallback)
# ----------------------------------------------------------------
class BasicKeyboardRecognizer:
    def listen_once(self, timeout=None):
        """Gets input from the keyboard."""
        # Timeout is difficult to implement reliably cross-platform for input()
        # This version ignores the timeout parameter.
        print("⌨️ Anna viestisi (tai 'lopeta'):")
        try:
            input_text = input()
            if input_text and input_text.strip():
                normalized_input = input_text.strip()
                print(f"Käyttäjä: {normalized_input}")
                if normalized_input.lower() in ["lopeta", "exit", "quit", "peruuta"]:
                    return "lopeta" # Standardize exit command
                return normalized_input
            else:
                return None # Empty input
        except EOFError:
             logger.warning("EOFError received, treating as exit.")
             return "lopeta" # Treat EOF as wanting to exit
        except Exception as e:
            logger.error(f"Error reading keyboard input: {e}")
            return None


# ----------------------------------------------------------------
# Taxi Booking Workflow (Search Removed, Coordinates Null)
# ----------------------------------------------------------------
class TaxiBookingWorkflow:
    def __init__(self, tts, recognizer, openai_client):
        self.tts: ElevenLabsTTS = tts
        self.recognizer: Union[GoogleSpeechRecognizer, BasicKeyboardRecognizer] = recognizer
        self.openai_client: OpenAI = openai_client

        # State and collected information
        self.state = "start"
        self.pickup_info = {}           # {"street": "...", "number": "...", "city": "..."}
        self.destination_info = None    # {} or {"street": "...", "number": "...", "city": "..."}
        self.pickup_time_raw = None     # Raw user input like "heti", "14:30", "vartin päästä"
        self.additional_info_notes = "" # Free text notes
        self.requirements_raw = []      # List of keywords like ["lemmikki", "pyörätuoli"]
        self.passenger_name = None      # Passenger's name
        self.generated_payload = None   # Store the final payload

    # --- Intent Analysis Functions (using OpenAI) ---
    def _analyze_with_openai(self, prompt, context_description):
        """Helper function to call OpenAI API."""
        try:
            logger.info(f"Sending {context_description} prompt to OpenAI...")
            # Using ChatCompletion endpoint
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini", # Or your preferred model
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, # Low temperature for more deterministic extraction
                max_tokens=150,  # Limit response length
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            analysis_result = response.choices[0].message.content.strip()
            logger.info(f"OpenAI {context_description} analysis result: {analysis_result}")
            return analysis_result
        except Exception as e:
            logger.error(f"Error analyzing {context_description} intent with OpenAI: {e}")
            # Check for specific API errors if possible (e.g., authentication)
            if "authentication" in str(e).lower():
                logger.error("OpenAI API authentication failed. Check your API key.")
            return None # Return None on failure

    def analyze_address_intent(self, user_input, is_destination=False):
        """
        Analyzes user input to extract address details (street, number, city) using OpenAI.
        Handles Finnish input and specific keywords like "ei määränpäätä" or "lopeta".
        Returns:
            - dict: {"street": "...", "number": "...", "city": "..."} if successful.
            - "NO_DESTINATION": If user indicates no destination (only when is_destination=True).
            - "EXIT": If user wants to exit.
            - None: If extraction fails or input is irrelevant.
        """
        context = "destination" if is_destination else "pickup"
        task_description = f"Extract the street name, street number (including letters like 'B' or '12 A 3'), and city from the following Finnish user input for a taxi booking ({context})."
        if is_destination:
            task_description += ' If the user clearly indicates they do not have or want a destination (e.g., "ei määränpäätä", "ei ole", "minne vain"), return the specific string "NO_DESTINATION".'

        prompt = f"""
        User input (Finnish): "{user_input}"

        Task: {task_description}
        Also, if the user wants to exit or cancel (e.g., "lopeta", "peruuta", "ei kiitos"), return the specific string "EXIT".

        Return ONLY the result in ONE of the following JSON formats (or specific strings "NO_DESTINATION", "EXIT"):
        1. If address found: {{"street": "[street name]", "number": "[street number]", "city": "[city name]"}}
           - Include street number details (e.g., "12 B", "5 A 1").
           - Use null if a field is not mentioned (e.g., city might be missing). Ensure city is null only if not mentioned, not if the analysis failed.
        2. If no destination indicated (only applicable when asking for destination): "NO_DESTINATION"
        3. If user wants to exit/cancel: "EXIT"
        4. If extraction fails or input is irrelevant: null

        Examples:
        Input: "Nouto osoitteesta Esimerkkikatu 12 B, Helsinki" -> Output: {{"street": "Esimerkkikatu", "number": "12 B", "city": "Helsinki"}}
        Input: "Osoite on Testitie 3" -> Output: {{"street": "Testitie", "number": "3", "city": null}}
        Input: "Nummelaan, Tuusankaari 12" -> Output: {{"street": "Tuusankaari", "number": "12", "city": "Nummela"}}
        Input: "Mä haluan mennä Helsingin keskustaan" (destination) -> Output: {{"street": null, "number": null, "city": "Helsinki"}}
        Input: "Ei määränpäätä" (destination) -> Output: "NO_DESTINATION"
        Input: "Lopeta" -> Output: "EXIT"
        Input: "En tiedä vielä" (destination) -> Output: null
        Input: "Jotain ihan muuta" -> Output: null
        """
        analysis_result = self._analyze_with_openai(prompt, f"{context} address")

        if not analysis_result: return None

        # Handle specific string outputs first
        analysis_upper = analysis_result.strip().upper()
        # Remove potential surrounding quotes for comparison
        if analysis_upper.startswith('"') and analysis_upper.endswith('"'):
             analysis_upper = analysis_upper[1:-1]

        if analysis_upper == 'NO_DESTINATION':
             # Only valid if asking for destination
             return "NO_DESTINATION" if is_destination else None
        if analysis_upper == 'EXIT':
             return "EXIT"
        if analysis_upper == 'NULL':
             return None

        # Try parsing as JSON
        try:
            # Handle potential markdown code blocks around JSON
            if analysis_result.strip().startswith("```json"):
                analysis_result = analysis_result.strip()[7:-3].strip()
            elif analysis_result.strip().startswith("```"):
                 analysis_result = analysis_result.strip()[3:-3].strip()

            address_data = json.loads(analysis_result)
            if isinstance(address_data, dict):
                # Validate basic structure and ensure at least street or city is present
                street = address_data.get("street")
                number = address_data.get("number") # Keep as string or null
                city = address_data.get("city")

                # Normalize: Convert empty strings to None for consistency
                if isinstance(street, str) and not street.strip(): street = None
                if isinstance(number, str) and not number.strip(): number = None
                if isinstance(city, str) and not city.strip(): city = None

                if street or city: # Must have at least street or city
                    return {"street": street, "number": number, "city": city}
                else:
                    logger.warning(f"OpenAI address analysis missing street and city: {analysis_result}")
                    return None
            else:
                 logger.warning(f"OpenAI address analysis was not a dict: {analysis_result}")
                 return None
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse OpenAI address analysis as JSON: {analysis_result}")
            # Fallback: Check if it contains keywords as a last resort (less reliable)
            if "NO_DESTINATION" in analysis_result and is_destination: return "NO_DESTINATION"
            if "EXIT" in analysis_result: return "EXIT"
            return None

    def analyze_city_intent(self, user_input):
        """Analyzes user input specifically to extract a city name."""
        prompt = f"""
        User input (Finnish): "{user_input}"
        Task: Extract ONLY the city name from the user's input.
        - If the user provides a city name (e.g., "Helsinki", "Se on Espoossa", "Nummelassa"), return just the city name string.
        - If the user wants to exit or cancel (e.g., "lopeta", "peruuta"), return the specific string "EXIT".
        - If no city name is found or the input is irrelevant, return null.

        Return ONLY the city name string, "EXIT", or null.

        Examples:
        Input: "Helsinki" -> Output: "Helsinki"
        Input: "Nummela" -> Output: "Nummela"
        Input: "Se on Nummelassa" -> Output: "Nummela"
        Input: "Vantaa" -> Output: "Vantaa"
        Input: "Se on Vantaalla" -> Output: "Vantaa"
        Input: "Lopeta" -> Output: "EXIT"
        Input: "En tiedä" -> Output: null
        Input: "Jossain siellä" -> Output: null
        """
        analysis_result = self._analyze_with_openai(prompt, "pickup city extraction")

        if not analysis_result or analysis_result.lower() == 'null':
            return None

        cleaned_result = analysis_result.strip('"').strip()

        if cleaned_result.upper() == "EXIT":
            return "EXIT"
        elif cleaned_result:
            # Basic check to avoid returning overly long unrelated strings
            if len(cleaned_result.split()) <= 3: # Allow for multi-word city names but not full sentences
                return cleaned_result
            else:
                logger.warning(f"City extraction result too long, likely not a city: '{cleaned_result}'")
                return None
        else:
            return None

    def analyze_time_intent(self, user_input):
        """
        Analyzes user input for pickup time using OpenAI.
        Returns the raw time expression (e.g., "14:30", "heti", "vartin päästä"), "EXIT", or None.
        """
        prompt = f"""
        User input (Finnish): "{user_input}"
        Task: Extract the requested pickup time for a taxi booking.
        - Recognize specific times (e.g., "14:30", "klo 15", "puoli neljä"). Return the normalized time string (e.g., "14:30", "15:00", "15:30").
        - Recognize keywords for immediate pickup (e.g., "heti", "nyt", "mahdollisimman pian"). Return the specific keyword "heti" for these.
        - Keep relative times like "vartin päästä", "puolen tunnin kuluttua" exactly as they are.
        - If the user wants to exit (e.g., "lopeta", "peruuta"), return the specific keyword "EXIT".
        - If no time information is mentioned or the input is irrelevant, return null.

        Return ONLY the extracted time string, "heti", "EXIT", or null.
        Ensure the output contains ONLY the result and nothing else.

        Examples:
        Input: "Heti kiitos" -> Output: "heti"
        Input: "Nyt heti" -> Output: "heti"
        Input: "Kello 14:30" -> Output: "14:30"
        Input: "Vartin päästä" -> Output: "vartin päästä"
        Input: "Noin puoli kolme iltapäivällä" -> Output: "puoli kolme iltapäivällä" # Keep descriptive for now
        Input: "Lopeta" -> Output: "EXIT"
        Input: "Ei väliä" -> Output: null
        Input: "Joo" -> Output: null
        """
        analysis_result = self._analyze_with_openai(prompt, "time extraction")

        if not analysis_result or analysis_result.lower() == 'null':
            return None

        # Clean up potential quotes and whitespace
        cleaned_result = analysis_result.strip('"').strip()

        if cleaned_result.upper() == "EXIT":
            return "EXIT"
        elif cleaned_result:
             return cleaned_result # Return the raw extracted expression or "heti"
        else:
             return None # Should not happen if null check passed, but safety first

    def analyze_structured_time_intent(self, time_expression: str) -> Optional[Dict]:
        """
        Uses OpenAI to convert a raw Finnish time expression into a structured format.
        Args:
            time_expression: The raw string from analyze_time_intent (e.g., "14:30", "heti", "vartin päästä").
        Returns:
            - Dict: {"type": "absolute", "hour": HH, "minute": MM}
            - Dict: {"type": "relative", "minutes_offset": MM}
            - Dict: {"type": "immediate"}
            - None: If parsing fails.
        """
        if not time_expression:
            return None

        # Provide current time context for better interpretation by LLM
        # Use local time for context, as users speak in local time
        now_local = datetime.datetime.now()
        current_time_str = now_local.strftime("%H:%M")
        current_date_str = now_local.strftime("%Y-%m-%d (%A)") # Add day for context

        prompt = f"""
        Parse the following Finnish time expression into a structured JSON format.
        The current local time is approximately {current_time_str} on {current_date_str}.

        Time expression: "{time_expression}"

        Task: Convert the expression into one of the following structured JSON objects.
        Return ONLY the JSON object or null.

        1. Absolute Time: {{"type": "absolute", "hour": HH, "minute": MM}}
           - Use 24-hour format (HH: 0-23, MM: 0-59 integers).
           - Interpret times like "puoli kolme iltapäivällä" correctly (e.g., 14:30).
           - Assume today unless the expression clearly indicates otherwise (e.g., "huomenna", "ylihuomenna"). If a future date is implied, you can optionally add `"day_offset": d` where d=1 for tomorrow, d=2 for day after, etc. If no offset mentioned, assume 0.

        2. Relative Time: {{"type": "relative", "minutes_offset": MM}}
           - MM is the offset in minutes from the current time (integer).
           - Examples: "vartin päästä" -> 15, "puolen tunnin kuluttua" -> 30, "tunnin päästä" -> 60.

        3. Immediate: {{"type": "immediate"}}
           - For expressions like "heti", "nyt", "mahdollisimman pian", "asap".

        4. Unparseable: null
           - If the expression doesn't represent a time or is too ambiguous.

        Return ONLY the JSON object or the literal null.

        Examples (assuming current time is 10:45):
        Input: "heti" -> Output: {{"type": "immediate"}}
        Input: "nyt" -> Output: {{"type": "immediate"}}
        Input: "14:30" -> Output: {{"type": "absolute", "hour": 14, "minute": 30}}
        Input: "klo 15" -> Output: {{"type": "absolute", "hour": 15, "minute": 0}}
        Input: "puoli neljä iltapäivällä" -> Output: {{"type": "absolute", "hour": 15, "minute": 30}}
        Input: "kymmenen aamulla" -> Output: {{"type": "absolute", "hour": 10, "minute": 0}}
        Input: "vartin päästä" -> Output: {{"type": "relative", "minutes_offset": 15}}
        Input: "puolen tunnin kuluttua" -> Output: {{"type": "relative", "minutes_offset": 30}}
        Input: "kello 2 yöllä" -> Output: {{"type": "absolute", "hour": 2, "minute": 0}} # Could imply next day if current time is late PM
        Input: "huomenna kello 9" -> Output: {{"type": "absolute", "hour": 9, "minute": 0, "day_offset": 1}}
        Input: "ei väliä" -> Output: null
        Input: "joskus iltapäivällä" -> Output: null (too ambiguous)
        """
        analysis_result = self._analyze_with_openai(prompt, "structured time parsing")

        if not analysis_result or analysis_result.lower() == 'null':
            logger.warning(f"Could not parse time expression '{time_expression}' into structured format.")
            # Fallback: if original expression was 'heti', return immediate type
            if time_expression.lower() == "heti":
                return {"type": "immediate"}
            return None

        try:
             # Handle potential markdown code blocks around JSON
            if analysis_result.strip().startswith("```json"):
                analysis_result = analysis_result.strip()[7:-3].strip()
            elif analysis_result.strip().startswith("```"):
                analysis_result = analysis_result.strip()[3:-3].strip()

            structured_time = json.loads(analysis_result)
            # Basic validation of the parsed structure
            if isinstance(structured_time, dict) and "type" in structured_time:
                type = structured_time["type"]
                if type == "absolute" and "hour" in structured_time and "minute" in structured_time:
                    # Ensure hour/minute are valid integers
                    structured_time["hour"] = int(structured_time["hour"])
                    structured_time["minute"] = int(structured_time["minute"])
                    structured_time["day_offset"] = int(structured_time.get("day_offset", 0))
                    if 0 <= structured_time["hour"] <= 23 and 0 <= structured_time["minute"] <= 59:
                        return structured_time
                elif type == "relative" and "minutes_offset" in structured_time:
                    structured_time["minutes_offset"] = int(structured_time["minutes_offset"])
                    return structured_time
                elif type == "immediate":
                    return structured_time

            logger.warning(f"OpenAI returned invalid structure for time '{time_expression}': {analysis_result}")
            return None # Invalid structure

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse or validate structured time JSON for '{time_expression}' from OpenAI result '{analysis_result}': {e}")
            # Fallback for immediate if JSON parsing failed but word was 'heti'
            if "immediate" in analysis_result or time_expression.lower() == "heti":
                 return {"type": "immediate"}
            return None

    def analyze_additional_info_intent(self, user_input):
        """
        Analyzes user input for special requirements (pets, wheelchair) and other notes using OpenAI.
        Returns a dict: {"requirements": ["kw1", "kw2"], "notes": "...", "intent": "continue" | "exit" | "none"}
        """
        prompt = f"""
        User input (Finnish): "{user_input}"
        Task: Analyze the user's response regarding special requirements or additional notes for a taxi booking.
        - Identify keywords for common requirements: "lemmikki", "koira", "kissa" (for pet), "pyörätuoli" (for wheelchair). Add the IDENTIFIED keyword (e.g., "koira", "pyörätuoli") to a 'requirements' list. Standardize pet keywords to "lemmikki".
        - Extract any other relevant notes the user provides (e.g., "minulla on paljon matkatavaroita", "ovi on sisäpihalla"). Put this in the 'notes' string. Ignore simple confirmations like "ei mitään".
        - Determine the overall intent:
            - "continue": User provided requirements/notes OR explicitly said none are needed (e.g., "ei mitään", "ei ole").
            - "exit": User wants to stop or cancel (e.g., "lopeta", "peruuta").
            - "none": Input is irrelevant or just a simple acknowledgement (e.g., "selvä", "joo").

        Return ONLY a JSON object with the keys "requirements" (list of strings), "notes" (string), and "intent" (string: "continue", "exit", or "none").

        Examples:
        Input: "Joo, mulla on koira mukana" -> Output: {{"requirements": ["lemmikki"], "notes": "koira mukana", "intent": "continue"}}
        Input: "Ei mitään erityistä" -> Output: {{"requirements": [], "notes": "", "intent": "continue"}}
        Input: "Tarvitsen pyörätuolipaikan ja apua kantamisessa" -> Output: {{"requirements": ["pyörätuoli"], "notes": "Tarvitsee apua kantamisessa", "intent": "continue"}}
        Input: "Minulla on kissa ja paljon laukkuja" -> Output: {{"requirements": ["lemmikki"], "notes": "kissa ja paljon laukkuja", "intent": "continue"}}
        Input: "Lopeta" -> Output: {{"requirements": [], "notes": "", "intent": "exit"}}
        Input: "Peruuta tilaus" -> Output: {{"requirements": [], "notes": "", "intent": "exit"}}
        Input: "Selvä" -> Output: {{"requirements": [], "notes": "", "intent": "none"}}
        Input: "Onks kaikki ok?" -> Output: {{"requirements": [], "notes": "", "intent": "none"}}
        Input: "Mulla on aika paljon matkatavaroita." -> Output: {{"requirements": [], "notes": "aika paljon matkatavaroita", "intent": "continue"}}
        """
        analysis_result = self._analyze_with_openai(prompt, "additional info")

        # Default structure in case of failure
        default_output = {"requirements": [], "notes": "", "intent": "none"}

        if not analysis_result:
            return default_output

        try:
            # Handle potential markdown code blocks around JSON
            if analysis_result.strip().startswith("```json"):
                analysis_result = analysis_result.strip()[7:-3].strip()
            elif analysis_result.strip().startswith("```"):
                 analysis_result = analysis_result.strip()[3:-3].strip()

            data = json.loads(analysis_result)
            # Validate the received structure
            if isinstance(data, dict) and "intent" in data:
                 # Ensure keys exist and have correct types
                 reqs = data.get("requirements", [])
                 notes = data.get("notes", "")
                 intent = data.get("intent")

                 # Basic type validation
                 if not isinstance(reqs, list): reqs = []
                 if not isinstance(notes, str): notes = ""
                 if intent not in ["continue", "exit", "none"]: intent = "none" # Default to none if invalid

                 # Standardize pet keywords
                 std_reqs = []
                 for r in reqs:
                     if r.lower() in ["koira", "kissa", "lemmikki"]:
                         if "lemmikki" not in std_reqs:
                             std_reqs.append("lemmikki")
                     else:
                         if r not in std_reqs:
                            std_reqs.append(r)


                 return {"requirements": std_reqs, "notes": notes, "intent": intent}
            else:
                 logger.warning(f"OpenAI additional info analysis did not return a valid dict: {analysis_result}")
                 # Fallback: Check for keywords if parsing failed
                 intent_val = "exit" if "exit" in analysis_result.lower() else "none"
                 return {"requirements": [], "notes": "", "intent": intent_val}

        except json.JSONDecodeError:
             # WARNING log moved below to avoid duplicate logging when the original analysis result itself is logged
             # logger.warning(f"Failed to parse OpenAI additional info analysis as JSON: {analysis_result}") # Duplicates previous log
             # Fallback: Check for keywords manually (less reliable)
             reqs_found_raw = [kw for kw in ["lemmikki", "koira", "kissa", "pyörätuoli"] if kw in analysis_result.lower()]
             reqs_found_std = []
             for r in reqs_found_raw:
                 if r.lower() in ["koira", "kissa", "lemmikki"]:
                     if "lemmikki" not in reqs_found_std:
                         reqs_found_std.append("lemmikki")
                 elif r.lower() == "pyörätuoli":
                     if "pyörätuoli" not in reqs_found_std:
                        reqs_found_std.append("pyörätuoli")

             intent_val = "exit" if ("lopeta" in analysis_result.lower() or "peruuta" in analysis_result.lower()) else "none"
             # Try to capture notes if it doesn't seem like just keywords or exit
             # Check if the result contains non-requirement words
             words_in_result = set(re.findall(r'\b\w+\b', analysis_result.lower()))
             requirement_words = {"lemmikki", "koira", "kissa", "pyörätuoli"}
             other_words = words_in_result - requirement_words - {"lopeta", "peruuta"}

             notes_val = ""
             # Heuristic: if there are other words and it's not just "ei mitään", consider it a note.
             if other_words and not any(neg in analysis_result.lower() for neg in ["ei mitään", "ei ole"]):
                  notes_val = analysis_result.strip()

             # If user said "ei mitään", treat as continue intent
             if any(neg in analysis_result.lower() for neg in ["ei mitään", "ei ole"]):
                 intent_val = "continue"
                 notes_val = "" # Don't keep "ei mitään" as a note
             # If requirements or notes were found (and not exit), intent is continue
             elif reqs_found_std or (notes_val and intent_val != "exit"):
                 intent_val = "continue"

             # If only notes were found, ensure the original full input is used as notes
             if not reqs_found_std and intent_val == "continue" and notes_val:
                  notes_val = analysis_result.strip()


             return {"requirements": reqs_found_std, "notes": notes_val, "intent": intent_val}


    def analyze_name_intent(self, user_input):
        """
        Analyzes user input to extract a name using OpenAI.
        Returns:
            - ("name", "extracted_name"): If name found.
            - ("exit", None): If user wants to exit.
            - ("unknown", None): If no name or irrelevant input.
        """
        prompt = f"""
        User input (Finnish): "{user_input}"
        Task: Extract the person's name if they are providing one for a taxi booking.
        - Focus on extracting just the name. Common patterns are "Nimeni on [Name]", "Olen [Name]", or just "[Name]".
        - Do NOT include introductory phrases like "Nimeni on" or "Olen" in the extracted name.
        - If the user wants to exit or cancel (e.g., "lopeta", "peruuta"), return the specific string "EXIT".
        - If no name is provided, or the input is irrelevant (e.g., "Ei kiitos", "Voit käyttää numeroa"), return the specific string "UNKNOWN".

        Return ONLY ONE of the following:
        1. "NAME: [Extracted Name]" (e.g., "NAME: Matti Virtanen")
        2. "EXIT"
        3. "UNKNOWN"

        Ensure the output is exactly one line and matches one of these formats precisely.

        Examples:
        Input: "Nimeni on Matti Virtanen" -> Output: "NAME: Matti Virtanen"
        Input: "Olen Liisa Jokinen" -> Output: "NAME: Liisa Jokinen"
        Input: "Pekka" -> Output: "NAME: Pekka"
        Input: "Lopeta" -> Output: "EXIT"
        Input: "Ei kiitos, käytä numeroa" -> Output: "UNKNOWN"
        Input: "En halua sanoa" -> Output: "UNKNOWN"
        Input: "Selvä" -> Output: "UNKNOWN"
        Input: "En halua antaa nimeä" -> Output: "UNKNOWN"
        """
        analysis_result = self._analyze_with_openai(prompt, "name extraction")

        if not analysis_result:
            return "unknown", None

        # Check the response format strictly
        if analysis_result.startswith("NAME:"):
            extracted_name = analysis_result[len("NAME:"):].strip()
            if extracted_name: # Ensure name isn't empty
                return "name", extracted_name
            else:
                logger.warning("OpenAI returned 'NAME:' but name was empty.")
                return "unknown", None
        elif analysis_result == "EXIT":
            return "exit", None
        elif analysis_result == "UNKNOWN":
            return "unknown", None
        else:
            # If the format is wrong, log it and treat as unknown
            logger.warning(f"OpenAI name analysis returned unexpected format: {analysis_result}")
            # ADDED: Try to extract name even if format is wrong, as a fallback
            potential_name = analysis_result.strip()
            # Very basic check: not EXIT/UNKNOWN and contains likely name characters
            if potential_name and potential_name != "EXIT" and potential_name != "UNKNOWN" and len(potential_name.split()) <= 3:
                 logger.info(f"Treating unexpected format as name (fallback): '{potential_name}'")
                 return "name", potential_name
            return "unknown", None


    def analyze_yes_no_intent(self, user_input):
        """
        Analyzes user input for yes/no/exit confirmation using OpenAI.
        Returns: "yes", "no", "exit", or "unknown".
        """
        prompt = f"""
        Analyze the following Finnish user input to determine if it's an affirmative ('yes'), negative ('no'), or an explicit exit/cancel command in the context of confirming a taxi booking.

        User input: "{user_input}"

        Consider common Finnish expressions:
        - Affirmative ("kyllä", "joo", "juu", "okei", "sopii", "on oikein", "vahvistan"): Return "yes".
        - Negative ("ei", "en", "ei ole", "väärin", "ei käy"): Return "no".
        - Exit/Cancel ("lopeta", "peruuta", "peru"): Return "exit".

        Return EXACTLY one word: "yes", "no", "exit", or "unknown".
        Do not include any other text or explanation.

        Examples:
        Input: "Kyllä on" -> Output: yes
        Input: "Joo sopii" -> Output: yes
        Input: "Ei ole oikein" -> Output: no
        Input: "Peruuta koko homma" -> Output: exit
        Input: "Lopetetaan" -> Output: exit
        Input: "Mitä sanoitkaan?" -> Output: unknown
        Input: "Odota hetki" -> Output: unknown
        """
        analysis_result = self._analyze_with_openai(prompt, "yes/no confirmation")

        if not analysis_result:
            return "unknown"

        # Normalize the result (lowercase, strip whitespace)
        intent = analysis_result.strip().lower()

        # Check against expected outputs
        if intent == "yes":
            return "yes"
        elif intent == "no":
            return "no"
        elif intent == "exit":
            return "exit"
        else:
            logger.warning(f"OpenAI yes/no analysis returned unexpected value: '{analysis_result}', treating as 'unknown'.")
            # Optional: More robust checking if LLM fails sometimes
            # Fallback based on keywords
            input_lower = user_input.lower()
            if any(word in input_lower for word in ["kyllä", "joo", "juu", "okei", "sopii", "vahvista", "oikein"]): return "yes"
            if any(word in input_lower for word in ["ei", "en ", "väärin", "ei käy"]): return "no"
            if any(word in input_lower for word in ["lopeta", "peruuta", "peru"]): return "exit"
            return "unknown"


    # --- JSON Payload Construction (Coordinates=None) ---
    def build_json_payload(self):
        """
        Constructs the final JSON payload dictionary for the API.
        Uses collected information and sets coordinates to None.
        Returns the payload dictionary or None if validation fails.
        """
        logger.info("Building JSON payload...")

        # --- Basic Payload Validation (Moved earlier) ---
        # Ensure critical info exists *before* calculating time etc.
        pickup = self.pickup_info
        passenger_name = self.passenger_name

        if not pickup or not pickup.get("street") or not pickup.get("city"):
            logger.error("Payload Validation Error: Missing required pickup information (Street and City).")
            # Return None *before* attempting further processing
            return None # Indicate failure

        if not passenger_name:
             # Use default phone number as fallback if name wasn't captured but flow proceeded
             logger.warning("Passenger name missing, using default phone number.")
             passenger_name = DEFAULT_PHONE_NUMBER


        # --- Calculate Pickup Time ---
        pickup_datetime_utc = None
        pickup_time_str = self.pickup_time_raw or "heti" # Default to 'heti' if not provided

        logger.info(f"Attempting to parse raw time expression: '{pickup_time_str}'")
        structured_time = self.analyze_structured_time_intent(pickup_time_str)
        logger.info(f"Structured time analysis result: {structured_time}")

        # Use current UTC time as the reference for relative/immediate calculations
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        if structured_time:
            time_type = structured_time.get("type")
            try:
                if time_type == "immediate":
                    pickup_datetime_utc = now_utc + datetime.timedelta(minutes=NOW_OFFSET_MINUTES)
                    logger.info(f"Immediate pickup requested. Calculated UTC time: {pickup_datetime_utc}")
                elif time_type == "relative":
                    offset = int(structured_time.get("minutes_offset", NOW_OFFSET_MINUTES))
                    pickup_datetime_utc = now_utc + datetime.timedelta(minutes=offset)
                    logger.info(f"Relative pickup requested (+{offset} min). Calculated UTC time: {pickup_datetime_utc}")
                elif time_type == "absolute":
                    hour = int(structured_time.get("hour", now_utc.hour))
                    minute = int(structured_time.get("minute", now_utc.minute))
                    day_offset = int(structured_time.get("day_offset", 0))

                    # Construct the datetime object based on UTC date + offset
                    target_date_utc = (now_utc + datetime.timedelta(days=day_offset)).date()
                    # Create a timezone-aware UTC datetime object directly
                    pickup_datetime_utc_candidate = datetime.datetime(
                        target_date_utc.year, target_date_utc.month, target_date_utc.day,
                        hour, minute, tzinfo=datetime.timezone.utc
                    )

                    # If the absolute time for today/offset day is significantly in the past,
                    # and no day offset was explicitly given, assume it's for the *next* day (or day after offset).
                    # Add a small buffer (e.g., 15 min) to avoid issues near midnight.
                    if day_offset == 0 and pickup_datetime_utc_candidate < (now_utc - datetime.timedelta(minutes=15)):
                        logger.info(f"Absolute time {hour:02d}:{minute:02d} is in the past for today. Assuming tomorrow.")
                        pickup_datetime_utc = pickup_datetime_utc_candidate + datetime.timedelta(days=1)
                    else:
                         pickup_datetime_utc = pickup_datetime_utc_candidate
                    logger.info(f"Absolute pickup requested ({hour:02d}:{minute:02d}, offset: {day_offset} days). Calculated UTC time: {pickup_datetime_utc}")

            except (ValueError, TypeError) as e:
                 logger.error(f"Error processing structured time {structured_time}: {e}. Falling back.")
                 pickup_datetime_utc = None # Fallback needed

        # Fallback if structured parsing failed or wasn't possible
        if not pickup_datetime_utc:
             logger.warning(f"Could not determine specific pickup time from '{pickup_time_str}'. Defaulting to UTC now + {NOW_OFFSET_MINUTES} min.")
             pickup_datetime_utc = now_utc + datetime.timedelta(minutes=NOW_OFFSET_MINUTES)

        # Format to ISO 8601 with Z for UTC
        pickup_time_iso = pickup_datetime_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z" # Milliseconds + Z

        # --- Assemble Payload ---
        payload = {
          "TripType": DEFAULT_TRIP_TYPE,
          "PickupTime": pickup_time_iso,
          "Pickup": {
            "Place": "", # Assuming Place is not captured, keep default
            "Street": self.pickup_info.get("street"),
            "StreetNumber": self.pickup_info.get("number"),
            "Apartment": "", # Not explicitly captured, default to empty
            "PostalCode": None, # Not captured without search
            "City": self.pickup_info.get("city"),
            "Latitude": None, # Explicitly None as requested
            "Longitude": None, # Explicitly None as requested
          },
          "Destination": None, # Initialize as None
          "AdditionalInfo": self.additional_info_notes,
          "VehicleType": "", # Not captured, default to empty
          "Requirements": [], # Initialize, will be populated below
          "Passenger": {
            "Id": 0, # Default ID
            "Name": passenger_name, # Use validated/defaulted name
            "Phonenumber": DEFAULT_PHONE_NUMBER # Use default phone number
          },
          "PaymentMethod": "", # Not captured, default to empty
          "PriceIncVat": 0, # Default
          "DiscountId": 0   # Default
        }

        # Fill Destination if provided (and not explicitly 'NO_DESTINATION')
        if self.destination_info and isinstance(self.destination_info, dict): # Check if it's a dict
             payload["Destination"] = {
                "Place": "",
                "Street": self.destination_info.get("street"),
                "StreetNumber": self.destination_info.get("number"),
                "Apartment": "",
                "PostalCode": None,
                "City": self.destination_info.get("city"),
                "Latitude": None, # Explicitly None
                "Longitude": None, # Explicitly None
             }
        # No need for explicit else, it's initialized to None


        # Process Requirements (Map keywords to API codes)
        processed_requirements = []
        for req_keyword in self.requirements_raw:
            # Use the standardized keywords stored earlier
            req_code = REQUIREMENTS_MAP.get(req_keyword.lower())
            if req_code is not None:
                if req_code not in processed_requirements: # Avoid duplicates
                    processed_requirements.append(req_code)
            else:
                logger.warning(f"Unmapped requirement keyword: '{req_keyword}'")

        # Ensure requirements list follows API spec (e.g., [0] if empty, remove 0 if others present)
        if not processed_requirements:
            payload["Requirements"] = [0] # API requirement for no special needs
        else:
            # Remove 0 if other codes are present (assuming 0 means "none")
            unique_reqs = sorted(processed_requirements)
            if len(unique_reqs) > 1 and 0 in unique_reqs:
                 payload["Requirements"] = [r for r in unique_reqs if r != 0]
            else:
                 payload["Requirements"] = unique_reqs

        # --- Final Validation (Redundant due to earlier check, but safe) ---
        # pickup_payload = payload.get("Pickup") # Use variable defined earlier
        # passenger_payload = payload.get("Passenger") # Use variable defined earlier
        # if not pickup_payload or not pickup_payload.get("Street") or not pickup_payload.get("City"):
        #     logger.error("Payload Validation Error: Missing required pickup information (Street and City) at final stage.")
        #     return None # Indicate failure
        # if not passenger_payload or not passenger_payload.get("Name"):
        #      logger.error("Payload Validation Error: Missing passenger name at final stage.")
        #      return None # Indicate failure

        logger.info("Payload built successfully.")
        self.generated_payload = payload # Store the generated payload
        return payload


    # --- Main Workflow Logic ---
    def run(self):
        """
        Runs the main state machine for the taxi booking conversation.
        Handles user interaction, intent analysis, and payload generation.
        Returns True to continue, False when the workflow should terminate.
        """
        try:
            # --- Initial State / Greeting ---
            if self.state == "start":
                self.tts.speak("Hei, tämä on taksin tilauspalvelu.")
                self.state = "ask_pickup"
                self.tts.speak("Mistä osoitteesta noudetaan?")
                return True # Continue workflow

            # --- Get User Input (if needed for current state) ---
            user_input = None
            # Added 'ask_pickup_city' to states requiring input
            if self.state in ["ask_pickup", "ask_pickup_city", "ask_destination", "ask_time", "ask_additional", "ask_name", "final_confirmation"]:
                user_input = self.recognizer.listen_once(timeout=30) # 30-second listen timeout

                # Handle no input / timeout
                if not user_input:
                    if self.state == "final_confirmation":
                         # If timeout during final confirmation, cancel for safety
                         self.tts.speak("En kuullut vastausta. Perutaan tilaus varmuuden vuoksi.")
                         self.state = "end_cancel"
                         return False # End workflow
                    else:
                         # Ask user to repeat if timeout during info gathering
                         self.tts.speak("En kuullut vastausta. Voitko toistaa?")
                         return True # Repeat current state

                # Standardize exit command from recognizer
                if isinstance(user_input, str) and user_input.lower() == "lopeta":
                    user_input = "lopeta" # Ensure consistent exit keyword for analysis

            # === State Machine Logic ===
            if self.state == "ask_pickup":
                # Analyze input for pickup address
                address_result = self.analyze_address_intent(user_input, is_destination=False)
                logger.info(f"Pickup Address Intent Analysis: {address_result}")

                if isinstance(address_result, dict):
                    street = address_result.get("street")
                    number = address_result.get("number")
                    city = address_result.get("city")

                    # Check if street is present but city is missing
                    if street and not city:
                        self.pickup_info = {"street": street, "number": number, "city": None} # Store partial info
                        # Construct prompt for missing city
                        addr_part = f"{street}{f', {number}' if number else ''}" # Combine street and number if available
                        self.tts.speak(f"Sain osoitteen {addr_part}. Missä kaupungissa tämä sijaitsee?")
                        self.state = "ask_pickup_city" # Transition to ask for city
                    # Check if both street and city are present
                    elif street and city:
                        self.pickup_info = address_result # Store full info
                        pickup_desc_parts = [street, number, city]
                        pickup_desc = ", ".join(filter(None, pickup_desc_parts))
                        self.tts.speak(f"Selvä, nouto osoitteesta {pickup_desc}.")
                        self.state = "ask_destination"
                        self.tts.speak("Mikä on matkan määränpää? Voit myös sanoa 'ei määränpäätä'.")
                    # Handle other cases (e.g., only city found, or insufficient info)
                    else:
                        self.tts.speak("En valitettavasti saanut selvää nouto-osoitteesta. Voisitko sanoa kadun, numeron ja kaupungin?")
                        # Stay in ask_pickup state

                elif address_result == "EXIT":
                    self.state = "end_cancel" # Move to cancel state
                else: # Includes None or unexpected results
                    self.tts.speak("En valitettavasti saanut selvää nouto-osoitteesta. Voisitko sanoa kadun, numeron ja kaupungin uudelleen?")
                    # Stay in ask_pickup state

            # +++ NEW STATE: ask_pickup_city +++
            elif self.state == "ask_pickup_city":
                city_result = self.analyze_city_intent(user_input)
                logger.info(f"Pickup City Intent Analysis: {city_result}")

                if city_result and city_result != "EXIT":
                    self.pickup_info['city'] = city_result # Update stored info with the city
                    # Confirm the full address now
                    full_pickup_parts = [self.pickup_info.get('street'), self.pickup_info.get('number'), self.pickup_info.get('city')]
                    full_pickup_desc = ", ".join(filter(None, full_pickup_parts))
                    self.tts.speak(f"Selvä, kaupunki {city_result}. Nouto siis osoitteesta {full_pickup_desc}.")
                    self.state = "ask_destination" # Proceed to next step
                    self.tts.speak("Mikä on matkan määränpää? Voit myös sanoa 'ei määränpäätä'.")
                elif city_result == "EXIT":
                    self.state = "end_cancel"
                else: # Includes None or failure
                    self.tts.speak("En saanut selvää kaupungista. Voisitko sanoa sen uudelleen?")
                    # Stay in ask_pickup_city state

            elif self.state == "ask_destination":
                # Analyze input for destination address
                address_result = self.analyze_address_intent(user_input, is_destination=True)
                logger.info(f"Destination Address Intent Analysis: {address_result}")

                if isinstance(address_result, dict):
                    self.destination_info = address_result
                    dest_desc_parts = [self.destination_info.get('street'), self.destination_info.get('number'), self.destination_info.get('city')]
                    dest_desc = ", ".join(filter(None, dest_desc_parts))
                    self.tts.speak(f"Selvä, määränpää on {dest_desc}.")
                    self.state = "ask_time"
                    self.tts.speak("Moneltako nouto olisi? Voit sanoa kellonajan tai 'heti'.")
                elif address_result == "NO_DESTINATION":
                    self.destination_info = None # Store None to indicate no destination
                    self.tts.speak("Selvä, ei määränpäätä.")
                    self.state = "ask_time"
                    self.tts.speak("Moneltako nouto olisi? Voit sanoa kellonajan tai 'heti'.")
                elif address_result == "EXIT":
                    self.state = "end_cancel"
                else:
                    self.tts.speak("En saanut selvää määränpäästä. Voitko sanoa sen uudelleen tai sanoa 'ei määränpäätä'?")
                    # Stay in ask_destination state

            elif self.state == "ask_time":
                # Analyze input for pickup time
                time_result = self.analyze_time_intent(user_input)
                logger.info(f"Time Intent Analysis: {time_result}")

                if time_result == "EXIT":
                    self.state = "end_cancel"
                elif time_result:
                    self.pickup_time_raw = time_result
                    # Give simple confirmation based on raw input
                    time_desc = "heti" if self.pickup_time_raw.lower() == "heti" else f"noin {self.pickup_time_raw}"
                    self.tts.speak(f"Selvä, nouto {time_desc}.")
                    self.state = "ask_additional"
                    self.tts.speak("Onko tilaukseen liittyen erityistarpeita, kuten lemmikkiä tai pyörätuolia, tai muuta huomioitavaa?")
                else:
                    self.tts.speak("En ymmärtänyt kellonaikaa. Voitko sanoa ajan muodossa 'kello 14:30', 'vartin päästä' tai 'heti'?")
                    # Stay in ask_time state

            elif self.state == "ask_additional":
                 # Analyze input for additional info and requirements
                 add_info_result = self.analyze_additional_info_intent(user_input)
                 # Log the raw result for debugging potential JSON issues
                 logger.info(f"Raw Additional Info Analysis Result: {add_info_result}")

                 intent = add_info_result.get("intent", "none")

                 if intent == "exit":
                     self.state = "end_cancel"
                 else:
                      # Store requirements and notes even if intent is 'none' (might be relevant later)
                      self.requirements_raw = add_info_result.get("requirements", [])
                      self.additional_info_notes = add_info_result.get("notes", "")

                      # Give confirmation only if something specific was mentioned or confirmed none
                      if intent == "continue":
                            if self.requirements_raw or self.additional_info_notes:
                                # Construct confirmation message
                                req_text = ""
                                if self.requirements_raw:
                                     req_text += f"Erityistarpeet ({', '.join(self.requirements_raw)}) kirjattu."
                                notes_text = ""
                                if self.additional_info_notes:
                                     notes_text += f" Lisätieto '{self.additional_info_notes}' kirjattu."

                                confirmation_speech = f"Selvä. {req_text}{notes_text}".strip()
                                self.tts.speak(confirmation_speech)
                            else:
                                # If intent was 'continue' but reqs/notes are empty, means user said "no" or "none"
                                self.tts.speak("Selvä, ei erityistarpeita tai lisätietoja.")
                      # If intent is 'none', don't give confirmation, just proceed silently

                      # Proceed to ask for name
                      self.state = "ask_name"
                      self.tts.speak("Hyvä. Vielä tilauksen vahvistukseksi, kerrotko nimesi?")

            elif self.state == "ask_name":
                # Analyze input for passenger name
                intent, name = self.analyze_name_intent(user_input)
                logger.info(f"Name Intent Analysis: intent='{intent}', name='{name}'")

                if intent == "name" and name:
                    self.passenger_name = name
                    self.tts.speak(f"Kiitos, {self.passenger_name}.")
                    # Directly move to build payload, search is removed
                    self.state = "build_payload" # Transition to build payload
                    # self.tts.speak("Valmistellaan tilausta...") # Moved confirmation to after successful build
                elif intent == "exit":
                    self.state = "end_cancel"
                else: # Includes "unknown"
                    # Ask again, suggest using phone number as alternative
                    self.tts.speak("En saanut selvää nimestäsi. Voisitko kertoa sen uudelleen? Voimme myös käyttää puhelinnumeroasi tilauksen tunnisteena.")
                    # Stay in ask_name state

            # --- Build Payload State (Entered after name is confirmed) ---
            elif self.state == "build_payload":
                 # Attempt to build the payload
                 # Payload is now stored in self.generated_payload by build_json_payload
                 if self.build_json_payload():
                      # Payload built successfully
                      logger.info(f"Payload generated, proceeding to final confirmation.")
                      self.tts.speak("Valmistellaan tilausta...") # Speak confirmation now

                      # Print payload to log/console for debugging
                      print(f"\n--- Generated Payload (for confirmation) ---\n{json.dumps(self.generated_payload, indent=2, ensure_ascii=False)}\n-------------------------\n")

                      # Construct confirmation message for the user
                      pickup = self.generated_payload.get('Pickup', {})
                      pickup_desc_parts = [pickup.get('Street'), pickup.get('StreetNumber'), pickup.get('City')]
                      pickup_desc = ", ".join(filter(None, pickup_desc_parts))

                      # Try to format time nicely from ISO string for confirmation
                      time_iso = self.generated_payload.get('PickupTime', '')
                      time_desc = "noin " + self.pickup_time_raw if self.pickup_time_raw and self.pickup_time_raw.lower() != "heti" else "heti pian" # Use "heti pian" for immediate
                      try:
                           # Parse ISO string back to datetime object (assuming Z means UTC)
                           pickup_dt_utc = datetime.datetime.fromisoformat(time_iso.replace('Z', '+00:00'))
                           # Convert to local timezone for display
                           pickup_dt_local = pickup_dt_utc.astimezone(tz=None) # Uses system's local timezone
                           # Format nicely (e.g., "klo 14:30" or "27.03. klo 10:00")
                           # Check if it's within the next ~24 hours to decide format
                           now_local = datetime.datetime.now(pickup_dt_local.tzinfo)
                           if now_local <= pickup_dt_local < (now_local + datetime.timedelta(hours=23)):
                                time_desc = pickup_dt_local.strftime("klo %H:%M") # Time only if today/soon
                           else:
                                time_desc = pickup_dt_local.strftime("%d.%m. klo %H:%M") # Date and time if further out
                      except ValueError:
                           logger.warning(f"Could not parse ISO time '{time_iso}' for confirmation message.")
                           # Use the fallback time_desc already set

                      dest = self.generated_payload.get('Destination')
                      dest_desc = "ei määränpäätä"
                      if dest: # Check if destination exists in payload
                          dest_desc_parts = [dest.get('Street'), dest.get('StreetNumber'), dest.get('City')]
                          dest_desc = ", ".join(filter(None, dest_desc_parts))
                          if not dest_desc: # Handle case where dest dict exists but is empty
                               dest_desc = "ei määriteltyä osoitetta"


                      passenger_name_conf = self.generated_payload.get('Passenger', {}).get('Name', DEFAULT_PHONE_NUMBER)
                      reqs = self.generated_payload.get('Requirements', [])
                      req_desc = ""
                      # Check if requirements list is not empty and doesn't just contain 0
                      if reqs and reqs != [0]:
                           req_names = []
                           for code in reqs:
                                # Find key(s) for the code in REQUIREMENTS_MAP
                                keys = [k for k, v in REQUIREMENTS_MAP.items() if v == code]
                                if keys:
                                    # Prefer shorter names if multiple map to same code
                                    keys.sort(key=len)
                                    req_names.append(keys[0]) # Add the first (shortest) keyword
                           if req_names:
                              req_desc = f"Erityistarpeet: {', '.join(req_names)}. "

                      notes_conf = self.generated_payload.get('AdditionalInfo', '')
                      notes_desc = f"Lisätieto: {notes_conf}. " if notes_conf else ""


                      confirmation_msg = (
                          f"Vahvistetaan tilaus: Nouto osoitteesta {pickup_desc}, "
                          f"aika {time_desc}. "
                          f"Määränpää {dest_desc}. "
                          f"{req_desc}" # Add requirements description
                          f"{notes_desc}" # Add notes description
                          f"Tilaus nimellä {passenger_name_conf}. "
                          f"Onko tämä kaikki oikein?"
                      )
                      self.tts.speak(confirmation_msg)
                      self.state = "final_confirmation" # Move to wait for yes/no
                 else:
                      # Payload building failed (error already logged in build_json_payload)
                      self.tts.speak("Valitettavasti tilaustietojen kokoamisessa tapahtui virhe. Nouto-osoitteesta puuttuu tietoja. Yritäthän tilausta uudelleen myöhemmin.")
                      self.state = "end_error" # End with error


            # --- Final Confirmation State ---
            elif self.state == "final_confirmation":
                # Analyze user's yes/no/exit response
                intent = self.analyze_yes_no_intent(user_input)
                logger.info(f"Final Confirmation Intent Analysis: {intent}")

                if intent == "yes":
                    self.tts.speak("Kiitos! Tilaus on vahvistettu.")
                    print("\n[SYSTEM] --- TILAUS VAHVISTETTU (SIMULOITU) ---")
                    # --- Save Payload to File ---
                    try:
                        payload_filename = f"booking_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
                        with open(payload_filename, 'w', encoding='utf-8') as f:
                            json.dump(self.generated_payload, f, indent=2, ensure_ascii=False)
                        print(f"[SYSTEM] Payload saved to: {payload_filename}")
                        logger.info(f"Final payload saved to {payload_filename}")
                    except Exception as e:
                         print(f"[SYSTEM] Error saving payload to file: {e}")
                         logger.error(f"Failed to save payload to file: {e}")
                    # --- End Simulation ---
                    self.state = "end_success" # End successfully
                elif intent == "no" or intent == "exit":
                    self.tts.speak("Selvä, tilaus peruttu.")
                    self.state = "end_cancel" # End with cancellation
                else: # Includes "unknown"
                    self.tts.speak("Anteeksi, en ymmärtänyt vahvistusta. Ovatko tiedot oikein? Vastaa 'kyllä' tai 'ei'.")
                    # Stay in final_confirmation state

            # --- Check for End States ---
            if self.state in ["end_success", "end_cancel", "end_error"]:
                logger.info(f"Workflow finished with state: {self.state}")
                return False # Signal workflow termination

            # If not ended, continue the loop
            return True

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Exiting workflow.")
            try: self.tts.speak("Ohjelma keskeytettiin.")
            except: pass
            self.state = "end_error"
            return False # End workflow
        except Exception as e:
            # Catch any unexpected errors during state execution
            logger.exception(f"Critical error during workflow execution in state '{self.state}': {e}")
            try: self.tts.speak("Valitettavasti tapahtui odottamaton virhe. Lopetetaan.")
            except: pass
            self.state = "end_error"
            return False # End workflow


# ----------------------------------------------------------------
# Main Application Runner
# ----------------------------------------------------------------
def main():
    print("--- Suomalainen Taksin Tilausavustaja ---")
    print("=========================================")
    print("Ladataan asetuksia ja alustetaan komponentteja...")

    tts = None # Define outside try block for potential use in except block

    try:
        # Load API keys and configuration from .env file or environment variables
        load_dotenv()
        elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        # Google Credentials can be path or let SDK find default
        google_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        # --- Initialize Components ---
        # Initialize TTS (ElevenLabs)
        if not elevenlabs_api_key:
            print("VAROITUS: ELEVENLABS_API_KEY puuttuu. Puhe ei toimi.")
            logger.warning("ELEVENLABS_API_KEY missing. TTS disabled.")
            # Provide a dummy TTS object that just prints if API key is missing
            class DummyTTS:
                def speak(self, text): print(f"Assistant (TTS Disabled): {text}")
            tts = DummyTTS()
        else:
            tts = ElevenLabsTTS(api_key=elevenlabs_api_key)

        # Initialize OpenAI Client
        if not openai_api_key:
            print("VIRHE: OPENAI_API_KEY puuttuu. Avustaja ei voi toimia.")
            logger.critical("OPENAI_API_KEY environment variable not set.")
            if tts: tts.speak("Virhe: Tarvittava OpenAI-avain puuttuu.")
            return # Cannot run without OpenAI
        openai_client = OpenAI(api_key=openai_api_key)
        logger.info("OpenAI client initialized.")

        # Initialize Speech Recognizer (Google Speech or Keyboard Fallback)
        recognizer = None
        use_google_speech = True # Flag to try Google Speech first

        # --- Check for command-line argument to force keyboard input ---
        if '--keyboard' in sys.argv:
            use_google_speech = False
            logger.info("Command-line flag '--keyboard' detected. Forcing keyboard input.")
            print("INFO: '--keyboard' lippu havaittu. Käytetään näppäimistöä.")


        if use_google_speech:
            try:
                # Try Google Speech Recognizer first
                recognizer = GoogleSpeechRecognizer(credentials_path=google_credentials_path)
                print("INFO: Käytetään Google puheentunnistusta.")
                logger.info("Using Google Speech Recognizer.")
            except (FileNotFoundError, ImportError, Exception) as e:
                # Fallback to keyboard if Google Speech fails
                print(f"VAROITUS: Google puheentunnistuksen alustus epäonnistui ({e}). Siirrytään näppäimistösyöttöön.")
                logger.warning(f"Google Speech Recognizer initialization failed: {e}. Falling back to keyboard input.")
                recognizer = BasicKeyboardRecognizer()
                print("INFO: Käytetään näppäimistöä syötteenä.")
                logger.info("Using Basic Keyboard Recognizer.")
        else:
            # Directly use keyboard if flag was set
            recognizer = BasicKeyboardRecognizer()
            print("INFO: Käytetään näppäimistöä syötteenä.")
            logger.info("Using Basic Keyboard Recognizer.")


        # --- Create and Run Workflow ---
        workflow = TaxiBookingWorkflow(tts, recognizer, openai_client)

        print("\nINFO: Aloitetaan tilauskeskustelu...")
        # Loop the workflow state machine until it signals completion
        while workflow.run():
            # The run method returns True to continue, False to stop.
            time.sleep(0.1) # Small pause between states if needed

        # Workflow has finished (either success, cancel, or error)
        print("\nINFO: Keskustelu päättyi.")
        final_state = workflow.state
        print(f"INFO: Lopullinen tila: {final_state}")

    except KeyboardInterrupt:
        print("\nNäkemiin! (Keskeytetty näppäimistöltä)")
        if tts:
             try: tts.speak("Näkemiin!")
             except: pass
    except Exception as e:
        # Catch critical errors during initialization or unexpected issues
        logger.critical(f"Critical error in main execution: {e}", exc_info=True)
        print(f"\nKRIITTINEN VIRHE: {e}")
        print("Ohjelman suoritus päättyi virheeseen.")
        if tts:
            try: tts.speak("Ohjelmassa tapahtui vakava virhe.")
            except: pass

    print("\n--- Tilausavustaja suljettu ---")

if __name__ == "__main__":
    # Example: Run with python AuroraVGoogle.py --keyboard to force keyboard input
    main()
