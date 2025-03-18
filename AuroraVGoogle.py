import os
import sys
import time
import logging
import re
import subprocess
import threading
import queue
from typing import Optional, Tuple, Dict, Any

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
        logging.FileHandler('assistant.log')
    ]
)
logger = logging.getLogger()

# Load environment variables
load_dotenv()

# ----------------------------------------------------------------
# ElevenLabs Text-to-Speech
# ----------------------------------------------------------------
class ElevenLabsTTS:
    """Text-to-Speech using ElevenLabs API."""
    
    def __init__(self, api_key):
        """Initialize ElevenLabs TTS with API key."""
        self.api_key = api_key
        self.client = ElevenLabs(api_key=api_key)
        
        # Default voice ID - you can change this to your preferred voice
        self.voice_id = "YSabzCJMvEHDduIDMdwV"  # Default voice Aurora Fin
        #self.model_id = "eleven_multilingual_v2"  # Multilingual model 	
        self.model_id = "eleven_flash_v2_5"  # fastest speech synthesis

        logger.info("ElevenLabs TTS initialized")
    
    def speak(self, text):
        """Convert text to speech and play it."""
        try:
            print(f"Assistant: {text}")
            
            # Convert text to speech
            audio = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id=self.model_id,
                output_format="mp3_44100_128",
            )
            
            # Play the audio
            play(audio)
            
            # Add a small delay after speaking
            time.sleep(0.1)
            
            return True
        except Exception as e:
            logger.error(f"ElevenLabs TTS error: {e}")
            print(f"[VOICE WOULD SAY]: {text}")
            return False

# ----------------------------------------------------------------
# Google Speech Recognition
# ----------------------------------------------------------------
class GoogleSpeechRecognizer:
    """Speech recognition using Google Cloud Speech-to-Text API."""
    
    def __init__(self, credentials_path=None):
        """Initialize Google Speech with optional credentials path."""
        self.credentials_path = credentials_path
        
        # Set credentials environment variable if provided
        if credentials_path and os.path.exists(credentials_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            
        try:
            # Initialize client
            self.client = speech.SpeechClient()
            logger.info("Google Speech Recognizer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Google Speech client: {e}")
            raise
    
    def listen_once(self, timeout=15):
        """Listen for speech once and return the recognized text."""
        print("🎤 Listening for speech... (Please speak now)")
        
        try:
            # Import necessary libraries for microphone input
            import pyaudio
            import audioop
            
            # Audio recording parameters
            RATE = 16000
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            MAX_SECONDS = min(timeout, 60)  # Limit to 60 seconds max
            
            # Silence parameters
            SILENCE_THRESHOLD = 500  # Adjust based on testing
            SILENCE_SECONDS = 1.5  # Stop after 1.5 seconds of silence
            silence_chunks = int(SILENCE_SECONDS * RATE / CHUNK)
            
            print(f"Waiting up to {MAX_SECONDS} seconds for speech...")
            
            # Initialize PyAudio
            p = pyaudio.PyAudio()
            
            # Open stream
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
            
            # Start recording
            frames = []
            silence_count = 0
            has_speech = False
            start_time = time.time()
            
            while True:
                # Check for timeout
                elapsed = time.time() - start_time
                if elapsed > MAX_SECONDS:
                    logger.info(f"Recording timed out after {elapsed:.1f} seconds")
                    break
                
                # Read audio chunk
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                # Measure volume and detect silence
                volume = audioop.rms(data, 2)  # 2 bytes per sample for paInt16
                
                # If volume is above threshold, mark as speech and reset silence counter
                if volume > SILENCE_THRESHOLD:
                    has_speech = True
                    silence_count = 0
                # If we have detected speech and now have silence, increment counter
                elif has_speech:
                    silence_count += 1
                    
                # If enough silence after speech, stop recording
                if has_speech and silence_count >= silence_chunks:
                    logger.info(f"Detected {SILENCE_SECONDS}s of silence after speech, stopping recording")
                    break
            
            # Stop and close the stream
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            if not has_speech:
                logger.info("No speech detected during recording")
                return None
                
            print("Processing speech...")
            
            # Combine all recorded audio
            audio_data = b''.join(frames)
            
            # Configure recognition request
            audio = speech.RecognitionAudio(content=audio_data)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=RATE,
                language_code="fi-FI",  # Finnish language
                enable_automatic_punctuation=True,
            )
            
            # Send the request to Google Speech API
            response = self.client.recognize(config=config, audio=audio)
            
            # Process the response
            transcript = ""
            for result in response.results:
                transcript += result.alternatives[0].transcript
            
            if transcript:
                print(f"User: {transcript}")
                return transcript
            else:
                print("No speech detected or recognized")
                return None
                
        except Exception as e:
            logger.error(f"Error in Google speech recognition: {e}")
            return None
        
# ----------------------------------------------------------------
# Basic Keyboard Input Recognition (Fallback)
# ----------------------------------------------------------------
class BasicKeyboardRecognizer:
    """Simple keyboard-based input as a fallback for speech recognition."""
    
    def __init__(self):
        pass
    
    def listen_once(self, timeout=None):
        """Get keyboard input with optional timeout."""
        print("⌨️ Enter your message:")
        
        try:
            # Set up timeout handling if requested
            if timeout:
                import signal
                
                # Define timeout handler
                def timeout_handler(signum, frame):
                    raise TimeoutError("Input timed out")
                
                # Set timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
            
            # Get input
            user_input = input()
            
            # Clear timeout if it was set
            if timeout:
                signal.alarm(0)
            
            if user_input.strip():
                print(f"User: {user_input}")
                return user_input
            else:
                return None
                
        except TimeoutError:
            print("Input timed out")
            return None
        except Exception as e:
            logger.error(f"Error getting keyboard input: {e}")
            return None

# ----------------------------------------------------------------
# Improved Workflow with Specialized Intent Analysis
# ----------------------------------------------------------------
class ImprovedWorkflow:
    """Improved workflow with specialized intent analysis functions."""
    
    def __init__(self, tts, recognizer, openai_client):
        """Initialize the workflow with TTS, recognizer, and OpenAI client."""
        self.tts = tts
        self.recognizer = recognizer
        self.openai_client = openai_client
        self.state = "start"
        self.user_name = None
        self.user_address = None  # Added for the third question
    
    def analyze_yes_no_intent(self, user_input):
        """
        Analyze if the user input is affirmative (yes) or negative (no).
        Specialized for the first question in the workflow.
        """
        try:
            # Create a focused prompt just for yes/no analysis
            prompt = f"""
            Analyze the following user input in Finnish and determine if it's affirmative or negative.
            User input: "{user_input}"
            
            If the user is saying "yes", "ok", "sure", or any affirmative answer in Finnish 
            (like "kyllä", "joo", "juu", "okei", "toki", "haluan" etc.), return EXACTLY "INTENT: YES".
            
            If the user is saying "no", "nope", or any negative answer in Finnish
            (like "ei", "en", "en halua", etc.), return EXACTLY "INTENT: NO".
            
            If the user says they want to exit, return EXACTLY "INTENT: EXIT".
            
            For any other response that cannot be clearly classified as yes or no,
            return EXACTLY "INTENT: UNKNOWN".
            
            Your response should be ONLY ONE LINE with INTENT: followed by the classification.
            """
            
            # Send to OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Use a fast, efficient model for simple intent detection
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract the response
            intent_analysis = response.choices[0].message.content.strip()
            logger.info(f"Yes/No intent analysis result: {intent_analysis}")
            
            # Parse the intent - this should be simpler now with focused response
            intent_match = re.search(r"INTENT:\s*(\w+)", intent_analysis)
            
            if intent_match:
                intent = intent_match.group(1).strip().lower()
                return intent
            else:
                # Fallback parsing 
                if "YES" in intent_analysis.upper():
                    return "yes"
                elif "NO" in intent_analysis.upper():
                    return "no"
                elif "EXIT" in intent_analysis.upper():
                    return "exit"
                else:
                    return "unknown"
                
        except Exception as e:
            logger.error(f"Error analyzing yes/no intent: {e}")
            return "unknown"
    
    def analyze_name_intent(self, user_input):
        """
        Extract name from user input.
        Specialized for the second question in the workflow.
        """
        try:
            # Create a focused prompt just for name extraction
            prompt = f"""
                The user's speech has been transcribed from Finnish voice recognition software. Your goal is to determine if the user is providing their name in the transcribed text.

                User input (transcribed): "{user_input}"

                Task:

                Extract the person's name if they are providing one.
                Return the extraction in the format: NAME: [extracted name].
                If the user explicitly states they want to exit, return EXIT.
                If the user does not provide a name or says something unrelated, return UNKNOWN.
                Important Details:

                Common Finnish name-introduction patterns include:
                "Nimeni on [Name]"
                "Olen [Name]"
                Just the name itself (e.g., "Matti Virtanen").
                Do not include "nimeni on" or "olen" in your returned name.
                Your response must be exactly one line with only the required output.
                Examples:

                Input: "Nimeni on Matti Virtanen" → Output: "NAME: Matti Virtanen"
                Input: "Olen Liisa" → Output: "NAME: Liisa"
                Input: "Juhani Korhonen" → Output: "NAME: Juhani Korhonen"
            """
            
            # Send to OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Fast model is sufficient for name extraction
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract the response
            name_analysis = response.choices[0].message.content.strip()
            logger.info(f"Name analysis result: {name_analysis}")
            
            # Parse the result
            if name_analysis.startswith("NAME:"):
                extracted_name = name_analysis[5:].strip()
                return "name", extracted_name
            elif "EXIT" in name_analysis.upper():
                return "exit", None
            else:
                return "unknown", None
                
        except Exception as e:
            logger.error(f"Error analyzing name intent: {e}")
            return "unknown", None

    def analyze_address_intent(self, user_input):
        """
        Extract address from user input.
        Specialized for the third question in the workflow.
        """
        try:
            # Create a focused prompt just for address extraction
            prompt = f"""
                The user's speech has been transcribed from Finnish voice recognition software. Your goal is to determine if the user is providing an address in the transcribed text.

                User input (transcribed): "{user_input}"

                Task:

                Extract the address if the user is providing one.
                Return the extraction in the format: ADDRESS: [extracted address].
                If the user explicitly states they want to exit, return EXIT.
                If the user does not provide an address or says something unrelated, return UNKNOWN.
                Important Details:

                Common Finnish address patterns include:
                "Osoite on [Address]"
                "[Street name] [number], [postal code] [city]"
                Just the address itself (e.g., "Mannerheimintie 10, 00100 Helsinki").
                Do not include "osoite on" in your returned address.
                Your response must be exactly one line with only the required output.
                Examples:

                Input: "Osoite on Mannerheimintie 10, 00100 Helsinki" → Output: "ADDRESS: Mannerheimintie 10, 00100 Helsinki"
                Input: "Hämeentie 42, Turku" → Output: "ADDRESS: Hämeentie 42, Turku"
                Input: "Se on Koulukatu 1" → Output: "ADDRESS: Koulukatu 1"
            """
            
            # Send to OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Fast model is sufficient for address extraction
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract the response
            address_analysis = response.choices[0].message.content.strip()
            logger.info(f"Address analysis result: {address_analysis}")
            
            # Parse the result
            if address_analysis.startswith("ADDRESS:"):
                extracted_address = address_analysis[8:].strip()
                return "address", extracted_address
            elif "EXIT" in address_analysis.upper():
                return "exit", None
            else:
                return "unknown", None
                
        except Exception as e:
            logger.error(f"Error analyzing address intent: {e}")
            return "unknown", None
    
    def write_to_notepad(self, text):
        """Write text to Notepad."""
        try:
            # Wait a bit for Notepad to be ready
            time.sleep(0.5)
            
            # Try using pyautogui if available
            try:
                import pyautogui
                pyautogui.write(text)
                return True
            except ImportError:
                logger.warning("pyautogui not available for writing to Notepad")
                
                # Try using win32com as an alternative
                try:
                    import win32com.client
                    shell = win32com.client.Dispatch("WScript.Shell")
                    shell.SendKeys(text)
                    return True
                except ImportError:
                    logger.warning("win32com not available for writing to Notepad")
                    
                    # Last resort: Create a text file
                    with open("user_name.txt", "w", encoding="utf-8") as f:
                        f.write(text)
                    logger.info(f"Wrote to user_name.txt instead of Notepad: {text}")
                    print(f"Note: Created user_name.txt with '{text}' since automation libraries aren't available")
                    return True
        except Exception as e:
            logger.error(f"Error writing to Notepad: {e}")
            return False
    
    def run(self):
        """Run the improved workflow with specialized intent analysis."""
        try:
            # Initial greeting - updated to mention three questions
            self.tts.speak("Hei, tämä on esimerkki. Tavoitteena on kysyä sinulta kolme kysymystä ja tehdä niiden perusteella toimia. Ensimmäinen kysymys: Haluatko avata muistion?")
            
            # State machine for the workflow
            while True:
                # Get user input
                user_input = self.recognizer.listen_once(timeout=30)
                
                if not user_input:
                    self.tts.speak("En kuullut vastaustasi. Yritetään uudelleen.")
                    continue
                
                # Use different intent analysis based on the current state
                if self.state == "start":
                    # For the first question, use yes/no intent analysis
                    intent = self.analyze_yes_no_intent(user_input)
                    logger.info(f"Analyzed yes/no intent: {intent}")
                    
                    if intent == "yes":
                        # Open Notepad
                        try:
                            subprocess.Popen("notepad.exe")
                            self.tts.speak("Avasin sinulle muistion. Haluaisitko kertoa nimesi?")
                            self.state = "ask_name"
                        except Exception as e:
                            logger.error(f"Error opening Notepad: {e}")
                            self.tts.speak("En valitettavasti pystynyt avaamaan muistiota. Lopetetaan ohjelma.")
                            break
                    elif intent == "no":
                        self.tts.speak("Et halunnut avata muistiota. Lopetetaan ohjelma.")
                        break
                    elif intent == "exit":
                        self.tts.speak("Lopetetaan ohjelma.")
                        break
                    else:
                        self.tts.speak("En ymmärtänyt vastaustasi. Haluatko avata muistion? Vastaa kyllä tai ei.")
                
                elif self.state == "ask_name":
                    # For the second question, use name extraction
                    intent, name = self.analyze_name_intent(user_input)
                    logger.info(f"Analyzed name intent: {intent}, name: {name}")
                    
                    if intent == "name" and name:
                        self.user_name = name
                        self.tts.speak(f"Nimesi on {self.user_name}. Kirjoitan nimesi muistioon.")
                        
                        # Write name to Notepad
                        if self.write_to_notepad(self.user_name + "\n"):
                            self.tts.speak("Nimesi on nyt kirjoitettu muistioon. Mistä osoitteesta haku?")
                            self.state = "ask_address"
                        else:
                            self.tts.speak("En pystynyt kirjoittamaan nimeäsi muistioon, mutta jatketaan silti. Mistä osoitteesta haku?")
                            self.state = "ask_address"
                    elif intent == "exit":
                        self.tts.speak("Lopetetaan ohjelma.")
                        break
                    else:
                        self.tts.speak("En saanut selvää nimestäsi. Voisitko kertoa sen uudelleen?")
                
                elif self.state == "ask_address":
                    # For the third question, use address extraction
                    intent, address = self.analyze_address_intent(user_input)
                    logger.info(f"Analyzed address intent: {intent}, address: {address}")
                    
                    if intent == "address" and address:
                        self.user_address = address
                        self.tts.speak(f"Osoite on {self.user_address}. Kirjoitan osoitteen muistioon.")
                        
                        # Write address to Notepad - append to the name
                        if self.write_to_notepad(self.user_address):
                            self.tts.speak("Osoite on nyt kirjoitettu muistioon. Kiitos käytöstä, lopetetaan ohjelma.")
                        else:
                            self.tts.speak("En pystynyt kirjoittamaan osoitetta muistioon, mutta kiitos käytöstä. Lopetetaan ohjelma.")
                        
                        break
                    elif intent == "exit":
                        self.tts.speak("Lopetetaan ohjelma.")
                        break
                    else:
                        self.tts.speak("En saanut selvää osoitteesta. Voisitko kertoa sen uudelleen?")
            
            print("Workflow completed successfully.")
            return True
                
        except KeyboardInterrupt:
            self.tts.speak("Ohjelma keskeytettiin.")
        except Exception as e:
            logger.error(f"Error in workflow: {e}")
            self.tts.speak("Tapahtui virhe. Lopetetaan ohjelma.")
            
        return False

# ----------------------------------------------------------------
# Main Application
# ----------------------------------------------------------------
def main():
    """Run the Finnish voice assistant with improved workflow."""
    print("Suomalainen Ääniavustaja - Finnish Voice Assistant")
    print("================================================")
    print("Paranneltu versio V3 - Improved Version V3")
    print()
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Check for required API keys
        elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        google_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        if not elevenlabs_api_key:
            print("ELEVENLABS_API_KEY must be set in the environment or .env file.")
            sys.exit(1)
        
        if not openai_api_key:
            print("OPENAI_API_KEY must be set in the environment or .env file.")
            sys.exit(1)
        
        # Initialize TTS using ElevenLabs
        tts = ElevenLabsTTS(api_key=elevenlabs_api_key)
        
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=openai_api_key)
        
        # Select recognizer based on available credentials
        recognizer = None
        
        # Try Google Speech recognizer if credentials are available
        if google_credentials_path:
            try:
                print("Initializing Google speech recognition...")
                recognizer = GoogleSpeechRecognizer(credentials_path=google_credentials_path)
                print("Google speech recognizer initialized")
            except Exception as e:
                print(f"Error initializing Google speech recognizer: {e}")
                print("Falling back to keyboard input")
        else:
            print("GOOGLE_APPLICATION_CREDENTIALS not found, using keyboard input")
            
        # Fallback to keyboard if speech recognition isn't available
        if not recognizer:
            recognizer = BasicKeyboardRecognizer()
            print("Using keyboard for input")
        
        # Initialize and run the improved workflow
        workflow = ImprovedWorkflow(tts, recognizer, openai_client)
        workflow.run()
            
    except KeyboardInterrupt:
        print("\nSuljetaan avustaja. Näkemiin!")
    except Exception as e:
        logger.error(f"Vakava virhe: {e}")
        print(f"Vakava virhe: {e}")
        
    print("Ääniavustaja lopetettu.")

if __name__ == "__main__":
    main()