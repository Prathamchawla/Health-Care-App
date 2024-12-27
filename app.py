from flask import Flask, render_template, request, jsonify
from playsound import playsound
from gtts import gTTS
import os
import speech_recognition as sr
import wave
import pyaudio
import uuid
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import MarianMTModel, MarianTokenizer
load_dotenv()
app = Flask(__name__)

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192")

# Supported languages dictionary
LANGUAGES = {
    'english': 'en',  # English
    'french': 'fr',   # French
    'german': 'de',   # German
    'spanish': 'es',  # Spanish
    'hindi': 'hi'     # Hindi
}


# Global variables for audio recording state
is_recording = False
audio_frames = []

# Audio recording settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Home route
@app.route('/')
def index():
    return render_template('index.html', languages=LANGUAGES)

# Start recording route
@app.route('/start_record', methods=['POST'])
def start_record():
    global is_recording, audio_frames
    if is_recording:
        return jsonify({"success": False, "message": "Already recording."})
    
    is_recording = True
    audio_frames = []

    # Start recording in a new thread
    def record_audio():
        global is_recording, audio_frames
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("Listening...")
        while is_recording:
            data = stream.read(CHUNK)
            audio_frames.append(data)
        stream.stop_stream()
        stream.close()

    import threading
    threading.Thread(target=record_audio).start()
    return jsonify({"success": True, "message": "Recording started. Click 'Stop Recording' to process."})

# Stop recording route
@app.route('/stop_record', methods=['POST'])
def stop_record():
    global is_recording, audio_frames
    if not is_recording:
        return jsonify({"success": False, "message": "Not currently recording."})
    
    is_recording = False

    # Save the recorded audio to a unique WAV file
    filename = f"recorded_audio_{uuid.uuid4().hex}.wav"
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(audio_frames))

    # Process the recorded audio
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(filename) as source:
            audio_data = recognizer.record(source)
            recorded_text = recognizer.recognize_google(audio_data)
        os.remove(filename)  # Delete the file after processing

        # Enhance the recorded text with AI
        try:
            # Create a prompt for the Groq model
            prompt_template = PromptTemplate(
                input_variables=["raw_text"],
                template=(
                    "You are an AI specialized in transcription enhancement and medical terminology.\n"
                    "Given the following text, please correct any errors, enhance the transcription for accuracy, "
                    "and provide explanations for any medical terms in points:\n\n"
                    "Text: {raw_text}\n\n"
                    "Output the improved transcription and explanations."
                ),
            )
            chain = LLMChain(llm=llm, prompt=prompt_template)
            enhanced_text = chain.run(raw_text=recorded_text)

            # Split the enhanced text into points
            enhanced_text_points = enhanced_text.split("\n")  # Split by line breaks

            return jsonify({
                "success": True,
                "recorded_text": recorded_text,
                "enhanced_text": enhanced_text_points
            })
        except Exception as e:
            return jsonify({"success": False, "error": f"Error during enhancement: {str(e)}"})

    except sr.UnknownValueError:
        os.remove(filename)
        return jsonify({"success": False, "error": "Could not understand the audio. Please try again."})
    except sr.RequestError as e:
        os.remove(filename)
        return jsonify({"success": False, "error": f"Google Speech Recognition error: {e}"})
    except Exception as e:
        os.remove(filename)
        return jsonify({"success": False, "error": f"Error during audio processing: {str(e)}"})


@app.route('/translate', methods=['POST'])
def translate_huggingface():
    data = request.get_json()
    text = data.get("text", "").strip()
    target_lang = data.get("lang", "")

    if not text:
        return jsonify({"success": False, "error": "No text provided for translation."})
    if target_lang not in LANGUAGES.values():
        return jsonify({"success": False, "error": "Invalid or unsupported target language."})

    # Hugging Face uses specific model names for language pairs
    # Example: "Helsinki-NLP/opus-mt-en-fr" for English to French
    language_model_mapping = {
        'en': {
            'fr': 'Helsinki-NLP/opus-mt-en-fr',
            'de': 'Helsinki-NLP/opus-mt-en-de',
            'es': 'Helsinki-NLP/opus-mt-en-es',
            'hi': 'Helsinki-NLP/opus-mt-en-hi',
        },
        # Add mappings for other source languages if needed
    }
    source_lang = 'en'  # Assuming input is in English
    model_name = language_model_mapping.get(source_lang, {}).get(target_lang)

    if not model_name:
        return jsonify({"success": False, "error": "Unsupported language pair."})

    try:
        # Load tokenizer and model
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        # Encode and translate text
        encoded_text = tokenizer(text, return_tensors="pt", padding=True)
        translated_tokens = model.generate(**encoded_text)
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        return jsonify({"success": True, "translated_text": translated_text})
    except Exception as e:
        return jsonify({"success": False, "error": f"Error during translation: {str(e)}"})





# Playback route
@app.route('/playback', methods=['POST'])
def playback():
    from time import sleep
    import pygame  # Add pygame for reliable playback

    data = request.get_json()
    text = data.get("text")
    lang = data.get("lang")

    if not text:
        return jsonify({"success": False, "error": "No text provided for playback."})
    if lang not in LANGUAGES.values():
        return jsonify({"success": False, "error": "Invalid language selected."})

    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        filename = "temp_audio.mp3"
        tts.save(filename)

        # Initialize pygame mixer
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        # Wait for playback to complete
        while pygame.mixer.music.get_busy():
            sleep(1)

        # Clean up
        pygame.mixer.quit()
        os.remove(filename)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": f"Error during playback: {str(e)}"})


if __name__ == '__main__':
    app.run(debug=True)
