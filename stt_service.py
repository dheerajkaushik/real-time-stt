import uvicorn
import numpy as np
import io
import logging
import torch
import asyncio
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

# Set up logging
logging.basicConfig(level=logging.INFO)

# --- Configuration & Model Loading ---
MODEL_SIZE = "base" 
SAMPLE_RATE = 16000 # Required sample rate for Whisper and VAD
VAD_SILENCE_DURATION = 2.0 # seconds of silence required to trigger transcription
VAD_THRESHOLD = 0.5 # VAD probability threshold

# Determine Device
try:
    # Use torch to check for CUDA availability (most robust check)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"
    
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
try:
    logging.info(f"Loading Whisper model: {MODEL_SIZE} on {DEVICE}...")
    compute_type = "float16" if DEVICE == "cuda" else "int8"
    whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=compute_type)
    logging.info("Whisper Model loaded successfully.")

    # Load Silero VAD model directly using torch.hub to bypass local module import issues
    logging.info("Loading VAD Model via torch.hub...")
    # This downloads the model weight files if they aren't locally cached.
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False
    )
    
    vad_model.to(DEVICE)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    logging.info("VAD Model loaded successfully.")

except Exception as e:
    logging.error(f"Error loading models: {e}")
    raise HTTPException(status_code=500, detail="Failed to load necessary models. Check PyTorch/faster-whisper/VAD installation.")
# -------------------------------------


# Helper function to convert 16-bit PCM bytes to 16kHz float array
def pcm_to_float(pcm_bytes):
    """Converts 16-bit PCM bytes to a float32 NumPy array normalized to [-1, 1]."""
    # The frontend sends 16-bit PCM data (dtype=np.int16)
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

# Helper function to transcribe audio chunk
def transcribe_segment(audio_data):
    """Transcribes a given float32 audio array using faster-whisper."""
    # We run this in a separate executor thread to prevent blocking the asyncio loop
    segments, info = whisper_model.transcribe(
        audio_data, 
        beam_size=5, 
        language=None, # Auto-detect multilingual
        vad_filter=False # VAD is handled externally by Silero
    )
    transcript = " ".join([segment.text for segment in segments])
    return transcript.strip()


@app.websocket("/ws/stt")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("WebSocket connection established for streaming transcription.")
    
    current_speech_buffer = [] # Buffer for the current segment of speech
    silence_counter = 0        # Counter to track continuous silence
    # Duration of one incoming chunk in seconds (approx 4096 samples / 16000 Hz)
    CHUNK_DURATION = 4096 / SAMPLE_RATE 
    
    # Required number of silent chunks to trigger transcription
    SILENCE_CHUNKS_REQUIRED = VAD_SILENCE_DURATION / CHUNK_DURATION
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            # Use 'END_STREAM' for final cleanup
            if data == b'END_STREAM':
                logging.info("Client stream ended. Processing final buffer.")
                break

            # 1. Convert incoming chunk bytes to float32
            audio_chunk_float = pcm_to_float(data)
            
            # 2. VAD Check: Pass chunk through Silero VAD model
            # Ensure the input is a torch tensor and moved to the correct device
            audio_tensor = torch.from_numpy(audio_chunk_float).to(DEVICE)
            vad_prob = vad_model(audio_tensor, SAMPLE_RATE).item()

            if vad_prob > VAD_THRESHOLD:
                # --- SPEECH DETECTED ---
                current_speech_buffer.append(data)
                silence_counter = 0 # Reset silence counter
                
            else:
                # --- SILENCE DETECTED ---
                if current_speech_buffer:
                    silence_counter += 1
                    current_speech_buffer.append(data) # Keep appending to ensure complete phrase
                    
                    if silence_counter >= SILENCE_CHUNKS_REQUIRED:
                        # Silence threshold reached: Time to transcribe the segment!
                        full_speech_bytes = b"".join(current_speech_buffer)
                        full_speech_float = pcm_to_float(full_speech_bytes)
                        
                        # Transcribe in a separate executor thread
                        transcript = await asyncio.get_event_loop().run_in_executor(
                            None, transcribe_segment, full_speech_float
                        )
                        
                        if transcript:
                            logging.info(f"Segment Transcript: {transcript}")
                            # Send the result back immediately for continuous output
                            await websocket.send_text(transcript)
                        
                        # Reset buffer after successful transcription
                        current_speech_buffer = []
                        silence_counter = 0
                
    except Exception as e:
        logging.error(f"WebSocket closed with error: {e}")
    finally:
        # Final Transcription on disconnect/END_STREAM if buffer remains
        if current_speech_buffer:
             logging.info("Processing remaining audio after stream termination.")
             full_speech_bytes = b"".join(current_speech_buffer)
             full_speech_float = pcm_to_float(full_speech_bytes)
             
             transcript = await asyncio.get_event_loop().run_in_executor(
                 None, transcribe_segment, full_speech_float
             )
             if transcript:
                 await websocket.send_text(transcript)
                 
        await websocket.send_text("__END_OF_TRANSCRIPT__")
        await websocket.close()
        logging.info("WebSocket connection closed.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)