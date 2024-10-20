import os
import librosa
import soundfile as sf
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from moviepy.editor import VideoFileClip
import streamlit as st
from pydub import AudioSegment
import shutil

# Set Hugging Face API Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["general"]["HUGGINGFACEHUB_API_TOKEN"]

# Streamlit app title
st.title('Instagram Caption Generator from Video')
st.markdown('**Upload a video with audio and provide some context to generate a caption!**')

# Step 1: Upload video file
video_file = st.file_uploader("Upload a video with audio", type=["mp4", "mov", "avi"], key="video_file")

# Step 5: Ask for video context from the user
context = st.text_input("What is the video about?", key="context")

# Button to start processing
if st.button("Start Processing", key="start_processing"):
    if video_file is not None and context:
        # Save the uploaded file
        with open("uploaded_video.mp4", "wb") as f:
            f.write(video_file.getbuffer())
        st.success('Video file uploaded successfully!')

        # Step 2: Process video to extract audio using moviepy
        st.write("Processing video...")
        with st.spinner("Extracting audio..."):
            video = VideoFileClip("uploaded_video.mp4")
            audio = video.audio
            if audio is None:
                st.error("No audio found in the video!")
            else:
                audio.write_audiofile("extracted_audio.mp3")
                st.write("Audio extracted.")

        # Step 3: Chunking audio if it's too large
        st.write("Chunking audio into smaller parts...")
        input_file = 'extracted_audio.mp3'

        # Stream over 30 seconds chunks rather than load the full file
        stream = librosa.stream(
            input_file,
            block_length=30,
            frame_length=16000,
            hop_length=16000
        )

        # Save the chunks as separate audio files
        for i, speech in enumerate(stream):
            sf.write(f'{i}.wav', speech, 44100)

        total_chunks = i + 1
        st.write(f"Audio chunked into {total_chunks} parts.")

        # Step 4: Use Hugging Face Whisper model for ASR (Automatic Speech Recognition)
        st.write("Transcribing audio...")
        with st.spinner("Transcribing audio..."):
            asr_model = pipeline(model="openai/whisper-base")

            # Transcribe each audio chunk
            audio_paths = [f'{i}.wav' for i in range(total_chunks)]
            transcriptions = []

            for audio_path in audio_paths:
                transcription = asr_model(audio_path)
                transcriptions.append(transcription['text'])

            # Combine all transcriptions into a full transcript
            full_transcript = ' '.join(transcriptions)
            st.write("Transcription completed.")

        # Step 6: Generate Instagram Caption using Llama 3.1-70B Instruct

        def generate_caption(input_text, context):
            model_id = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"

            # Initialize model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", torch_dtype="bfloat16"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            # Construct the input prompt
            prompt = f"Based on the video transcription: '{input_text}' and the context: '{context}', generate an Instagram caption."

            # Tokenize and generate output
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            outputs = model.generate(input_ids, max_new_tokens=128)

            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Generate Instagram Caption
        with st.spinner("Generating caption..."):
            caption = generate_caption(full_transcript, context)
            st.write("Generated Caption: ", caption)

        # Step 7: Generate Hashtags using the same model

        def generate_hashtags(caption):
            model_id = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"

            # Initialize model and tokenizer (reuse the same instance if needed)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", torch_dtype="bfloat16"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            # Construct the prompt for generating hashtags
            prompt = f"Generate 10 trending hashtags for the following Instagram post caption: '{caption}'."

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            outputs = model.generate(input_ids, max_new_tokens=50)

            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        with st.spinner("Generating hashtags..."):
            hashtags = generate_hashtags(caption)
            st.write("Generated Hashtags: ", hashtags)

        # Cleanup: Delete video and audio files
        os.remove("uploaded_video.mp4")
        os.remove("extracted_audio.mp3")
        for audio_path in audio_paths:
            os.remove(audio_path)

        st.success("Thanks for using this program!")
    else:
        st.warning("Please upload a video and provide context!")
