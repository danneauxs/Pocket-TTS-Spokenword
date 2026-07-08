import io
import logging
import os
import tempfile
import threading
from pathlib import Path
from queue import Queue

import torch
import typer
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from typing_extensions import Annotated

from pocket_tts.data.audio import stream_audio_chunks
from pocket_tts.default_parameters import (
    DEFAULT_AUDIO_PROMPT,
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_FRAMES_AFTER_EOS,
    DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_TEMPERATURE,
    DEFAULT_VARIANT,
)
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.utils.logging_utils import enable_logging
from pocket_tts.utils.utils import PREDEFINED_VOICES, size_of_dict

logger = logging.getLogger(__name__)

cli_app = typer.Typer(
    help="Kyutai Pocket TTS - Text-to-Speech generation tool", pretty_exceptions_show_locals=False
)


# ------------------------------------------------------
# The pocket-tts server implementation
# ------------------------------------------------------

# Global model instance
tts_model = None
global_model_state = None

web_app = FastAPI(
    title="Kyutai Pocket TTS API", description="Text-to-Speech generation API", version="1.0.0"
)
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://pod1-10007.internal.kyutai.org",
        "https://kyutai.org",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_app.get("/")
async def root():
    """Serve the frontend."""
    static_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(static_path)


@web_app.get("/health")
async def health():
    return {"status": "healthy"}


def write_to_queue(queue, text_to_generate, model_state):
    """Allows writing to the StreamingResponse as if it were a file."""

    class FileLikeToQueue(io.IOBase):
        def __init__(self, queue):
            self.queue = queue

        def write(self, data):
            self.queue.put(data)

        def flush(self):
            pass

        def close(self):
            self.queue.put(None)

    audio_chunks = tts_model.generate_audio_stream(
        model_state=model_state, text_to_generate=text_to_generate
    )
    stream_audio_chunks(FileLikeToQueue(queue), audio_chunks, tts_model.config.mimi.sample_rate)


def generate_data_with_state(text_to_generate: str, model_state: dict):
    queue = Queue()

    # Run your function in a thread
    thread = threading.Thread(target=write_to_queue, args=(queue, text_to_generate, model_state))
    thread.start()

    # Yield data as it becomes available
    i = 0
    while True:
        data = queue.get()
        if data is None:
            break
        i += 1
        yield data

    thread.join()


@web_app.post("/tts")
def text_to_speech(
    text: str = Form(...),
    voice_url: str | None = Form(None),
    voice_wav: UploadFile | None = File(None),
):
    """
    Generate speech from text using the pre-loaded voice prompt or a custom voice.

    Args:
        text: Text to convert to speech
        voice_url: Optional voice URL (http://, https://, or hf://)
        voice_wav: Optional uploaded voice file (mutually exclusive with voice_url)
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if voice_url is not None and voice_wav is not None:
        raise HTTPException(status_code=400, detail="Cannot provide both voice_url and voice_wav")

    # Use the appropriate model state
    if voice_url is not None:
        if not (
            voice_url.startswith("http://")
            or voice_url.startswith("https://")
            or voice_url.startswith("hf://")
            or voice_url in PREDEFINED_VOICES
        ):
            raise HTTPException(
                status_code=400, detail="voice_url must start with http://, https://, or hf://"
            )
        model_state = tts_model._cached_get_state_for_audio_prompt(voice_url, truncate=True)
        logging.warning("Using voice from URL: %s", voice_url)
    elif voice_wav is not None:
        # Use uploaded voice file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = voice_wav.file.read()
            temp_file.write(content)
            temp_file.flush()

            try:
                model_state = tts_model.get_state_for_audio_prompt(
                    Path(temp_file.name), truncate=True
                )
            finally:
                os.unlink(temp_file.name)
    else:
        # Use default global model state
        model_state = global_model_state

    return StreamingResponse(
        generate_data_with_state(text, model_state),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=generated_speech.wav",
            "Transfer-Encoding": "chunked",
        },
    )


@cli_app.command()
def serve(
    voice: Annotated[
        str, typer.Option(help="Path to voice prompt audio file (voice to clone)")
    ] = DEFAULT_AUDIO_PROMPT,
    host: Annotated[str, typer.Option(help="Host to bind to")] = "localhost",
    port: Annotated[int, typer.Option(help="Port to bind to")] = 8000,
    reload: Annotated[bool, typer.Option(help="Enable auto-reload")] = False,
):
    """Start the FastAPI server."""

    global tts_model, global_model_state
    tts_model = TTSModel.load_model(DEFAULT_VARIANT)

    # Pre-load the voice prompt
    global_model_state = tts_model.get_state_for_audio_prompt(voice)
    logger.info(f"The size of the model state is {size_of_dict(global_model_state) // 1e6} MB")

    uvicorn.run("pocket_tts.main:web_app", host=host, port=port, reload=reload)


# ------------------------------------------------------
# The pocket-tts single generation CLI implementation
# ------------------------------------------------------


@cli_app.command()
def generate(
    text: Annotated[
        str, typer.Option(help="Text to generate")
    ] = "Hello world. I am Kyutai's Pocket TTS. I'm fast enough to run on small CPUs. I hope you'll like me.",
    voice: Annotated[
        str, typer.Option(help="Path to audio conditioning file (voice to clone)")
    ] = DEFAULT_AUDIO_PROMPT,
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging output")] = False,
    variant: Annotated[str, typer.Option(help="Model signature")] = DEFAULT_VARIANT,
    lsd_decode_steps: Annotated[
        int, typer.Option(help="Number of generation steps")
    ] = DEFAULT_LSD_DECODE_STEPS,
    temperature: Annotated[
        float, typer.Option(help="Temperature for generation")
    ] = DEFAULT_TEMPERATURE,
    noise_clamp: Annotated[float, typer.Option(help="Noise clamp value")] = DEFAULT_NOISE_CLAMP,
    eos_threshold: Annotated[float, typer.Option(help="EOS threshold")] = DEFAULT_EOS_THRESHOLD,
    frames_after_eos: Annotated[
        int, typer.Option(help="Number of frames to generate after EOS")
    ] = DEFAULT_FRAMES_AFTER_EOS,
    output_path: Annotated[
        str, typer.Option(help="Output path for generated audio")
    ] = "./tts_output.wav",
    device: Annotated[str, typer.Option(help="Device to use")] = "cpu",
):
    """Generate speech using Kyutai Pocket TTS."""
    if "cuda" in device:
        # Cuda graphs capturing does not play nice with multithreading.
        os.environ["NO_CUDA_GRAPH"] = "1"

    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        tts_model = TTSModel.load_model(
            variant, temperature, lsd_decode_steps, noise_clamp, eos_threshold
        )
        tts_model.to(device)

        model_state_for_voice = tts_model.get_state_for_audio_prompt(voice)
        # Stream audio generation directly to file or stdout
        audio_chunks = tts_model.generate_audio_stream(
            model_state=model_state_for_voice,
            text_to_generate=text,
            frames_after_eos=frames_after_eos,
        )

        stream_audio_chunks(output_path, audio_chunks, tts_model.config.mimi.sample_rate)

        # Only print the result message if not writing to stdout
        if output_path != "-":
            logger.info("Results written in %s", output_path)
        logger.info("-" * 20)
        logger.info(
            "If you want to try multiple voices and prompts quickly, try the `serve` command."
        )
        logger.info(
            "If you like Kyutai projects, comment, like, subscribe at https://x.com/kyutai_labs"
        )


@cli_app.command()
def audiobook(
    json_file: Annotated[Path, typer.Argument(help="Path to JSON file with text chunks")],
    voice: Annotated[
        str, typer.Option(help="Path to audio conditioning file (voice to clone)")
    ] = DEFAULT_AUDIO_PROMPT,
    output_path: Annotated[
        str, typer.Option(help="Output path for generated audio")
    ] = "./audiobook_output.wav",
    start_index: Annotated[int, typer.Option(help="Start from chunk index (for resume)")] = 0,
    end_index: Annotated[
        int | None, typer.Option(help="End at chunk index (None = process all)")
    ] = None,
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging output")] = False,
    variant: Annotated[str, typer.Option(help="Model signature")] = DEFAULT_VARIANT,
    lsd_decode_steps: Annotated[
        int, typer.Option(help="Number of generation steps")
    ] = DEFAULT_LSD_DECODE_STEPS,
    base_temperature: Annotated[
        float, typer.Option(help="Base temperature (can be overridden by JSON)")
    ] = DEFAULT_TEMPERATURE,
    use_chunk_temperature: Annotated[
        bool, typer.Option(help="Use temperature from JSON chunks if available")
    ] = True,
    noise_clamp: Annotated[float, typer.Option(help="Noise clamp value")] = DEFAULT_NOISE_CLAMP,
    eos_threshold: Annotated[float, typer.Option(help="EOS threshold")] = DEFAULT_EOS_THRESHOLD,
    device: Annotated[str, typer.Option(help="Device to use")] = "cpu",
):
    """Generate audiobook from JSON file with text chunks.

    Processes a JSON file containing text chunks sequentially, streaming audio
    directly to disk without holding it in RAM. Supports resume from any chunk.

    JSON format expected:
    [
      {"_metadata": true, "voice_used": "...", ...},
      {"index": 0, "text": "...", "tts_params": {"temperature": 0.85, ...}},
      {"index": 1, "text": "...", ...}
    ]
    """
    import json
    import time
    from collections.abc import Iterator

    if "cuda" in device:
        os.environ["NO_CUDA_GRAPH"] = "1"

    # Load JSON file
    if not json_file.exists():
        logger.error(f"JSON file not found: {json_file}")
        raise typer.Exit(1)

    with open(json_file, "r") as f:
        chunks_data = json.load(f)

    # Separate metadata from chunks
    metadata = None
    chunks = []
    for item in chunks_data:
        if item.get("_metadata"):
            metadata = item
        else:
            chunks.append(item)

    if not chunks:
        logger.error("No text chunks found in JSON file")
        raise typer.Exit(1)

    # Validate chunk indices
    if end_index is None:
        end_index = len(chunks)
    if start_index < 0 or start_index >= len(chunks):
        logger.error(f"start_index {start_index} out of range (0-{len(chunks) - 1})")
        raise typer.Exit(1)
    if end_index <= start_index or end_index > len(chunks):
        logger.error(
            f"end_index {end_index} invalid (must be > {start_index} and <= {len(chunks)})"
        )
        raise typer.Exit(1)

    chunks_to_process = chunks[start_index:end_index]

    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        logger.info("=" * 60)
        logger.info("AUDIOBOOK GENERATION")
        logger.info("=" * 60)
        if metadata:
            logger.info(f"Metadata: {metadata}")
        logger.info(f"Total chunks in file: {len(chunks)}")
        logger.info(
            f"Processing chunks: {start_index} to {end_index - 1} ({len(chunks_to_process)} chunks)"
        )
        logger.info(f"Voice: {voice}")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 60)

        # Load model once
        logger.info("Loading TTS model...")
        tts_model = TTSModel.load_model(
            variant, base_temperature, lsd_decode_steps, noise_clamp, eos_threshold
        )
        tts_model.to(device)

        # Get voice state once
        logger.info(f"Loading voice: {voice}")
        model_state_for_voice = tts_model.get_state_for_audio_prompt(voice)

        # Create a generator that processes all chunks
        def generate_all_chunks() -> Iterator[torch.Tensor]:
            total_duration = 0.0
            start_time = time.time()
            elapsed = 0.0

            for i, chunk in enumerate(chunks_to_process):
                chunk_idx = chunk.get("index", start_index + i)
                text = chunk.get("text", "")
                word_count = chunk.get("word_count", len(text.split()))

                if not text.strip():
                    logger.warning(f"Chunk {chunk_idx}: Empty text, skipping")
                    continue

                # Get temperature from chunk or use base
                chunk_temp = base_temperature
                if use_chunk_temperature and "tts_params" in chunk:
                    chunk_temp = chunk["tts_params"].get("temperature", base_temperature)

                logger.info("-" * 60)
                logger.info(
                    f"Processing chunk {chunk_idx} ({i + 1}/{len(chunks_to_process)}): "
                    f"{word_count} words, temp={chunk_temp:.2f}"
                )
                logger.info(f"Text preview: {text[:100]}{'...' if len(text) > 100 else ''}")

                # Update model temperature if different
                if abs(tts_model.temp - chunk_temp) > 0.001:
                    tts_model.temp = chunk_temp

                chunk_start_time = time.time()

                # Generate audio for this chunk
                audio_chunks = tts_model.generate_audio_stream(
                    model_state=model_state_for_voice, text_to_generate=text, frames_after_eos=None
                )

                chunk_samples = 0
                for audio_chunk in audio_chunks:
                    chunk_samples += len(audio_chunk)
                    yield audio_chunk

                # Calculate stats
                chunk_duration = chunk_samples / tts_model.config.mimi.sample_rate
                chunk_time = time.time() - chunk_start_time
                total_duration += chunk_duration
                elapsed = time.time() - start_time

                logger.info(
                    f"Chunk {chunk_idx} done: {chunk_duration:.2f}s audio in {chunk_time:.2f}s "
                    f"({chunk_duration / chunk_time:.2f}x realtime)"
                )
                logger.info(
                    f"Progress: {i + 1}/{len(chunks_to_process)} chunks, "
                    f"{total_duration:.1f}s audio generated, "
                    f"{elapsed:.1f}s elapsed"
                )

            logger.info("=" * 60)
            logger.info(
                f"COMPLETE: Generated {total_duration:.2f}s ({total_duration / 60:.2f} min) "
                f"of audio in {elapsed:.1f}s ({total_duration / elapsed:.2f}x realtime)"
            )
            logger.info("=" * 60)

        # Stream all chunks to output file
        logger.info(f"Starting generation, streaming to {output_path}...")
        stream_audio_chunks(output_path, generate_all_chunks(), tts_model.config.mimi.sample_rate)

        logger.info(f"✓ Audiobook saved to: {output_path}")
        logger.info(f"To resume from chunk {end_index}, use: --start-index {end_index}")


@cli_app.command()
def batch_voices(
    text_file: Annotated[
        Path, typer.Argument(help="Path to text file to generate speech for")
    ],
    voices_dir: Annotated[
        Path, typer.Argument(help="Directory containing voice prompt files")
    ],
    output_dir: Annotated[
        Path, typer.Option(help="Output directory for generated audio files")
    ] = Path("./batch_output"),
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging output")] = False,
    variant: Annotated[str, typer.Option(help="Model signature")] = DEFAULT_VARIANT,
    lsd_decode_steps: Annotated[
        int, typer.Option(help="Number of generation steps")
    ] = DEFAULT_LSD_DECODE_STEPS,
    temperature: Annotated[
        float, typer.Option(help="Temperature for generation")
    ] = DEFAULT_TEMPERATURE,
    noise_clamp: Annotated[float, typer.Option(help="Noise clamp value")] = DEFAULT_NOISE_CLAMP,
    eos_threshold: Annotated[float, typer.Option(help="EOS threshold")] = DEFAULT_EOS_THRESHOLD,
    frames_after_eos: Annotated[
        int, typer.Option(help="Number of frames to generate after EOS")
    ] = DEFAULT_FRAMES_AFTER_EOS,
    device: Annotated[str, typer.Option(help="Device to use")] = "cpu",
):
    """Generate speech for the same text using multiple voice prompts in batch mode."""
    if "cuda" in device:
        # Cuda graphs capturing does not play nice with multithreading.
        os.environ["NO_CUDA_GRAPH"] = "1"

    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        # Read text file
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        logger.info(f"Generating audio for text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        logger.info(f"Using voices from: {voices_dir}")

        # Find all voice files in directory
        voice_extensions = ['*.wav', '*.mp3', '*.m4a', '*.flac']
        voice_files = []
        for ext in voice_extensions:
            voice_files.extend(voices_dir.glob(ext))

        if not voice_files:
            logger.error(f"No voice files found in {voices_dir} (supported: WAV, MP3, M4A, FLAC)")
            return

        logger.info(f"Found {len(voice_files)} voice files")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load TTS model once
        tts_model = TTSModel.load_model(
            variant, temperature, lsd_decode_steps, noise_clamp, eos_threshold
        )
        tts_model.to(device)

        # Process each voice
        successful = 0
        failed = 0

        for voice_path in voice_files:
            try:
                voice_name = voice_path.stem
                output_path = output_dir / f"{voice_name}_{text_file.stem}.wav"

                logger.info(f"Processing voice: {voice_name}")

                # Load and convert voice (if needed)
                from pocket_tts.data.voice_converter import VoicePromptConverter

                converter = VoicePromptConverter()
                converted_voice_path = converter.convert(voice_path, output_dir / "converted_voices")
                logger.debug(f"Using converted voice: {converted_voice_path}")

                # Generate audio
                model_state_for_voice = tts_model.get_state_for_audio_prompt(str(converted_voice_path))
                audio_chunks = tts_model.generate_audio_stream(
                    model_state=model_state_for_voice,
                    text_to_generate=text,
                    frames_after_eos=frames_after_eos,
                )

                # Save to file
                stream_audio_chunks(str(output_path), audio_chunks, tts_model.config.mimi.sample_rate)

                logger.info(f"✓ Generated: {output_path}")
                successful += 1

            except Exception as e:
                logger.error(f"✗ Failed to process {voice_path.name}: {e}")
                failed += 1

        logger.info("-" * 40)
        logger.info(f"Batch processing complete: {successful} successful, {failed} failed")
        logger.info(f"Output directory: {output_dir}")


@cli_app.command()
def gui():
    """Launch the GUI application."""
    from pocket_tts.gui.main_window import main

    main()


# Internal testing commands (not for end users)
@cli_app.command(hidden=True)
def test_preprocessing(text_file: str, output_json: str = None):
    """Internal: Test preprocessing pipeline."""
    import json

    from pocket_tts.config import ConfigManager
    from pocket_tts.preprocessing.chunker import SmartChunker
    from pocket_tts.preprocessing.emotion_analyzer import EmotionAnalyzer
    from pocket_tts.preprocessing.parameter_mapper import ParameterMapper
    from pocket_tts.preprocessing.structure_detector import StructureDetector

    config = ConfigManager.load_config()

    # Read text
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Process
    detector = StructureDetector()
    chunker = SmartChunker(
        mode=config.chunking["mode"],
        min_words=config.chunking["min_words"],
        max_words=config.chunking["max_words"],
    )

    analyzer = EmotionAnalyzer()
    mapper = ParameterMapper()

    structure = detector.analyze(text)
    chunks = chunker.chunk(structure)

    # Analyze emotions
    if chunks:
        texts = [c.text for c in chunks]
        emotions = analyzer.analyze_batch(texts)

        for chunk, emotion in zip(chunks, emotions):
            params = mapper.calculate_params(
                emotion=emotion["emotion"],
                punctuation=chunk.punctuation,
                boundary_type=chunk.boundary_type,
                word_count=chunk.word_count
            )
            
            # Get silence duration (ms) -> convert to seconds for storage
            silence_duration_ms = mapper.calculate_silence_duration_ms(chunk.boundary_type)
            silence_duration_sec = silence_duration_ms / 1000.0
            
            # Convert TTSParams to dict for JSON serialization
            chunk.tts_params = {
                "temperature": params.temperature,
                "frames_after_eos": params.frames_after_eos,
                "eos_threshold": params.eos_threshold,
                "lsd_decode_steps": params.lsd_decode_steps,
            }
            chunk.emotion = emotion["emotion"]
            chunk.emotion_scores = emotion["scores"]
            chunk.emotion_confidence = emotion["confidence"]
            
            # Store post-processing parameters
            chunk.post_process = {
                "silence_duration": silence_duration_sec
            }

    # Output JSON if requested
    if output_json:
        chunk_data = []
        for chunk in chunks:
            chunk_dict = {
                "index": chunk.index,
                "text": chunk.text,
                "word_count": chunk.word_count,
                "boundary_type": chunk.boundary_type.value,
                "punctuation": chunk.punctuation,
                "emotion": chunk.emotion.value if chunk.emotion else None,
                "emotion_confidence": chunk.emotion_confidence,
                "tts_params": chunk.tts_params,
                "post_process": chunk.post_process
            }
            chunk_data.append(chunk_dict)

        with open(output_json, "w") as f:
            json.dump(chunk_data, f, indent=2)

        logger.info(f"Results saved to {output_json}")
    else:
        logger.info(f"Preprocessing complete: {len(chunks)} chunks created")


if __name__ == "__main__":
    cli_app()
