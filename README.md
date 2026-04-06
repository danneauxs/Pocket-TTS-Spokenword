# DNXS-Spokenword Pocket-TTS

# Emotion-Driven Audiobook Generator

<img width="1446" height="622" alt="pocket-tts-logo-v2-transparent" src="https://github.com/user-attachments/assets/637b5ed6-831f-4023-9b4c-741be21ab238" />

An enhanced version of Kyutai's Pocket TTS that transforms plain text into emotionally expressive audiobooks. Uses advanced AI emotion analysis to create natural, expressive narration with intelligent text chunking and voice adaptation.

** DISCORD https://discord.gg/6hs3r3Wp

**✨ Key Features:**

- **Emotion Analysis**: Automatic detection of emotions in text using DistilRoBERTa
- **Smart Chunking**: Intelligent text segmentation based on sentence structure
- **Expressive TTS**: Emotion-aware parameter control for natural voice variation
- **Audiobook Generation**: Complete pipeline for book-to-audio conversion
- **Voice Cloning**: Custom voice support with emotion preservation
- **GUI Interface**: User-friendly desktop application for easy audiobook creation

Supports Python 3.10, 3.11, 3.12, 3.13 and 3.14. Requires PyTorch 2.5+. CPU-only operation.

## Main takeaways

* **Emotion-Driven Generation**: AI-powered emotion analysis for expressive narration
* **Smart Text Processing**: Intelligent chunking and structure detection
* **Audiobook Pipeline**: Complete book-to-audio conversion system
* **GUI Application**: Desktop interface for easy audiobook creation
* Runs on CPU only (no GPU required)
* Small model size, 100M parameters + emotion analysis
* Faster than real-time, ~4-6x speed on modern CPUs
* Uses 1-4 CPU cores optimally
* Voice cloning with emotion preservation
* English text support

## Trying it from the website, without installing anything (origanl model interface NOT this GUI program)

Navigate to the https://kyutai.org/tts to try basic TTS functionality directly in your browser. You can input text, select different voices, and generate speech without any installation.

## Audiobook Generation Features

# Installation

How to install:
This project uses uv, an extremely fast Python package manager.

1. Install uv:
   - **Linux & Windows (WSL)**: Open your terminal and run:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
     Note: After installation, you may need to restart your terminal or run `source $HOME/.local/bin/env` to add uv to your PATH.
   - **Windows (PowerShell)**: If you are using standard Windows PowerShell (not WSL), run:
     ```powershell
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```

```bash
instasll.sh
```

**This will download all necessary files.  You will probably need to go to the [kyutai/pocket-tts · Hugging Face](https://huggingface.co/kyutai/pocket-tts) page to accept TOS to download the model.  I suggest going there FIRST before running install.  FYI on first conversion there will be a delay as the model is downloaded.  This will take time depending on connection spped.

Note:  if using Windows with WSL you might need to install C++ runtime etc.  Use AI to help you if needed.

### Desktop GUI Application

Launch the full-featured audiobook generator with emotion analysis:

```bash
launch.sh
```

**Features:**

- Customizable chunking and emotion settings
- Progress tracking
- Voice cloning from audio prompts
- Individual chunk regeneration for chunk correction

### Smart Text Processing

- **Structure Detection**: Automatic chapter, paragraph, and sentence boundary detection
- **Emotion Analysis**: 6 emotion classes (joy, sadness, anger, fear, surprise, neutral)
- **Intelligent Chunking**: Configurable word limits with boundary respect
- **Parameter Mapping**: Emotion-to-TTS parameter conversion for expressive speech

### Configuration System

Create custom configurations for different content types:

```yaml
# Fiction books - high emotion sensitivity
emotion:
  sensitivity: 1.5
  keyword_boost: 0.3
chunking:
  mode: sentence
  max_words: 80

# Technical docs - neutral, structured
emotion:
  sensitivity: 0.7
chunking:
  mode: paragraph
  max_words: 120
```

Modify the voice with `--voice` and the text with `--text`. We provide a small catalog of voices.

You can take a look at [this page](https://huggingface.co/kyutai/tts-voices) which details the licenses
for each voice.

* [alba](https://huggingface.co/kyutai/tts-voices/blob/main/alba-mackenna/casual.wav)
* [marius](https://huggingface.co/kyutai/tts-voices/blob/main/voice-donations/Selfie.wav)
* [javert](https://huggingface.co/kyutai/tts-voices/blob/main/voice-donations/Butter.wav)
* [jean](https://huggingface.co/kyutai/tts-voices/blob/main/ears/p010/freeform_speech_01.wav)
* [fantine](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p244_023.wav)
* [cosette](https://huggingface.co/kyutai/tts-voices/blob/main/expresso/ex04-ex02_confused_001_channel1_499s.wav)
* [eponine](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p262_023.wav)
* [azelma](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p303_023.wav)

The `--voice` argument can also take a plain wav file as input for voice cloning.
Feel free to check out the [generate documentation](https://github.com/kyutai-labs/pocket-tts/tree/main/docs/generate.md) for more details and examples.
For trying multiple voices and prompts quickly, prefer using the `serve` command.

## New Features in This Enhanced Version

✨ **Emotion-Driven Audiobook Generation**

- AI-powered emotion analysis using DistilRoBERTa
- Intelligent text chunking with boundary respect
- Emotion-aware TTS parameter control
- Complete audiobook pipeline with progress tracking
- GUI desktop application
- Resume capability for interrupted generations
- Custom configuration system

## Supported and Enhanced Features

✅ **Core TTS**: All original PocketTTS functionality maintained
✅ **Emotion Analysis**: 6 emotion classes with configurable sensitivity
✅ **Smart Chunking**: Sentence/paragraph modes with word limits
✅ **Voice Cloning**: Enhanced with emotion preservation
✅ **GUI Interface**: Full desktop application for audiobook creation
✅ **Batch Processing**: Multiple file processing support
✅ **Configuration**: YAML-based customization system

## ## Prohibited use

Use of our model and enhanced audiobook generation features must comply with all applicable laws and regulations and must not result in, involve, or facilitate any illegal, harmful, deceptive, fraudulent, or unauthorized activity. Prohibited uses include, without limitation:

- Voice impersonation or cloning without explicit and lawful consent
- Misinformation, disinformation, or deception (including fake news, fraudulent calls, or presenting generated content as genuine recordings of real people or events)
- The generation of unlawful, harmful, libelous, abusive, harassing, discriminatory, hateful, or privacy-invasive content
- Copyright infringement through unauthorized audiobook generation
- Automated processing of content without proper licensing

**Audiobook Generation Notice**: This enhanced version is designed for personal and lawful audiobook creation. Users are responsible for ensuring they have appropriate rights to convert text content to audio format. Commercial distribution of generated audiobooks may require additional licensing.

We disclaim all liability for any non-compliant use.

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

Copyright 2024 [Your Name]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Authors

Manu Orsini*, Simon Rouard*, Gabriel De Marmiesse*, Václav Volhejn, Neil Zeghidour, Alexandre Défossez

*equal contribution
