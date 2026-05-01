# CODTECH AI Internship Tasks

This repository contains individual implementations for all four Artificial Intelligence internship tasks.

## Project Structure

- `task1_text_summarization/`
  - `text_summarizer.py`: Summarizes long text with NLP techniques.
- `task2_speech_to_text/`
  - `speech_to_text.py`: Transcribes short audio clips.
- `task3_neural_style_transfer/`
  - `neural_style_transfer.py`: Applies artistic style to photographs.
- `task4_text_generation/`
  - `text_generation_notebook.ipynb`: Notebook for topic-based text generation.
- `requirements.txt`: Python dependencies for all tasks.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

3. (Optional, but recommended) verify installation:

```bash
python -c "import torch, torchvision, speech_recognition, transformers; print('Dependencies OK')"
```

## Run Each Task Individually

### Task 1: Text Summarization

**Description:**  
This task builds a Natural Language Processing tool to summarize lengthy article text into concise output.  
The implementation supports two summarization strategies:
- **Transformer summarization** using `facebook/bart-large-cnn` for high-quality abstractive summaries.
- **Extractive fallback summarization** using sentence scoring and word-frequency ranking when transformer models are unavailable.

**What it demonstrates:** Input long text, process it with NLP techniques, and output a shorter meaningful summary.

```bash
python task1_text_summarization/text_summarizer.py --show-example
```

Or summarize custom text:

```bash
python task1_text_summarization/text_summarizer.py --text "Your long article text here."
```

Or enter text interactively:

```bash
python task1_text_summarization/text_summarizer.py --interactive
```

### Task 2: Speech to Text

**Description:**  
This task implements a basic speech recognition system that converts short audio clips into text.  
It uses the `SpeechRecognition` library and supports:
- **Sphinx engine** for offline transcription.
- **Google engine** for online transcription.

**What it demonstrates:** Loading audio, running automatic speech recognition, and printing transcription output.

```bash
python task2_speech_to_text/speech_to_text.py --audio path/to/audio.wav --engine sphinx
```

Available engines: `sphinx` (offline) and `google` (online API).

If your audio has a wrong extension (for example MP3 saved as `.wav`), convert first:

```bash
ffmpeg -y -i input_audio.wav -ac 1 -ar 16000 task2_speech_to_text/output_converted.wav
python task2_speech_to_text/speech_to_text.py --audio task2_speech_to_text/output_converted.wav --engine sphinx
```

### Task 3: Neural Style Transfer

**Description:**  
This task applies the artistic style of one image to the content of another image using a deep learning model.  
The implementation uses pretrained **VGG19** feature maps, computes style/content losses, and optimizes a target image to create stylized output.

**What it demonstrates:** Computer vision with neural networks, feature extraction, and image generation through optimization.

```bash
python task3_neural_style_transfer/neural_style_transfer.py --content path/to/content.jpg --style path/to/style.jpg --output styled_output.jpg
```

Example command with sample files in this repo:

```bash
python task3_neural_style_transfer/neural_style_transfer.py --content task3_neural_style_transfer/sample_content.jpg --style task3_neural_style_transfer/sample_style.jpg --output task3_neural_style_transfer/styled_output.jpg --steps 100
```

### Task 4: Text Generation

**Description:**  
This task demonstrates generative text modeling using **GPT-2** in a notebook environment.  
Users provide a prompt, and the model generates coherent paragraph-style text using sampling controls such as `temperature`, `top_k`, and `top_p`.

**What it demonstrates:** Prompt-based language generation and interactive experimentation in a notebook.

Open and run:

`task4_text_generation/text_generation_notebook.ipynb`

Or launch directly from terminal:

```bash
jupyter notebook task4_text_generation/text_generation_notebook.ipynb
```

The notebook demonstrates generated text from user prompts using GPT-2.
It now asks for prompt input when you run the generation cell.