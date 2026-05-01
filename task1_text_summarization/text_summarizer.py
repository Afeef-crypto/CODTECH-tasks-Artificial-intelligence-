import argparse
import re
from collections import Counter
from typing import List


DEFAULT_TEXT = """
Artificial intelligence is transforming many industries by automating repetitive tasks,
improving decision making, and enabling systems to learn from data. In healthcare, AI helps
in medical image analysis and patient risk prediction. In finance, it supports fraud detection
and algorithmic trading. In education, intelligent tutoring systems personalize learning
experiences for students. Despite these benefits, challenges remain around data privacy,
bias in models, and transparency of AI decisions. Responsible AI development requires robust
evaluation, ethical guidelines, and human oversight. As tools become more accessible, developers
must focus on building systems that are fair, reliable, and aligned with social needs.
""".strip()


def normalize_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def extractive_summary(text: str, max_sentences: int = 3) -> str:
    sentences = normalize_sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    freq = Counter(words)
    if not freq:
        return " ".join(sentences[:max_sentences])

    sentence_scores = []
    for idx, sentence in enumerate(sentences):
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower())
        score = sum(freq[token] for token in tokens) / max(len(tokens), 1)
        sentence_scores.append((idx, score, sentence))

    top = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:max_sentences]
    top_sorted_by_order = sorted(top, key=lambda x: x[0])
    return " ".join(item[2] for item in top_sorted_by_order)


def transformer_summary(text: str, max_length: int = 120, min_length: int = 30) -> str:
    from transformers import pipeline

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return result[0]["summary_text"]


def summarize_text(text: str, prefer_transformer: bool = True) -> str:
    if prefer_transformer:
        try:
            return transformer_summary(text)
        except Exception:
            pass
    return extractive_summary(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Text Summarization Tool")
    parser.add_argument("--text", type=str, default="", help="Input text to summarize")
    parser.add_argument(
        "--show-example",
        action="store_true",
        help="Run summarization on built-in example text",
    )
    parser.add_argument(
        "--disable-transformer",
        action="store_true",
        help="Use extractive summarizer only",
    )
    args = parser.parse_args()

    input_text = args.text.strip()
    if args.show_example or not input_text:
        input_text = DEFAULT_TEXT

    summary = summarize_text(input_text, prefer_transformer=not args.disable_transformer)

    print("\n--- INPUT TEXT ---\n")
    print(input_text)
    print("\n--- SUMMARY ---\n")
    print(summary)


if __name__ == "__main__":
    main()
