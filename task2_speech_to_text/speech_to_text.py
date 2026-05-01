import argparse
import pathlib
import speech_recognition as sr


def transcribe_audio(audio_path: str, engine: str = "sphinx") -> str:
    recognizer = sr.Recognizer()
    path = pathlib.Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with sr.AudioFile(str(path)) as source:
        audio = recognizer.record(source)

    if engine == "google":
        return recognizer.recognize_google(audio)
    if engine == "sphinx":
        return recognizer.recognize_sphinx(audio)
    raise ValueError("Unsupported engine. Use 'sphinx' or 'google'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic Speech-to-Text System")
    parser.add_argument("--audio", type=str, required=True, help="Path to .wav/.aiff/.flac audio file")
    parser.add_argument(
        "--engine",
        type=str,
        default="sphinx",
        choices=["sphinx", "google"],
        help="Recognition engine: sphinx (offline) or google (online)",
    )
    args = parser.parse_args()

    try:
        text = transcribe_audio(args.audio, args.engine)
        print("\n--- TRANSCRIPTION ---\n")
        print(text)
    except sr.UnknownValueError:
        print("Could not understand the audio.")
    except sr.RequestError as exc:
        print(f"Recognition service error: {exc}")
    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
