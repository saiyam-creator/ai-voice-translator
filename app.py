import gradio as gr
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator

# Load model once
model = WhisperModel("base", device="cpu", compute_type="int8")


def transcribe(audio, input_lang, output_lang):

    if audio is None:
        return "No audio provided", "", ""

    # Whisper transcription
    segments, info = model.transcribe(
        audio,
        language=None if input_lang == "auto" else input_lang,
    )

    text = " ".join([s.text for s in segments]).strip()
    detected = info.language if info else "unknown"

    # Translation
    translated = ""
    if output_lang != "none" and output_lang != detected:
        try:
            translated = GoogleTranslator(
                source=detected, target=output_lang
            ).translate(text)
        except Exception as e:
            translated = f"Translation error: {e}"

    return text, detected, translated


langs = {
    "auto": "Auto Detect",
    "en": "English",
    "hi": "Hindi",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "ar": "Arabic",
    "none": "No Translation",
}


app = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(type="filepath", label="🎤 Speak"),
        gr.Dropdown(list(langs.keys()), value="auto", label="Input Language"),
        gr.Dropdown(list(langs.keys()), value="none", label="Output Language"),
    ],
    outputs=[
        gr.Textbox(label="Transcription"),
        gr.Textbox(label="Detected Language"),
        gr.Textbox(label="Translation"),
    ],
    title="🎙️ AI Voice Translator",
    description="Speak in any language → get transcription + translation",
)


if __name__ == "__main__":
    app.launch()
