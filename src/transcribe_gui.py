import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from pathlib import Path

import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

SAMPLE_RATE = 16000
SCRIPT_DIR = Path(__file__).parent.parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / "outputs" / "stage2" / "best_model"


class SpeechToTextModel:
    
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def transcribe(self, audio: np.ndarray) -> str:
        if len(audio) == 0:
            return ""
        
        max_val = max(abs(audio.max()), abs(audio.min()))
        if max_val > 0:
            audio = audio / max_val
        
        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        
        input_values = inputs.input_values.to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return transcription


class SpeechToTextApp:
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🎤 Speech to Text")
        self.root.geometry("700x550")
        self.root.resizable(True, True)
        
        self.model = None
        self.is_recording = False
        self.audio_buffer = []
        self.stream = None
        
        self._setup_styles()
        self._create_widgets()
        self._load_model_async()
    
    def _setup_styles(self):
        style = ttk.Style()
        style.configure("Record.TButton", font=("Segoe UI", 12, "bold"))
        style.configure("Stop.TButton", font=("Segoe UI", 12, "bold"))
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 10))
    
    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = ttk.Label(
            main_frame,
            text="🎤 Speech to Text",
            style="Title.TLabel"
        )
        title_label.pack(pady=(0, 15))
        
        self.status_var = tk.StringVar(value="Loading model...")
        self.status_label = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            style="Status.TLabel"
        )
        self.status_label.pack(pady=(0, 10))
        
        self.progress = ttk.Progressbar(
            main_frame,
            mode="indeterminate",
            length=300
        )
        self.progress.pack(pady=(0, 15))
        self.progress.start()
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        self.record_btn = ttk.Button(
            button_frame,
            text="🎙️ Start Recording",
            command=self._toggle_recording,
            style="Record.TButton",
            width=20
        )
        self.record_btn.pack(side=tk.LEFT, padx=5)
        self.record_btn.config(state=tk.DISABLED)
        
        self.clear_btn = ttk.Button(
            button_frame,
            text="🗑️ Clear",
            command=self._clear_text,
            width=12
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.file_btn = ttk.Button(
            button_frame,
            text="📁 Load Audio",
            command=self._load_audio_file,
            width=12
        )
        self.file_btn.pack(side=tk.LEFT, padx=5)
        self.file_btn.config(state=tk.DISABLED)
        
        duration_frame = ttk.Frame(main_frame)
        duration_frame.pack(pady=10, fill=tk.X)
        
        ttk.Label(duration_frame, text="Recording duration (seconds):").pack(side=tk.LEFT)
        
        self.duration_var = tk.IntVar(value=5)
        self.duration_slider = ttk.Scale(
            duration_frame,
            from_=2,
            to=30,
            variable=self.duration_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        self.duration_slider.pack(side=tk.LEFT, padx=10)
        
        self.duration_label = ttk.Label(
            duration_frame,
            textvariable=self.duration_var,
            width=3
        )
        self.duration_label.pack(side=tk.LEFT)
        
        text_frame = ttk.LabelFrame(main_frame, text="Transcription", padding="10")
        text_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.text_area = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            font=("Consolas", 12),
            height=12
        )
        self.text_area.pack(fill=tk.BOTH, expand=True)
        
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        device_label = ttk.Label(
            main_frame,
            text=f"Device: {device}",
            font=("Segoe UI", 9),
            foreground="gray"
        )
        device_label.pack(side=tk.BOTTOM, pady=(10, 0))
    
    def _load_model_async(self):
        def load():
            try:
                model_path = Path(DEFAULT_MODEL_PATH)
                if not model_path.exists():
                    self.root.after(0, lambda: self._on_model_error(
                        f"Model not found at {model_path}"
                    ))
                    return
                
                self.model = SpeechToTextModel(str(model_path))
                self.root.after(0, self._on_model_loaded)
                
            except Exception as e:
                self.root.after(0, lambda: self._on_model_error(str(e)))
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def _on_model_loaded(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.status_var.set("✅ Ready - Click 'Start Recording' to begin")
        self.record_btn.config(state=tk.NORMAL)
        self.file_btn.config(state=tk.NORMAL)
    
    def _on_model_error(self, error: str):
        self.progress.stop()
        self.progress.pack_forget()
        self.status_var.set(f"❌ Error: {error}")
        messagebox.showerror("Model Error", f"Failed to load model:\n{error}")
    
    def _toggle_recording(self):
        if not self.is_recording:
            self._start_recording()
        else:
            self._stop_recording()
    
    def _start_recording(self):
        self.is_recording = True
        self.audio_buffer = []
        
        duration = self.duration_var.get()
        self.record_btn.config(text="⏹️ Stop Recording")
        self.status_var.set(f"🔴 Recording... ({duration}s)")
        
        def record():
            try:
                audio = sd.rec(
                    int(duration * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype='float32'
                )
                sd.wait()
                
                if self.is_recording:
                    self.audio_buffer = audio.flatten()
                    self.root.after(0, self._on_recording_complete)
                    
            except Exception as e:
                self.root.after(0, lambda: self._on_recording_error(str(e)))
        
        thread = threading.Thread(target=record, daemon=True)
        thread.start()
    
    def _stop_recording(self):
        self.is_recording = False
        sd.stop()
        self.record_btn.config(text="🎙️ Start Recording")
        self.status_var.set("⏹️ Recording stopped")
    
    def _on_recording_complete(self):
        self.is_recording = False
        self.record_btn.config(text="🎙️ Start Recording")
        self.status_var.set("🔄 Transcribing...")
        
        def transcribe():
            try:
                if len(self.audio_buffer) == 0:
                    self.root.after(0, lambda: self._on_transcription_complete(""))
                    return
                
                if np.abs(self.audio_buffer).max() < 0.01:
                    self.root.after(0, lambda: self._on_transcription_complete(
                        "[No speech detected - audio too quiet]"
                    ))
                    return
                
                text = self.model.transcribe(self.audio_buffer)
                self.root.after(0, lambda: self._on_transcription_complete(text))
                
            except Exception as e:
                self.root.after(0, lambda: self._on_transcription_error(str(e)))
        
        thread = threading.Thread(target=transcribe, daemon=True)
        thread.start()
    
    def _on_transcription_complete(self, text: str):
        self.status_var.set("✅ Ready")
        
        if text.strip():
            current_text = self.text_area.get("1.0", tk.END).strip()
            if current_text:
                self.text_area.insert(tk.END, "\n\n")
            
            self.text_area.insert(tk.END, text)
            self.text_area.see(tk.END)
    
    def _on_recording_error(self, error: str):
        self.is_recording = False
        self.record_btn.config(text="🎙️ Start Recording")
        self.status_var.set(f"❌ Recording error")
        messagebox.showerror("Recording Error", f"Failed to record:\n{error}")
    
    def _on_transcription_error(self, error: str):
        self.status_var.set(f"❌ Transcription error")
        messagebox.showerror("Transcription Error", f"Failed to transcribe:\n{error}")
    
    def _clear_text(self):
        self.text_area.delete("1.0", tk.END)
    
    def _load_audio_file(self):
        filetypes = [
            ("Audio files", "*.wav *.flac *.mp3 *.ogg *.m4a"),
            ("WAV files", "*.wav"),
            ("FLAC files", "*.flac"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=filetypes
        )
        
        if not filepath:
            return
        
        self.status_var.set(f"🔄 Loading {Path(filepath).name}...")
        
        def process():
            try:
                audio, sr = sf.read(filepath, dtype='float32')
                
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                
                if sr != SAMPLE_RATE:
                    import torchaudio
                    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
                    resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                    audio = resampler(audio_tensor).squeeze().numpy()
                
                text = self.model.transcribe(audio)
                
                self.root.after(0, lambda: self._on_file_transcribed(
                    Path(filepath).name, text
                ))
                
            except Exception as e:
                self.root.after(0, lambda: self._on_transcription_error(str(e)))
        
        thread = threading.Thread(target=process, daemon=True)
        thread.start()
    
    def _on_file_transcribed(self, filename: str, text: str):
        self.status_var.set("✅ Ready")
        
        current_text = self.text_area.get("1.0", tk.END).strip()
        if current_text:
            self.text_area.insert(tk.END, "\n\n")
        
        self.text_area.insert(tk.END, f"[{filename}]\n{text}")
        self.text_area.see(tk.END)


def main():
    root = tk.Tk()
    app = SpeechToTextApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
