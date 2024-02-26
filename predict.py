# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
import subprocess

from cog import BasePredictor, Input, Path

import inference

from time import time

from functools import wraps
import torch


def make_mem_efficient(cls: BasePredictor):
    if not torch.cuda.is_available():
        return cls

    old_setup = cls.setup
    old_predict = cls.predict

    @wraps(old_setup)
    def new_setup(self, *args, **kwargs):
        ret = old_setup(self, *args, **kwargs)
        _move_to(self, "cpu")
        return ret

    @wraps(old_predict)
    def new_predict(self, *args, **kwargs):
        _move_to(self, "cuda")
        try:
            ret = old_predict(self, *args, **kwargs)
        finally:
            _move_to(self, "cpu")
        return ret

    cls.setup = new_setup
    cls.predict = new_predict

    return cls

#g-text to speech
import tempfile
from gtts import gTTS
from pathlib import Path
from datetime import datetime
import random

def text_to_speech(text, language):
    tts = gTTS(text=text, lang=language)
    # Get the current date and time
    current_datetime = datetime.now()
    # Format the date and time as a string (excluding the year)
    datetime_str = current_datetime.strftime("%m%d%H%M%S")
    # Generate a random 5-digit number
    random_number = random.randint(10000, 99999)
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Construct the audio file name using the current date and time and the random number
        audio_path = Path(temp_dir) / f"audio_{datetime_str}_{random_number}.mp3"
        tts.save(audio_path)
        print (audio_path)
        return audio_path

###

def _move_to(self, device):
    try:
        self = self.cached_models
    except AttributeError:
        pass
    for attr, value in vars(self).items():
        try:
            value = value.to(device)
        except AttributeError:
            pass
        else:
            print(f"Moving {self.__name__}.{attr} to {device}")
            setattr(self, attr, value)
    torch.cuda.empty_cache()


@make_mem_efficient
class Predictor(BasePredictor):
    cached_models = inference

    def setup(self):
        inference.do_load("checkpoints/wav2lip_gan.pth")

    def predict(
        self,
        face: Path = Input(description="video/image that contains faces to use"),
        text: str = Input(description="Text to convert to speech", default="I am a teacher"),
        pads: str = Input(
            description="Padding for the detected face bounding box.\n"
            "Please adjust to include chin at least\n"
            'Format: "top bottom left right"',
            default="0 10 0 0",
        ),
        smooth: bool = Input(
            description="Smooth face detections over a short temporal window",
            default=True,
        ),
        fps: float = Input(
            description="Can be specified only if input is a static image",
            default=25.0,
        ),
        out_height: int = Input(
            description="Output video height. Best results are obtained at 480 or 720",
            default=480,
        ),
    ) -> Path:
        try:
            os.remove("results/result_voice.mp4")
        except FileNotFoundError:
            pass

        face_ext = os.path.splitext(face)[-1]
        if face_ext not in [".mp4", ".mov", ".png" , ".jpg" , ".jpeg" , ".gif", ".mkv", ".webp"]:
            raise ValueError(f'Unsupported face format {face_ext!r}')

        if not text.strip():
            raise ValueError("Text input cannot be empty")
        
        language = "en"    
        audio = text_to_speech(text, language)    
        
        audio_ext = os.path.splitext(audio)[-1]
        
        if not os.path.exists(audio):
            raise ValueError(f'Text was not converted to audio properly, contact developer')

        args = [
            "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
            "--face", str(face),
            "--audio", str(audio),
            "--pads", *pads.split(" "),
            "--fps", str(fps),
            "--out_height", str(out_height),
        ]
        if not smooth:
            args += ["--nosmooth"]

        print("-> run:", " ".join(args))
        inference.args = inference.parser.parse_args(args)

        s = time()

        try:
            inference.main()
        except ValueError as e:
            print('-> Encountered error, skipping lipsync:', e)

            args = [
                "ffmpeg", "-y",
                # "-vsync", "0", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                "-stream_loop", "-1",
                "-i", str(face),
                "-i", str(audio),
                "-shortest",
                "-fflags", "+shortest",
                "-max_interleave_delta", "100M",
                "-map", "0:v:0",
                "-map", "1:a:0",
                # "-c", "copy",
                # "-c:v", "h264_nvenc",
                "results/result_voice.mp4",
            ]
            print("-> run:", " ".join(args))
            print(subprocess.check_output(args, encoding="utf-8"))

        print(time() - s)

        return Path("results/result_voice.mp4")
