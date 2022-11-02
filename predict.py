# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os

from cog import BasePredictor, Input, Path

import inference

from time import time


class Predictor(BasePredictor):
    def setup(self):
        inference.do_load("checkpoints/wav2lip_gan.pth")

    def predict(
        self,
        face: Path = Input(description="video/image that contains faces to use"),
        audio: Path = Input(description="video/audio file to use as raw audio source"),
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

        print("-> args:", " ".join(args))
        inference.args = inference.parser.parse_args(args)
        s = time()
        inference.main()
        print(time() - s)

        return Path("results/result_voice.mp4")
