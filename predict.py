# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os

from cog import BasePredictor, Input, Path

import inference


class Predictor(BasePredictor):
    def setup(self):
        inference.do_load("wav2lip_gan.pth")

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
        resize_factor: int = Input(
            description="Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p",
            default=1,
        ),
        face_det_batch_size: int = Input(
            default=16, description="Batch size for face detection"
        ),
        wav2lip_batch_size: int = Input(
            default=128, description="Batch size for Wav2Lip model(s)"
        ),
    ) -> Path:
        try:
            os.remove("results/result_voice.mp4")
        except FileNotFoundError:
            pass

        args = [
            "--checkpoint_path",
            "wav2lip_gan.pth",
            "--face",
            str(face),
            "--audio",
            str(audio),
            "--pads",
            *pads.split(" "),
            "--fps",
            str(fps),
            "--resize_factor",
            str(resize_factor),
            "--face_det_batch_size",
            str(face_det_batch_size),
            "--wav2lip_batch_size",
            str(wav2lip_batch_size),
        ]
        if not smooth:
            args += ["--nosmooth"]

        print("-> args:", " ".join(args))
        inference.args = inference.parser.parse_args(args)
        inference.main()

        return Path("results/result_voice.mp4")
