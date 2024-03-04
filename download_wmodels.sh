#!/usr/bin/env bash

mkdir ./checkpoints

#### download the new links.
wget -nc https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/lipsync_expert.pth -O  ./checkpoints/lipsync_expert.pth
wget -nc https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/s3fd-619a316812.pth -O  ./checkpoints/s3fd-619a316812.pth
wget -nc https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/visual_quality_disc.pth -O  ./checkpoints/visual_quality_disc.pth
wget -nc https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip.pth -O  ./checkpoints/wav2lip.pth
wget -nc https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth -O  ./checkpoints/wav2lip_gan.pth

