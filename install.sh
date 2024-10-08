#!/bin/bash

# Update package list and install ffmpeg
sudo apt-get update
sudo apt-get install -y ffmpeg

# Install Python dependencies
pip install -r requirements.txt
