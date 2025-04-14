SYNTHETIC MOTION ADDITION TO IMAGE TO CREATE REAL VIDEO DENOISING DATASET (24VI20IIITK)

Project Overview

This project generates synthetic video data by learning to add realistic motion to static images .
It is designed to assist in training models for video denoising and related video understanding tasks.
The system is based on MAGE+, which builds upon a VQ-VAE architecture and an autoregressive transformer for motion modeling.


Base repository:
// link to github paper 

Steps to build model:
VQ-VAE Training
MAGE Training

 Result trained model:
 models/MAGE+/catergenv2_diverse/model_best.pth

 Steps to execute:
 01) upload the .yuv file and reference avi file in root directory
 02) root dir>& .venv/Scripts/python.exe load_image.py
 03) Enter 0 for avi 1 for yuv as per the input file type
 04) Make a separate folder in base directory as Video 
 05) The output will be stored in Video folder

 
 
 
 
