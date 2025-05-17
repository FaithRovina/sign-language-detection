Sign Language Detection System - Deployment Guide
=================================================

GitHub Repository: https://github.com/FaithRovina/sign-language-detection

System Overview
---------------
This is a multi-modal accessibility assistant system that combines real-time sign language detection, speech recognition, and visual aids. The system uses YOLOv5 for hand gesture detection and integrates with Google Cloud Speech-to-Text services.

System Requirements
------------------
1. Hardware:
   - Webcam for video input
   - Microphone for audio input
   - Minimum 8GB RAM
   - Minimum 2GB free disk space

2. Software:
   - Python 3.7 or higher
   - Anaconda/Miniconda (recommended)
   - Git
   - Google Chrome/Firefox (for running the web interface)

Installation Instructions
------------------------
1. Clone the Repository:
   ```bash
   git clone https://github.com/FaithRovina/sign-language-detection.git
   cd sign-language-detection
   ```

2. Create and Activate Conda Environment:
   ```bash
   conda create -n sign-dec python=3.7 -y
   conda activate sign-dec
   ```

3. Install Required Packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Install YOLOv5:
   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   cd yolov5
   pip install -r requirements.txt
   cd ..
   ```

5. Install Additional Dependencies:
   ```bash
   pip install opencv-python-headless
   pip install streamlit
   ```

6. Set Up Google Cloud Credentials:
   - Create a Google Cloud project
   - Enable Speech-to-Text API
   - Create service account and download credentials JSON
   - Place the credentials file in the project root directory
   - Rename it to `credentials.json`

Running the Application
----------------------
1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to:
   http://localhost:8501

Application Features
-------------------
1. Sign Language Detection:
   - Real-time hand gesture recognition
   - Video upload capability
   - Webcam integration

2. Speech Recognition:
   - Real-time audio transcription
   - Audio file upload
   - Text-to-speech conversion

3. Visual Aids:
   - Image processing
   - Real-time video processing
   - Accessibility features

Testing the Application
-----------------------
1. Sign Language Detection:
   - Open the application in your browser
   - Navigate to the Sign Language page
   - Start webcam feed to test real-time detection
   - Upload test videos to verify video processing

2. Speech Recognition:
   - Go to the Speech page
   - Start audio recording
   - Test with different speech inputs
   - Upload audio files for transcription

Troubleshooting
--------------
1. Webcam Issues:
   - Ensure no other applications are using the webcam
   - Check camera permissions in your operating system

2. Audio Issues:
   - Verify microphone permissions
   - Check audio device settings
   - Ensure no background noise

3. Model Loading Issues:
   - Check internet connection for model downloads
   - Verify sufficient disk space
   - Ensure correct Python version (3.7+)

Support
-------
For any issues or questions, please create an issue on the GitHub repository:
https://github.com/FaithRovina/sign-language-detection/issues

Contributing
------------
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

License
-------
This project is licensed under the MIT License - see the LICENSE file for details.
