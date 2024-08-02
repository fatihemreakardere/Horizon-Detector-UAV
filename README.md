
# Horizon-Detector-UAV

Horizon-Detector-UAV is a project developed for the UAV team "Sarkan". This project is designed to detect the horizon line from either a camera feed or a video file. It also includes scripts for LAN communication to send and receive data, optimized for use with a Raspberry Pi 4 B.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Running from Camera](#running-from-camera)
  - [Running from Video](#running-from-video)
- [LAN Communication](#lan-communication)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Horizon-Detector-UAV.git
   cd Horizon-Detector-UAV
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running from Camera

To run the horizon detection using a camera feed, execute the following command:

```bash
python main_from_camera.py
```

### Running from Video

To run the horizon detection using a video file, execute the following command:

```bash
python main_from_video.py --video videos/v1.mp4
```

## LAN Communication

The `RP4` folder contains scripts for LAN communication, specifically designed for use with a Raspberry Pi 4 B.

### Sending Data

To send data over LAN, use the `LAN_send.py` script:

```bash
python RP4/LAN_send.py
```

### Receiving Data

To receive data over LAN, use the `LAN_recive.py` script:

```bash
python RP4/LAN_recive.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
