# 2D SLAM Project

This project implements a 2D Simultaneous Localization and Mapping (SLAM) algorithm using FastSLAM with particle filters. It processes 2D LiDAR scan data to create an occupancy grid map of the environment while simultaneously estimating the robot's trajectory.

## Features
- **Occupancy Grid Mapping**
- **Scan Matching** for improved pose estimation
- **FastSLAM** algorithm implementation
- **Particle filter** for robot pose belief representation

## Requirements
- Python 3.9.13
- `rasterio`
- `numpy`
- `matplotlib`
- `Pillow`
- `scipy`

## Installation
Clone the repository and install the required packages by running:

```bash
pip install -r requirements.txt
```
## Dataset
The project uses the Intel Research Lab dataset collected by Dirk Hähnel. The dataset includes:
- 910 readings
- 180° Field of View 2D LiDAR data
- Wheel odometry data (x, y, theta)
- Ground truth data

## Algorithm Overview

### Occupancy Grid
- Represents the environment as a grid of cells
- Each cell contains a probability of being occupied
- Updates based on LiDAR readings

### Scan Matching
- Improves pose estimation using LiDAR scans
- Uses a multi-resolution strategy for efficiency
- Performs coarse-to-fine matching

### FastSLAM
- Approximates robot pose distribution using particles
- Combines scan matching and motion model
- Uses importance sampling and particle filtering
- Produces a globally consistent map with loop closure

## Results
The algorithm successfully produces a globally consistent map and achieves correct loop closure when the robot revisits previously mapped areas. The final output includes:
- Occupancy grid map of the environment
- Estimated robot trajectory
- Generated images of the estimated map from SLAM at various stages
- GIF animation created from the sequence of generated map images
- Final map converted into a TIFF format using the longitude and latitude of the Intel Research Lab for real-world coordinate representation

## Reference
- Intel Research Lab SLAM dataset provided by Dirk Hähnel
- Qin Zou, Qin Sun, Long Chen, Bu Nie, and Qingquan Li, *A Comparative Analysis of LiDAR SLAM-based Indoor Navigation for Autonomous Vehicles*, IEEE Transactions on Intelligent Transportation Systems, 2020