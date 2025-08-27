# Traffic Signal Violation Detection System using Computer Vision

![Violation_Detection_Frame](Images/detected%20frame.png)

## Introduction
This software helps build a system from scratch, enhancing understanding of system architecture, computer vision, GUIs using Python's Tkinter, and OpenCV. It offers valuable experience for anyone looking to dive into computer vision and real-time system development.

## Table of Contents
- [Traffic Signal Violation Detection System using Computer Vision](#traffic-signal-violation-detection-system-using-computer-vision)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Preface](#preface)
  - [Objectives](#objectives)
  - [Techniques](#techniques)
    - [Vehicle Classification](#vehicle-classification)
    - [Violation Detection](#violation-detection)
  - [Applications](#applications)
    - [Computer Vision](#computer-vision)
    - [Graphical User Interface](#graphical-user-interface)
  - [Authors](#authors)

## Preface
Traffic violations, especially in congested areas, are becoming more serious globally. This project provides a real-time solution to detect traffic signal violations, helping authorities monitor roads more effectively and prevent accidents. The system accurately enforces traffic laws by detecting violations and providing real-time feedback.

## Objectives
The primary goal is to automate traffic signal violation detection, making it easier for the traffic police department to monitor and respond to violations effectively. The system focuses on detecting and tracking vehicles to ensure accurate violation detection.

## Techniques
### Vehicle Classification
We identify moving objects in video footage using the YOLOv3 object detection model. YOLOv3 is a powerful algorithm for object detection, using the Darknet-53 architecture to classify vehicles accurately.

![Darknet Architecture](Images/Darknet53.png)

### Violation Detection
YOLOv3 is used to detect cars, and violations are identified by drawing a traffic line in the video footage, representing a red light signal. Any vehicle crossing this line is flagged as a violation, with bounding boxes changing color from green to red upon detection.

## Applications
### Computer Vision
The system utilizes OpenCV for image processing and TensorFlow to implement the vehicle classifier model using Darknet-53.

### Graphical User Interface
A user-friendly GUI built using Tkinter allows the administrator to interact with the system without needing to modify code. Features like opening video footage, drawing traffic lines, and visualizing detected violations are all integrated into the interface.

![Initial View](Images/initial%20view.jpg)  
*Figure 1: Initial user interface view*

The administrator opens video footage, draws a traffic line, and the system begins detecting violations. Output is displayed frame by frame, and the results are saved as an `output.mp4` file.

![Selected Region](Images/original%20frame.jpg)  
*Figure 2: Region of Interest - Drawing signal line*

## Authors
- **Padilam Kalyan Kumar**  
  *Email*: kalyanpadilam@gmail.com  
