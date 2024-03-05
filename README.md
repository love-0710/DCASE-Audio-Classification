## DCASE 2021 Task 1 - Audio Classification

This repository contains the code and resources for the DCASE 2021 Task 1 - Audio Classification project conducted as part of the Detection and Classification of Acoustic Scenes and Events (DCASE) challenge.
Dataset Information

The development dataset used for this task is the TAU Urban Acoustic Scenes 2020 Mobile dataset. This dataset contains recordings from 12 European cities covering 10 different acoustic scenes using 4 different devices. Additionally, synthetic data for 11 mobile devices was created based on the original recordings.
## Acoustic Scenes

- Airport
- Indoor shopping mall
- Metro station
- Pedestrian street
- Public square
- Street with medium level of traffic
- Travelling by a tram
- Travelling by a bus
- Travelling by an underground metro
- Urban park

## Recording Devices

- Device A: Soundman OKM II Klassik/studio A3 + Zoom F8 audio recorder
- Device B: Samsung Galaxy S7
- Device C: iPhone SE
- Device D: GoPro Hero5 Session

## Dataset Links

- [30 GB Dataset](https://zenodo.org/records/3819968)
- [11 GB Dataset](https://zenodo.org/records/4767109)

## Project Overview

- The goal of this project was to develop an audio classification model to classify the acoustic scenes present in the audio recordings. Initially, a smaller subset of the dataset (11 GB) was used with basic parameters, achieving an accuracy of 58%.
- Later, attempts were made to improve the model accuracy by using the full 30 GB dataset and adjusting model parameters, but only marginal improvements were achieved, reaching a maximum accuracy of around 80% so you can try this code with 30 GB data you will definetely achieved the good accuracy.
## Repository Contents

- code/: Contains the Python code for the audio classification model.
- resources/: Includes documentation, reports, and additional resources related to the project.
- data/: Placeholder directory for storing dataset files.
- models/: Placeholder directory for saving trained models.

## Clone this repository
    git clone https://github.com/love-0710/DCASE-Audio-Classification.git

## Usage

- Download the dataset from the provided links and place it in the data/ directory.
- Follow the instructions in the documentation provided in the resources/ directory to set up the environment and run the code.
- Experiment with different parameters and configurations to improve the model performance.

## Acknowledgments

This project was conducted as part of the DCASE 2021 challenge Task-1 and was made possible by the dataset provided by Tampere University of Technology. The research received funding from the European Research Council (grant agreement 637422 EVERYSOUND).
## Authors

- Love Kumar YAdav

## License

This project is licensed under the MIT License.
