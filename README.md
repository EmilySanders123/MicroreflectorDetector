# Microreflector Detector Introduction
This program identifies, stores, and matches reflective particles embedded in dendritic identifiers and secure text.  It is intended as a demonstration of technology that can identify labels by matching the pattern of reflective microparticles embedded in a label to a point constellation already on file. 

# Git Structure
The final version of this project is stored on the *main* branch.  This code is identical to the code in the *RatioComparisonMethod*.  The code implementing the ORB matching method and the constellation neighborhood matching method are stored in the *ORBMethod* branch and the *ConstellationNeighborhoodMethod* branch respectively.

The .exe file is based off of the code in the *main* branch.  To run code from other branches, please follow the "Running with Python" instructions below.

# To Run
### Setup
Open a terminal in the directory where the MicroreflectorMatching.exe file is stored.  You can open a Windows terminal in a specific directory by opening File Explorer, navigating to the desired folder, right-clicking, and selecting "Open in Terminal".

### Usage
The program can be run with the following base command: `\.MicroreflectorMatching.exe`

Usage: `.\MicroreflectorMatching.exe [-h] (-s STORE | -m MATCH | -d DISPLAY | -t TEST | -tr TEST_RATIO)`

**Options:**

*Only one flag may be used*

| Flag             | Value                   | Description                                                                                                        |
|------------------|-------------------------|--------------------------------------------------------------------------------------------------------------------|
| -h/--help        | N/A                     | Provides a description of the program and an explanation of all flags                                              |
| -s/--store       | Image filepath          | Stores the constellation information of the given image                                                            |
| -m/--match       | Image filepath          | Calculates the constellation of the given image and finds the best match in the stored constellations on record    |
| -d/--display     | Stored constellation ID | Displays the selected constellation with matching ID as points on a graph                                                             |
| -t/--test        | Image filepath          | Graphically represents all star generation steps of the given image, from the raw image to the final constellation |
| -tr/--test_ratio | Image filepath          | Graphically represents the ratio generation for all stars in the given image                                       |

# Running with Python
### Setup
1. Download or clone the project from GitHub.
2. If you do not have Python already installed, download and install it from [this website](https://www.python.org/downloads/).  If you are unsure of how to install, follow [this tutorial](https://realpython.com/installing-python/).
2. Once Python is installed, open a command-line terminal in the project's directory.  On Windows, you can do this by opening the folder in the File Explorer, right-clicking, and selecting "Open in Terminal".
3. Run the following command to install the required Python packages: `pip install -r requirements.txt`

### Usage
The program can be run with the following base command: `python MicroreflectorMatching.py`

**Options**
Program options are the same as indicated above for the .exe file.