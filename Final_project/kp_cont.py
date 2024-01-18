import numpy as np
import csv
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt

import camera_tools as ct

from cmac2 import CMAC

#Calibrate the camera to detect green box, if you haven't done this calibration before
low_green, high_green = ct.colorpicker()
print(low_green)
print(high_green)