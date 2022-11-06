import pandas as pd
import numpy as np
import os


class get_fred(vintage="current"):
    def __init__(self):
        self.vintage = vintage
