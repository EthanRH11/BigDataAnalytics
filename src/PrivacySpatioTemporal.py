import pandas as pd
import geopandas as gdp
import numpy as np
import matplotlib.pyplot as plt
import folium
from sklearn.cluster import DBSCAN
from shapely.geometry import Point

import pydp as dp

class PrivacySpatioTemporal:
    def __init__(self, epsilon = 1.0, delta=1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.data = None

    def loadData(self, filepath):
        if filepath.endswith('csv'):
            df = pd.read_csv('.csv'):

            geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
            self.data = gdp.GeoDataFrame(df, geometry=geometry)
        return self.data
