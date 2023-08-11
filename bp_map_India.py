# Границы Индия
# https://github.com/AnujTiwari/India-State-and-Country-Shapefile-Updated-Jan-2020
# Importing required Libraries
import geopandas as gpd
import pandas as pd
import folium
import branca
import requests
import json
from folium.features import GeoJson, GeoJsonTooltip, GeoJsonPopup

gdf = gpd.read_file('states_india.shp')
with open('states_india_1.json') as response:
    india = json.load(response)
#Creating a custom tile (optional)
import branca
# Create a white image of 4 pixels, and embed it in a url.
white_tile = branca.utilities.image_to_url([[1, 1], [1, 1]])

f = folium.Figure(width=680, height=750)
m = folium.Map([23.53, 78.3], maxZoom=6, minZoom=4.8, zoom_control=True, zoom_start=5,
               scrollWheelZoom=True, maxBounds=[[40, 68], [6, 97]], tiles=white_tile, attr='white tile',
               dragging=True).add_to(f)
# Add layers for Popup and Tooltips
popup = GeoJsonPopup(
    fields=['st_nm', 'cartodb_id'],
    aliases=['State', "Data points"],
    localize=True,
    labels=True,
    style="background-color: yellow;",
)
tooltip = GeoJsonTooltip(
    fields=['st_nm', 'cartodb_id'],
    aliases=['State', "Data points"],
    localize=True,
    sticky=False,
    labels=True,
    style="""
        background-color: #F0EFEF;
        border: 1px solid black;
        border-radius: 3px;
        box-shadow: 3px;
    """,
    max_width=800,
)
# Add choropleth layer
g = folium.Choropleth(
    geo_data=india,
    data=gdf,
    columns=['st_nm', 'cartodb_id'],
    key_on='properties.st_nm',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.4,
    legend_name='Data Points',
    highlight=True,

).add_to(m)
folium.GeoJson(
    india,
    style_function=lambda feature: {
        'fillColor': '#ffff00',
        'color': 'black',
        'weight': 0.2,
        'dashArray': '5, 5'
    },
    tooltip=tooltip,
    popup=popup).add_to(g)
f