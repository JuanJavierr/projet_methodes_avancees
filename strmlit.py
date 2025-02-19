import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import pyproj

# Define the transformation using NAD83 / MTM Zone 8 (EPSG:32188)
proj_mtm8 = pyproj.Transformer.from_crs("EPSG:32188", "EPSG:4326", always_xy=True)



def main():

    df = pd.read_csv("data_final.csv", sep=";")
    df["x"], df["y"] = proj_mtm8.transform(df["y"].values, df["x"].values)

    st.title("Interactive Intersection Map")

    # Ensure columns exist
    required_cols = {'rue_1','rue_2','x','y','acc','pi'}
    if not required_cols.issubset(df.columns):
        st.write("CSV missing one or more required columns.")
        return

    # Compute the center of the map
    lat_center = np.mean(df["x"])
    lon_center = np.mean(df["y"])

    # Allow user to select which metric to scale radius
    metric_for_radius = st.selectbox(
        "Select data column to scale marker size:",
        ["acc", "pi"]
    )

    # Allow user to select which metric to determine color
    metric_for_color = st.selectbox(
        "Select data column to color markers:",
        ["acc", "pi"]
    )

    # Pick maximum value to normalize radius
    max_val = df[metric_for_radius].max()
    # Avoid zero or negative scaling
    if max_val <= 0:
        max_val = 1

    # Define a color range function (basic numeric mapping)
    def color_range(value):
        # e.g. map low values to green, high to red
        # scale value to 0-255 range
        ratio = min(value / df[metric_for_color].max(), 1.0)
        r = int(255 * ratio)
        g = int(255 * (1 - ratio))
        b = 50  # constant
        return [r, g, b]

    # Create a column for color in the dataframe
    df["color"] = df[metric_for_color].apply(color_range)

    # Build the ScatterplotLayer
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        pickable=True,
        get_position="[y, x]",  # note order is [lon, lat]
        get_radius=f"{metric_for_radius} * 10" if metric_for_radius == "acc" else f"{metric_for_radius} / 10",
        get_fill_color="[color[0], color[1], color[2], 180]"
    )

    # Set up the map view
    view_state = pdk.ViewState(
        latitude=lat_center,
        longitude=lon_center,
        zoom=11,
        pitch=0
    )

    # Tooltip template
    tooltip = {
        "html": "<b>{rue_1} & {rue_2}</b><br/>"
                "Accidents: {acc}<br/>"
                "Ped. Flow: {pi}",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }

    # Render the Deck
    r = pdk.Deck(
        layers=[scatter_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/light-v9"
    )

    st.pydeck_chart(r)

if __name__ == "__main__":
    main()
