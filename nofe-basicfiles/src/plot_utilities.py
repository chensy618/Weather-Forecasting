import napari 
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def plot_results_napari(output_np, gps_np):
    # -----------------------------
    # Input data
    # A: (N, 3)  -> non-normalized RGB channels
    # B: (N, 2)  -> (latitude, longitude)
    # -----------------------------

    # -----------------------------
    # Per-channel normalization
    # -----------------------------
    A=output_np
    A_norm = np.zeros_like(A)
    lat = gps_np[:, 0]
    lon = gps_np[:, 1]

    for c in range(3):
        channel = A[:, c]
        cmin = channel.min()
        cmax = channel.max()
        if cmax > cmin:
            A_norm[:, c] = (channel - cmin) / (cmax - cmin)
        else:
            A_norm[:, c] = 0.0

    # Optional global brightness scaling (prevents additive washout)
    V = 0.5
    A_norm = np.clip(A_norm * V, 0, 1)

    # -----------------------------
    # Create regular lat–lon grid
    # -----------------------------

    grid_size = 300  # adjust resolution here

    lat_grid = np.linspace(lat.min(), lat.max(), grid_size)
    lon_grid = np.linspace(lon.min(), lon.max(), grid_size)

    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    points = np.column_stack([lon, lat])

    # -----------------------------
    # Interpolate each channel
    # -----------------------------

    Red = griddata(points, A_norm[:, 0], (lon_mesh, lat_mesh), method='linear')
    Green = griddata(points, A_norm[:, 1], (lon_mesh, lat_mesh), method='linear')
    Blue = griddata(points, A_norm[:, 2], (lon_mesh, lat_mesh), method='linear')

    Red[np.isnan(Red)] = 0.0
    Green[np.isnan(Green)] = 0.0
    Blue[np.isnan(Blue)] = 0.0

    # -----------------------------
    # Stack into 3 separate layers
    # -----------------------------

    image_stack = np.stack([Red, Green, Blue], axis=0)  # shape: (3, H, W)
    colormap_names = ['red', 'green', 'blue']
    opacity_per_layer = 0.75  # reasonable for additive blending

    # -----------------------------
    # Napari visualization
    # -----------------------------

    viewer = napari.Viewer(show=False)

    for i in range(3):
        viewer.add_image(
            image_stack[i],
            name=f'channel_{colormap_names[i]}',
            colormap=colormap_names[i],
            blending='additive',
            opacity=opacity_per_layer
        ).interpolation = 'linear'

    viewer.open()



def plot_results_matplotlib(output_np, gps_np):
    # -----------------------------
    # Input data
    # A: (N, 3)  -> non-normalized RGB channels
    # B: (N, 2)  -> (latitude, longitude)
    # -----------------------------
    
    A = output_np
    lat = gps_np[:, 0]
    lon = gps_np[:, 1]

    # -----------------------------
    # Per-channel normalization
    # -----------------------------
    A_norm = np.zeros_like(A, dtype=float)

    for c in range(3):
        channel = A[:, c]
        cmin = channel.min()
        cmax = channel.max()
        if cmax > cmin:
            A_norm[:, c] = (channel - cmin) / (cmax - cmin)
        else:
            A_norm[:, c] = 0.0

    # Global brightness scaling
    V = 0.5
    A_norm = np.clip(A_norm * V, 0.0, 1.0)

    # -----------------------------
    # Create regular lat–lon grid
    # -----------------------------
    grid_size = 300

    lat_grid = np.linspace(lat.min(), lat.max(), grid_size)
    lon_grid = np.linspace(lon.min(), lon.max(), grid_size)

    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    points = np.column_stack([lon, lat])

    # -----------------------------
    # Interpolate each channel
    # -----------------------------
    R = griddata(points, A_norm[:, 0], (lon_mesh, lat_mesh), method="linear")
    G = griddata(points, A_norm[:, 1], (lon_mesh, lat_mesh), method="linear")
    B = griddata(points, A_norm[:, 2], (lon_mesh, lat_mesh), method="linear")

    R = np.nan_to_num(R)
    G = np.nan_to_num(G)
    B = np.nan_to_num(B)

    # -----------------------------
    # Stack into RGB image
    # -----------------------------
    rgb_image = np.stack([R, G, B], axis=-1)  # (H, W, 3)
    rgb_image = np.clip(rgb_image, 0.0, 1.0)

    # -----------------------------
    # Matplotlib visualization
    # -----------------------------
    fig, ax = plt.subplots(figsize=(4, 4))

    ax.imshow(
        rgb_image,
        extent=[
            lon_grid.min(), lon_grid.max(),
            lat_grid.min(), lat_grid.max()
        ],
        origin="lower",
        interpolation="bilinear"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Interpolated RGB Map")

    fig.tight_layout()
    plt.close(fig)        # close before returning
    return fig