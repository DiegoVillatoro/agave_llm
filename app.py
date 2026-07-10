# app.py

import json
import glob
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio as rio
import scipy.ndimage as ndi
from PIL import Image

import matplotlib as mpl
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
from streamlit_plotly_events import plotly_events

from prompts import *
from explain_orthomap import create_descriptions
from graph import HelpdeskGraph, HelpdeskState


# =====================================================
# CONFIG
# =====================================================

TILE_SIZE = 500
PREVIEW_MAX_SIZE = 700
FOLDER_SAVE_AGENTS = r"E:\Experiments\agents\agave"

st.set_page_config(
    page_title="Agave Orthomosaic Sampler",
    layout="wide",
)


# =====================================================
# UTILS
# =====================================================

def normalize_image(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    maxv = img.max()
    if maxv <= 0:
        return img
    return img / maxv


def safe_rgb_from_src(src_rgb: np.ndarray) -> np.ndarray:
    """
    Input expected as H,W,C with channels in raster order.
    Converts to RGB display order.
    """
    if src_rgb.shape[-1] >= 3:
        out = src_rgb[..., [2, 1, 0]]
    else:
        out = src_rgb
    return normalize_image(out)


def confidence_color(confidence):
    confidence = str(confidence).lower()
    if confidence == "low":
        return "#F44336"
    if confidence == "medium":
        return "#FF9800"
    if confidence == "high":
        return "#4CAF50"
    return "#9E9E9E"


def condition_field(condition):
    condition = str(condition).lower()
    if condition == "health":
        return "#4CAF50"
    if condition == "review":
        return "#FF9800"
    if condition == "weak":
        return "#F44336"
    return "#9E9E9E"


def resize_nan_image(image, target_shape):
    """
    Resizes a 2D float array containing NaN values cleanly.
    """
    mask = np.ones_like(image, dtype=np.float32)
    mask[np.isnan(image)] = 0.0

    clean_image = np.nan_to_num(image, nan=0.0).astype(np.float32)
    zoom_factors = [t / o for t, o in zip(target_shape, image.shape)]

    resized_image = ndi.zoom(clean_image, zoom_factors, order=1)
    resized_mask = ndi.zoom(mask, zoom_factors, order=1)

    result = np.divide(
        resized_image,
        resized_mask,
        out=np.full_like(resized_image, np.nan, dtype=np.float32),
        where=resized_mask > 0
    )
    return result


def tile_filename(y, x):
    return f"rgb_{format(y, '05d')}_{format(x, '05d')}.png"


def tile_json_filename(y, x):
    return f"field_report_rgb_{format(y, '05d')}_{format(x, '05d')}.json"


# =====================================================
# CACHE RASTER
# =====================================================

@st.cache_data
def build_preview(path: str):
    with rio.open(path) as img:
        metadata = img.meta
        imgnp = img.read()

        mask_bg = imgnp[-1, ...].astype(np.float32)
        max_mask = mask_bg.max()
        if max_mask > 0:
            mask_bg = mask_bg / max_mask

        st.write(f"Read map {path}")

        img_rgb = imgnp[0:3, ...].transpose(1, 2, 0).astype(np.float32)
        img_rgb = normalize_image(img_rgb)

        scale = max(
            metadata["width"] / PREVIEW_MAX_SIZE,
            metadata["height"] / PREVIEW_MAX_SIZE,
            1,
        )

        preview_w = int(metadata["width"] / scale)
        preview_h = int(metadata["height"] / scale)

        img_preview = cv2.resize(
            img_rgb,
            (preview_w, preview_h),
            interpolation=cv2.INTER_LINEAR
        )

        _, max_gray = np.percentile(img_rgb, [0, 99.99])
        max_gray = max(float(max_gray), 1e-6)

        img_preview = np.clip(img_preview[..., [2, 1, 0]] / max_gray, 0, 1)

    return img_rgb, img_preview, scale, metadata, mask_bg

def list_tiles(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return list(folder.glob("*.png"))


# =====================================================
# TILE / DATASET FUNCTIONS
# =====================================================

def generate_tiles(
    image,
    mask_bg,
    output_dir,
    tile_size=500,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counter = 0
    H, W = mask_bg.shape

    ys_valid = np.where(mask_bg > 0)[0]
    if len(ys_valid) == 0:
        return 0

    ymin = ys_valid.min()
    ymax = ys_valid.max()

    thr_min_pixels = 100
    last_irow = ymax - thr_min_pixels + 1

    for y in range(ymin, last_irow + 1, tile_size):
        row_mask = mask_bg[y:min(y + tile_size, H)]
        xs = np.where(row_mask.any(axis=0))[0]

        if len(xs) == 0:
            continue

        xmin = xs.min()
        xmax = xs.max()
        last_icol = xmax - thr_min_pixels + 1

        for x in range(xmin, last_icol + 1, tile_size):
            im = image[
                y:min(y + tile_size, H),
                x:min(x + tile_size, W),
                :
            ]
            mask_tile = mask_bg[
                y:min(y + tile_size, H),
                x:min(x + tile_size, W)
            ]

            pad_h = tile_size - im.shape[0]
            pad_w = tile_size - im.shape[1]

            if pad_h > 0 or pad_w > 0:
                im = np.pad(
                    im,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="constant"
                )
                mask_tile = np.pad(
                    mask_tile,
                    ((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0
                )

            if mask_tile.mean() < 0.1:
                continue

            im = safe_rgb_from_src(im)
            im_uint8 = (255 * np.clip(im, 0, 1)).astype(np.uint8)

            img_save_path = output_dir / tile_filename(y, x)
            Image.fromarray(im_uint8).save(img_save_path)
            counter += 1

    return counter


# =====================================================
# HEATMAP
# =====================================================

@st.cache_data
def build_heatmap(folder_jsons, bgr, mask, tile_size=500):
    H, W = mask.shape
    heatmap = np.full((H, W), np.nan, dtype=np.float32)
    counter = 0
    tile_info = []

    ys_valid = np.where(mask > 0)[0]
    if len(ys_valid) == 0:
        return heatmap, counter, tile_info

    ymin = ys_valid.min()
    ymax = ys_valid.max()

    thr_min_pixels = 100
    last_irow = ymax - thr_min_pixels + 1

    for y in range(ymin, last_irow + 1, tile_size):
        row_mask = mask[y:min(y + tile_size, H)]
        xs = np.where(row_mask.any(axis=0))[0]

        if len(xs) == 0:
            continue

        xmin = xs.min()
        xmax = xs.max()
        last_icol = xmax - thr_min_pixels + 1

        for x in range(xmin, last_icol + 1, tile_size):
            im = bgr[
                y:min(y + tile_size, H),
                x:min(x + tile_size, W),
                :
            ]
            mask_tile = mask[
                y:min(y + tile_size, H),
                x:min(x + tile_size, W)
            ]

            pad_h = tile_size - im.shape[0]
            pad_w = tile_size - im.shape[1]

            if pad_h > 0 or pad_w > 0:
                im = np.pad(
                    im,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="constant"
                )
                mask_tile = np.pad(
                    mask_tile,
                    ((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0
                )

            if mask_tile.mean() < 0.1:
                continue

            json_file = Path(folder_jsons) / tile_json_filename(y, x)
            if not json_file.exists():
                continue

            counter += 1

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                summary = data[0]["analysis"]["regional_summary"]
                expected = float(summary["score_similarity_expected"])
                anomaly = float(summary["score_anomaly_condition"])

                denom = anomaly + expected
                score = anomaly / denom if denom > 0 else np.nan

                heatmap[y:min(y + tile_size, H), x:min(x + tile_size, W)] = score

                tile_info.append(
                    {
                        "id": counter,
                        "x": x,
                        "y": y,
                        "score": score,
                        "expected": expected,
                        "anomaly": anomaly,
                        "classification": summary.get("condition_classification", "Unknown"),
                        "json_file": str(json_file),
                    }
                )
            except Exception as e:
                st.warning(f"Could not load {json_file.name}: {e}")

    return heatmap, counter, tile_info


@st.cache_data
def create_blended(preview, heatmap, vmax_range, alpha):
    hm = np.nan_to_num(heatmap, nan=0.0)
    hm = np.clip(hm / max(vmax_range, 1e-6), 0, 1)

    cmap = mpl.colormaps["RdYlGn_r"]
    heat_rgba = cmap(hm)

    blended = preview * (1 - alpha) + heat_rgba[..., :3] * alpha
    return np.clip(blended, 0, 1).astype(np.float32)


def create_preview_figure(preview, scale, metadata, tile_size):
    fig = px.imshow(preview)

    fig.update_layout(
        dragmode="pan",
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_showscale=False,
    )

    preview_tile = tile_size / scale

    for x in np.arange(0, preview.shape[1], preview_tile):
        fig.add_shape(
            type="line",
            x0=x, y0=0,
            x1=x, y1=preview.shape[0],
            line=dict(width=1, color="rgba(255,255,255,0.35)")
        )

    for y in np.arange(0, preview.shape[0], preview_tile):
        fig.add_shape(
            type="line",
            x0=0, y0=y,
            x1=preview.shape[1], y1=y,
            line=dict(width=1, color="rgba(255,255,255,0.35)")
        )

    xs = []
    ys = []

    height = metadata["height"]
    width = metadata["width"]

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            xs.append((x + tile_size / 2) / scale)
            ys.append((y + tile_size / 2) / scale)

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(size=10, opacity=0.01),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    return fig


def create_heatmap_figure(blended, scale, tiles_df=None):
    fig = px.imshow(blended)

    fig.update_layout(
        dragmode="pan",
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_showscale=False,
    )

    if tiles_df is not None and len(tiles_df):
        centers_x = ((tiles_df["x"] + TILE_SIZE / 2) / scale).tolist()
        centers_y = ((tiles_df["y"] + TILE_SIZE / 2) / scale).tolist()

        hover_text = [
            f"Tile ID: {row['id']}<br>"
            f"x: {row['x']}<br>"
            f"y: {row['y']}<br>"
            f"score: {row['score']:.3f}<br>"
            f"classification: {row['classification']}"
            for _, row in tiles_df.iterrows()
        ]

        #fig.add_trace(
        #    go.Scatter(
        #        x=centers_x,
        #        y=centers_y,
        #        mode="markers",
        #        marker=dict(
        #            size=12,
        #            opacity=0.04,
        #            color=tiles_df["score"],
        #            colorscale="RdYlGn_r",
        #            showscale=True,
        #            colorbar=dict(title="Anomaly ratio")
        #        ),
        #        text=hover_text,
        #        hovertemplate="%{text}<extra></extra>",
        #        showlegend=False,
        #    )
        #)

    return fig


def render_tile_analysis(data):
    tile = data[0]

    region_id = tile["region_id"]
    x = tile["x"]
    y = tile["y"]

    summary = tile["analysis"]["regional_summary"]

    classification = summary["condition_classification"]
    expected = summary["score_similarity_expected"]
    anomaly = summary["score_anomaly_condition"]

    st.subheader("Tile Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("Region ID", region_id)
    c2.metric("Expected Similarity", f"{expected}%")
    c3.metric("Anomaly Score", f"{anomaly}%")

    st.caption(f"Coordinates: ({x}, {y})")

    if "Expected" in classification:
        st.success(classification)
    elif "Intermediate" in classification:
        st.warning(classification)
    else:
        st.error(classification)

    st.markdown("### Regional Interpretation")
    st.info(summary["overall_reasoning"])

    st.markdown("### Observations")
    observations = tile["analysis"]["observations"]

    for obs in observations:
        with st.expander(
            f"{obs['observation_name']} (Strength {obs['feature_strength_score']}/3)"
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence", obs["confidence"])
            c2.metric("Prevalence", obs["spatial_prevalence"])
            c3.metric("Artifact Risk", obs["artifact_assessment"])

            st.markdown("**Visual Description**")
            st.write(obs["visual_description"])

            features = []
            features.extend(obs.get("canopy_color_features", []))
            features.extend(obs.get("structural_features", []))
            features.extend(obs.get("texture_features", []))

            if features:
                st.markdown("**Detected Features**")
                cols = st.columns(min(4, len(features)))
                for i, feat in enumerate(features):
                    cols[i % len(cols)].success(feat)

            st.markdown("**Scientific Reasoning**")
            st.write(obs["scientific_reasoning"])


# =====================================================
# SIDEBAR INPUT
# =====================================================

st.sidebar.title("Orthomosaic")

orthomap_path = st.sidebar.text_input(
    "GeoTIFF path",
    r"E:\Experiments\Datasets Maps\Zone108_octubre_full.tif"
)

zone_name = orthomap_path.split("\\")[-1].split(".")[0]
zone_name = zone_name.removesuffix("_full")

if not Path(orthomap_path).exists():
    st.warning("Select a valid GeoTIFF.")
    st.stop()

src, preview, scale, metadata, mask_bg = build_preview(orthomap_path)

height = metadata["height"]
width = metadata["width"]

st.sidebar.success(f"{width:,} x {height:,}")

folder_map_root = Path(FOLDER_SAVE_AGENTS) / f"map_{zone_name}"
folder_map_root.mkdir(parents=True, exist_ok=True)

folder_tiles = folder_map_root
folder_jsons = folder_map_root / "Descriptions"
folder_good_samples = folder_map_root / "healthy_samples"
folder_good_samples.mkdir(parents=True, exist_ok=True)

preview_w = int(mask_bg.shape[1] / scale)
preview_h = int(mask_bg.shape[0] / scale)


# =====================================================
# SESSION STATE
# =====================================================

if "selected_tile" not in st.session_state:
    st.session_state.selected_tile = None

if "selected_tile_heatmap" not in st.session_state:
    st.session_state.selected_tile_heatmap = None

if "selected_tile_json_file" not in st.session_state:
    st.session_state.selected_tile_json_file = None

if "heatmap_ready" not in st.session_state:
    st.session_state.heatmap_ready = False

if "heatmap" not in st.session_state:
    st.session_state.heatmap = None

if "heatmap_preview" not in st.session_state:
    st.session_state.heatmap_preview = None

if "tile_info" not in st.session_state:
    st.session_state.tile_info = []

if "sample_tiles" not in st.session_state:
    st.session_state.sample_tiles = []

if "healthy_sample" not in st.session_state:
    st.session_state.healthy_sample = []


# =====================================================
# MAIN PREVIEW
# =====================================================

st.title("Agave Orthomosaic Sampler")

fig_preview = create_preview_figure(preview, scale, metadata, TILE_SIZE)

left, right = st.columns([2, 1])

with left:
    st.subheader("Orthomosaic Preview")

    selected = plotly_events(
        fig_preview,
        click_event=True,
        hover_event=False,
        select_event=False,
        key="preview_grid",
    )

    if selected:
        px_click = selected[0]["x"]
        py_click = selected[0]["y"]

        real_x = px_click * scale
        real_y = py_click * scale

        tile_x = int(real_x // TILE_SIZE)
        tile_y = int(real_y // TILE_SIZE)

        st.session_state.selected_tile = (tile_x, tile_y)

with right:
    st.subheader("Selected Tile")

    tile = st.session_state.selected_tile

    if tile is None:
        st.info("Click a tile.")
    else:
        tile_x, tile_y = tile

        x0 = tile_x * TILE_SIZE
        y0 = tile_y * TILE_SIZE

        patch = src[
            y0:y0 + TILE_SIZE,
            x0:x0 + TILE_SIZE,
            :
        ]
        patch = safe_rgb_from_src(patch)

        st.write(f"Tile: ({tile_x}, {tile_y})")
        st.write(f"Pixel: ({x0}, {y0})")

        st.image(patch, use_container_width=True)

        if st.button("Save as Healthy Sample"):
            patch_uint8 = (np.clip(patch, 0, 1) * 255).astype(np.uint8)

            Image.fromarray(patch_uint8).save(
                folder_good_samples / f"{x0}_{y0}.png"
            )

            with open(folder_good_samples / f"{x0}_{y0}.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "x": x0,
                        "y": y0,
                        "tile_x": tile_x,
                        "tile_y": tile_y,
                        "label": "healthy_reference",
                    },
                    f,
                    indent=2,
                )

            st.success("Sample saved.")


# =====================================================
# TILE DATASET GENERATION
# =====================================================

st.divider()
st.header("Tile Dataset Generation")

col_gen1, col_gen2 = st.columns([1, 1])

with col_gen1:
    if st.button("Generate 500x500 Tiles", use_container_width=True):
        with st.spinner("Generating tiles..."):
            n_tiles = generate_tiles(
                src,
                mask_bg,
                folder_tiles,
                tile_size=TILE_SIZE,
            )
        st.success(f"{n_tiles} tiles generated")

with col_gen2:
    st.info(f"Output folder: {folder_tiles}")

st.divider()
st.header("Generated Tile Preview")

#c1, c2 = st.columns([1, 1])

files = list_tiles(folder_tiles)

#with c1:
if st.button("Show Random Sample"):

    files = list_tiles(folder_tiles)
    st.write(f"{len(files)} tiles found")
    if len(files)>0:
        st.session_state.sample_tiles = random.sample(
            files,
            min(12, len(files))
        )

if st.session_state.sample_tiles:
    cols = st.columns(4)
    for i, tile_file in enumerate(st.session_state.sample_tiles):
        with cols[i % 4]:
            st.image(
                str(tile_file),
                caption=tile_file.stem,
                use_container_width=True,
            )

selected_tile_file = st.selectbox(
        "Inspect tile",
        files,
        format_func=lambda x: x.name,
        index=0 if len(files) else None,
    )
if selected_tile_file:
    st.image(str(selected_tile_file), width=500)
    st.code(selected_tile_file.name)
else:
    st.info("No generated tiles found yet.")


# =====================================================
# PROMPT VALIDATION
# =====================================================

st.divider()

left2, right2 = st.columns([1, 1])

with left2:
    st.header("GOOD DESCRIPTION PROMPT VALIDATION")
    with st.expander("GOOD DESCRIPTION PROMPT", expanded=True):
        st.text_area(
            "Prompt",
            value=GOOD_DESCRIPTION_PROMPT,
            height=600,
            disabled=True,
        )

healthy_images = sorted(folder_good_samples.glob("*.png"))

with right2:
    st.subheader("Healthy Reference Images")
    st.write(f"Found {len(healthy_images)} healthy samples.")

    sample_size = st.slider(
        "Number of examples",
        min_value=0,
        max_value=12,
        value=2,
        step=1,
    )

    if st.button("Refresh Sample"):
        st.session_state.healthy_sample = random.sample(
            healthy_images,
            min(sample_size, len(healthy_images))
        ) if len(healthy_images) else []

    if not st.session_state.healthy_sample and len(healthy_images):
        st.session_state.healthy_sample = random.sample(
            healthy_images,
            min(sample_size, len(healthy_images))
        )

    cols = st.columns(2)
    for i, img_path in enumerate(st.session_state.healthy_sample):
        with cols[i % 2]:
            st.image(
                str(img_path),
                caption=img_path.stem,
                use_container_width=True,
            )


# =====================================================
# PROMPT VIEWER
# =====================================================

st.divider()
st.header("Prompt Viewer")

tab1, tab2 = st.tabs(["ANALYSIS PROMPT", "GOOD DESCRIPTION PROMPT"])

with tab1:
    st.text_area(
        "ANALYSIS PROMPT",
        value=ANALYSIS_PROMPT,
        height=500,
        disabled=True,
    )

with tab2:
    st.text_area(
        "GOOD DESCRIPTION PROMPT",
        value=GOOD_DESCRIPTION_PROMPT,
        height=500,
        disabled=True,
    )


# =====================================================
# EXPLANATION GENERATION
# =====================================================

st.divider()
st.header("Generate tiles explanation")

if st.button(
    "Generate JSON files with Tiles Explanation using ANALYSIS PROMPT",
    use_container_width=True,
):
    with st.spinner("Generating tiles explanation..."):
        n_explains = create_descriptions(folder_tiles, folder_good_samples)

    st.success(f"{n_explains} tiles explanations generated")


# =====================================================
# HEATMAP ANALYSIS
# =====================================================

st.divider()
st.header("Orthomap Heatmap Analysis")

heat_col1, heat_col2, heat_col3 = st.columns([1, 1, 1])

with heat_col1:
    vmax_range = st.slider(
        "Heatmap max range",
        min_value=0.0,
        max_value=1.0,
        value=0.55,
        step=0.05,
    )

with heat_col2:
    alpha = st.slider(
        "Heatmap opacity",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
    )

with heat_col3:
    compute_heatmap_button = st.button(
        "Compute and Visualize Heatmap",
        use_container_width=True
    )

if compute_heatmap_button:
    with st.spinner("Computing heatmap..."):
        heatmap, counter, tile_info = build_heatmap(
            folder_jsons,
            src,
            mask_bg,
            tile_size=TILE_SIZE
        )

        heatmap_preview = resize_nan_image(heatmap, (preview_h, preview_w))

        st.session_state.heatmap = heatmap
        st.session_state.heatmap_preview = heatmap_preview
        st.session_state.tile_info = tile_info
        st.session_state.heatmap_ready = True

    st.success(f"Heatmap computed using {counter} analyzed tiles.")

if st.session_state.heatmap_ready and st.session_state.heatmap_preview is not None:
    heatmap_preview = st.session_state.heatmap_preview
    tile_info = st.session_state.tile_info

    tiles_df = pd.DataFrame(tile_info)
    if len(tiles_df):
        tiles_df["x1"] = tiles_df["x"] + TILE_SIZE
        tiles_df["y1"] = tiles_df["y"] + TILE_SIZE

    blended = create_blended(
        preview,
        heatmap_preview,
        vmax_range,
        alpha,
    )

    col1, col2 = st.columns([1.2, 1])

    fig_heatmap = create_heatmap_figure(
        blended,
        scale,
        tiles_df if len(tile_info) else None
    )

    with col1:
        st.subheader("Orthomap + Heatmap")
        selected2 = plotly_events(
            fig_heatmap,
            click_event=True,
            hover_event=False,
            select_event=False,
            key="heatmap_grid",
        )

        if selected2 and len(tiles_df):
            px_click = selected2[0]["x"]
            py_click = selected2[0]["y"]

            real_x = px_click * scale
            real_y = py_click * scale

            tile_row = tiles_df[
                (tiles_df["x"] <= real_x) & (real_x < tiles_df["x1"]) &
                (tiles_df["y"] <= real_y) & (real_y < tiles_df["y1"])
            ]

            if len(tile_row):
                tile_json_file = tile_row.iloc[0]["json_file"]
                tile_x = int(tile_row.iloc[0]["x"])
                tile_y = int(tile_row.iloc[0]["y"])

                st.session_state.selected_tile_heatmap = (
                    tile_x,
                    tile_y,
                    tile_json_file
                )

    with col2:
        st.subheader("Selected Heatmap Tile")

        tile = st.session_state.selected_tile_heatmap

        if tile is None:
            st.info("Click a tile on the heatmap.")
        else:
            tile_x, tile_y, tile_json_file = tile
            st.session_state.selected_tile_json_file = tile_json_file

            patch = src[
                tile_y:tile_y + TILE_SIZE,
                tile_x:tile_x + TILE_SIZE,
                :
            ]
            patch = safe_rgb_from_src(patch)

            st.write(f"Pixel: ({tile_x}, {tile_y})")
            st.image(patch, use_container_width=True)

            if len(tiles_df):
                row = tiles_df[
                    (tiles_df["x"] == tile_x) &
                    (tiles_df["y"] == tile_y)
                ]
                if len(row):
                    rr = row.iloc[0]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Score", f"{rr['score']:.3f}")
                    c2.metric("Expected", f"{rr['expected']:.1f}")
                    c3.metric("Anomaly", f"{rr['anomaly']:.1f}")
                    st.caption(rr["classification"])

    with st.expander("Heatmap Table", expanded=False):
        if len(tile_info):
            df_show = pd.DataFrame(tile_info).sort_values("score", ascending=False)
            st.dataframe(df_show, use_container_width=True)
        else:
            st.info("No tile information available.")
else:
    st.info("Press 'Compute and Visualize Heatmap' to generate the heatmap overlay.")


# =====================================================
# TILE JSON ANALYSIS
# =====================================================

if st.session_state.selected_tile_json_file:
    json_path = Path(st.session_state.selected_tile_json_file)
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        render_tile_analysis(data)


# =====================================================
# ORTHOMAP RAG ANALYSIS
# =====================================================

st.divider()
st.header("Orthomap RAG Analysis")

tile_status = st.selectbox(
    "Tiles to analyze",
    options=[
        "ALL",
        "Consistent with Expected Condition",
        "Intermediate / Uncertain",
        "Anomalous Condition",
    ],
    index=0,
    help=(
        "Select which tile classifications will be used "
        "to generate the retrieval query and final analysis."
    )
)

n_tiles = len(glob.glob(str(folder_jsons / "*.json")))

if tile_status == "ALL":
    st.info(f"Analysis will use all {n_tiles} generated tiles.")
else:
    st.info(
        f"Analysis will use only tiles classified as '{tile_status}'."
    )

if st.button("Generate RAG Query", use_container_width=True):
    with st.spinner("Generating final analysis..."):
        helpdesk = HelpdeskGraph()
        graph = helpdesk.compilar()
        
        estado_inicial = HelpdeskState(
            consulta=str(folder_jsons),
            tiles_status=tile_status,
            respuesta_rag=None,
            confianza=0.0,
            fuentes=[],
            contexto_rag=None,
            respuesta_final=None,
            historial=[]
        )

        resultado = graph.invoke(estado_inicial)

        st.success(f"{n_tiles} tiles processed")

        if resultado.get("respuesta_final"):
            st.markdown(resultado["respuesta_final"])


# =====================================================
# OVERALL REPORT FILES
# =====================================================

report_files = {
    "All Tiles": "OVERALL_ALL.json",
    "Good Tiles": "OVERALL_Consistent with Expected Condition.json",
    "Intermediate Tiles": "OVERALL_Intermediate - Uncertain.json",
    "Anomalous Tiles": "OVERALL_Anomalous Condition.json",
}

selected_report = st.selectbox(
    "Select report",
    list(report_files.keys()),
    index=0,
)

json_file = folder_map_root / report_files[selected_report]

if json_file.exists():
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    st.success(f"Loaded: {json_file.name}")
else:
    st.warning(f"File not found:\n{json_file}")
    data = None

if data:
    assessment = data["field_assessment"]

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            f"""
            <div style="
            padding:15px;
            border-radius:10px;
            background-color:{condition_field(assessment['condition_field'])};
            color:white;">
            <h3>Condition Severity</h3>
            <h2>{assessment['condition_field'].upper()}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div style="
            padding:15px;
            border-radius:10px;
            background-color:{confidence_color(assessment['confidence'])};
            color:white;">
            <h3>Confidence</h3>
            <h2>{assessment['confidence'].upper()}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Overall Interpretation")
    st.write(assessment["overall_interpretation"])

    st.divider()
    st.header("Observed Phenotypic Patterns")

    def patterns_to_df(patterns, category):
        rows = []
        for p in patterns:
            rows.append(
                {
                    "Category": category,
                    "Pattern": p["name"],
                    "Prevalence (%)": p["prevalence_percent"],
                    "Strength": p["strength_score"],
                }
            )
        return rows

    rows = []
    rows += patterns_to_df(data["observed_patterns"]["color_indicators"], "Color")
    rows += patterns_to_df(data["observed_patterns"]["structural_indicators"], "Structure")
    rows += patterns_to_df(data["observed_patterns"]["texture_indicators"], "Texture")

    patterns_df = pd.DataFrame(rows)
    st.dataframe(patterns_df, use_container_width=True)

    st.markdown(data["observed_patterns"]["summary"])

    st.subheader("Pattern Prevalence")
    prevalence_df = patterns_df.sort_values("Prevalence (%)", ascending=False)
    st.bar_chart(prevalence_df.set_index("Pattern")[["Prevalence (%)"]])

    st.subheader("Pattern Strength")
    strength_df = patterns_df.sort_values("Strength", ascending=False)
    st.bar_chart(strength_df.set_index("Pattern")[["Strength"]])

    st.divider()
    st.header("Plausible Agronomic Explanations")

    hyp_df = pd.DataFrame(
        [
            {
                "Category": h["category"],
                "Support Score": h["evidence_support_score"]
            }
            for h in data["hypotheses"]
        ]
    )

    if len(hyp_df):
        st.bar_chart(hyp_df.set_index("Category"))

    for h in sorted(
        data["hypotheses"],
        key=lambda x: x["evidence_support_score"],
        reverse=True
    ):
        with st.expander(f"{h['category']} ({h['evidence_support_score']}/100)"):
            st.markdown(f"**Hypothesis**: {h['hypothesis']}")
            st.markdown("**Supporting Observations**")
            for obs in h["supporting_observations"]:
                st.write("•", obs)

            st.markdown("**Retrieved Evidence**")
            for ev in h["supporting_retrieved_evidence"]:
                st.write("•", ev)

    st.divider()
    st.header("Uncertainty Analysis")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Major Uncertainties")
        for item in data["uncertainty_analysis"]["major_uncertainties"]:
            st.warning(item)

    with c2:
        st.subheader("Additional Information Needed")
        for item in data["uncertainty_analysis"]["required_additional_information"]:
            st.info(item)

    st.divider()
    st.header("Recommendations")

    priority_order = {
        "HIGH": 0,
        "MEDIUM": 1,
        "LOW": 2,
    }

    recommendations = sorted(
        data["recommendations"],
        key=lambda x: priority_order.get(x["priority"], 99)
    )

    for rec in recommendations:
        if rec["priority"] == "HIGH":
            st.error(f"{rec['priority']} - {rec['action']}")
        elif rec["priority"] == "MEDIUM":
            st.warning(f"{rec['priority']} - {rec['action']}")
        else:
            st.success(f"{rec['priority']} - {rec['action']}")

        st.caption(rec["justification"])

    if "generated_query" in data:
        st.divider()
        with st.expander("Generated Retrieval Query"):
            st.text_area(
                "",
                value=data["generated_query"],
                height=500
            )