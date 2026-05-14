import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import rasterio
from rasterio.windows import Window
import json
import tensorflow as tf
from scipy.ndimage import label as scipy_label
from scipy.stats import mode
import warnings
warnings.filterwarnings('ignore')

# Patch Keras BatchNormalization loading for old saved models
_original_bn_from_config = tf.keras.layers.BatchNormalization.from_config

@classmethod
def _patched_bn_from_config(cls, config):
    config = dict(config)
    config.pop("renorm", None)
    config.pop("renorm_clipping", None)
    config.pop("renorm_momentum", None)
    return _original_bn_from_config(config)

tf.keras.layers.BatchNormalization.from_config = _patched_bn_from_config

print('Imports loaded successfully')

DATA_PATH = '/work_beegfs/sungg1179/'
MODEL_PATH = DATA_PATH + 'unet_model.keras'
NORM_PARAMS_PATH = DATA_PATH + 'unet_norm_params.json'

TILE_SIZE = 128
INFER_STRIDE = 64
NUM_MONTHS = 5
NUM_BANDS = 17
TOTAL_BANDS = NUM_MONTHS * NUM_BANDS
NUM_CLASSES = 4
CLASS_NAMES = ['Cereals', 'Canola', 'Soybeans', 'Corn']
MIN_PATCH_SIZE = 50
PRED_BATCH = 32

SITES = {
    'Pembina': [
        DATA_PATH + 'Pembina/Pembina_composite_2024_5.tif',
        DATA_PATH + 'Pembina/Pembina_composite_2024_6.tif',
        DATA_PATH + 'Pembina/Pembina_composite_2024_7.tif',
        DATA_PATH + 'Pembina/Pembina_composite_2024_8.tif',
        DATA_PATH + 'Pembina/Pembina_composite_2024_9.tif'
    ],
    'Sheridan': [
        DATA_PATH + 'Sheridan/Sheridan_composite_2024_5.tif',
        DATA_PATH + 'Sheridan/Sheridan_composite_2024_6.tif',
        DATA_PATH + 'Sheridan/Sheridan_composite_2024_7.tif',
        DATA_PATH + 'Sheridan/Sheridan_composite_2024_8.tif',
        DATA_PATH + 'Sheridan/Sheridan_composite_2024_9.tif'
    ],
    'Morris': [
        DATA_PATH + 'Morris/Morris_composite_2024_5.tif',
        DATA_PATH + 'Morris/Morris_composite_2024_6.tif',
        DATA_PATH + 'Morris/Morris_composite_2024_7.tif',
        DATA_PATH + 'Morris/Morris_composite_2024_8.tif',
        DATA_PATH + 'Morris/Morris_composite_2024_9.tif'
    ],
    'Rose': [
        DATA_PATH + 'Rose/Rose_ROI_composite_2024_5.tif',
        DATA_PATH + 'Rose/Rose_ROI_composite_2024_6.tif',
        DATA_PATH + 'Rose/Rose_ROI_composite_2024_7.tif',
        DATA_PATH + 'Rose/Rose_ROI_composite_2024_8.tif',
        DATA_PATH + 'Rose/Rose_ROI_composite_2024_9.tif'
    ]
}

_model = None
_norm_params = None


def get_sites():
    return list(SITES.keys())


def load_model_and_norm():
    global _model, _norm_params

    if _model is None:
        print('\n' + '=' * 60)
        print('LOADING MODEL AND NORMALIZATION PARAMETERS')
        print('=' * 60)
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
        print(f'Model loaded from {MODEL_PATH}')

    if _norm_params is None:
        with open(NORM_PARAMS_PATH, 'r') as f:
            raw = json.load(f)
        _norm_params = {
            'mean': np.array(raw['mean'], dtype=np.float32),
            'std': np.array(raw['std'], dtype=np.float32)
        }
        print(f'Normalization params loaded from {NORM_PARAMS_PATH}')
        print(f'Norm params shape: mean={_norm_params["mean"].shape}, std={_norm_params["std"].shape}')

    return _model, _norm_params


def run_inference(site_name):
    if site_name not in SITES:
        raise ValueError(f"Unknown site '{site_name}'. Valid sites: {get_sites()}")

    model, norm_params = load_model_and_norm()
    tif_paths = SITES[site_name]

    print('\n' + '=' * 60)
    print(f'INFERENCE: {site_name.upper()}')
    print('=' * 60)

    missing = [p for p in tif_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f'Missing files for {site_name}: {missing}')

    with rasterio.open(tif_paths[0]) as src:
        profile = src.profile
        height = src.height
        width = src.width

    print(f'Image dimensions: {height} x {width}')

    hann_1d = np.hanning(TILE_SIZE).astype(np.float32)
    hann_2d = np.outer(hann_1d, hann_1d)
    hann_2d = np.clip(hann_2d, 0.01, None)

    all_tile_rows = list(range(0, height - TILE_SIZE + 1, INFER_STRIDE))
    all_tile_cols = list(range(0, width - TILE_SIZE + 1, INFER_STRIDE))

    if all_tile_rows[-1] + TILE_SIZE < height:
        all_tile_rows.append(height - TILE_SIZE)
    if all_tile_cols[-1] + TILE_SIZE < width:
        all_tile_cols.append(width - TILE_SIZE)

    tile_positions = [(tr, tc) for tr in all_tile_rows for tc in all_tile_cols]
    total_tiles = len(tile_positions)

    print(f'Tile size: {TILE_SIZE}x{TILE_SIZE}, Inference stride: {INFER_STRIDE}')
    print(f'Total tiles (with overlap): {total_tiles}')

    prob_sum = np.zeros((height, width, NUM_CLASSES), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    nodata_mask = np.zeros((height, width), dtype=bool)

    tile_count = 0
    srcs = [rasterio.open(tif) for tif in tif_paths]

    try:
        for batch_start in range(0, total_tiles, PRED_BATCH):
            batch_end = min(batch_start + PRED_BATCH, total_tiles)
            batch_positions = tile_positions[batch_start:batch_end]
            n_batch = len(batch_positions)

            batch_tiles = np.zeros((n_batch, TILE_SIZE, TILE_SIZE, TOTAL_BANDS), dtype=np.float32)

            for t, (tr, tc) in enumerate(batch_positions):
                window = Window(tc, tr, TILE_SIZE, TILE_SIZE)

                for m, src in enumerate(srcs):
                    tile_data = src.read(window=window).transpose(1, 2, 0)
                    nan_pixels = np.any(np.isnan(tile_data), axis=-1)
                    nodata_mask[tr:tr+TILE_SIZE, tc:tc+TILE_SIZE] |= nan_pixels

                    for b in range(NUM_BANDS):
                        band_idx = m * NUM_BANDS + b
                        tile_data[nan_pixels, b] = norm_params['mean'][band_idx]

                    batch_tiles[t, :, :, m*NUM_BANDS:(m+1)*NUM_BANDS] = tile_data
                    del tile_data

            batch_tiles -= norm_params['mean']
            batch_tiles /= norm_params['std']

            preds = model.predict(batch_tiles, verbose=0)
            del batch_tiles

            for t, (tr, tc) in enumerate(batch_positions):
                for c in range(NUM_CLASSES):
                    prob_sum[tr:tr+TILE_SIZE, tc:tc+TILE_SIZE, c] += preds[t, :, :, c] * hann_2d
                weight_sum[tr:tr+TILE_SIZE, tc:tc+TILE_SIZE] += hann_2d

            tile_count += n_batch
            if tile_count % (PRED_BATCH * 5) == 0 or tile_count == total_tiles:
                pct = tile_count / total_tiles * 100
                print(f'  {tile_count}/{total_tiles} tiles done ({pct:.1f}%)')

            del preds

    finally:
        for s in srcs:
            s.close()

    print('Computing window-weighted average...')
    weight_sum[weight_sum == 0] = 1
    for c in range(NUM_CLASSES):
        prob_sum[:, :, c] /= weight_sum

    classified = np.argmax(prob_sum, axis=-1).astype(np.uint8) + 1
    del prob_sum, weight_sum

    classified[nodata_mask] = 0

    out_profile = profile.copy()
    out_profile.update(count=1, dtype='uint8', nodata=0, compress='lzw')

    output_path = DATA_PATH + f'UNet_{site_name}_Classified.tif'
    with rasterio.open(output_path, 'w', **out_profile) as dst:
        dst.write(classified, 1)

    print(f'Classified map saved to {output_path}')

    print(f'Post-processing: removing patches < {MIN_PATCH_SIZE} pixels...')
    classified_clean = classified.copy()
    total_cleaned = 0

    for cls in range(1, NUM_CLASSES + 1):
        class_mask = (classified == cls)
        labeled_array, num_features = scipy_label(class_mask)

        if num_features == 0:
            continue

        patch_sizes = np.bincount(labeled_array.ravel(), minlength=num_features + 1)
        small_patches = np.where(patch_sizes[1:] < MIN_PATCH_SIZE)[0] + 1

        for patch_id in small_patches:
            patch_mask = (labeled_array == patch_id)
            patch_size = patch_sizes[patch_id]

            rows, cols = np.where(patch_mask)
            r_min = max(0, rows.min() - 3)
            r_max = min(height, rows.max() + 4)
            c_min = max(0, cols.min() - 3)
            c_max = min(width, cols.max() + 4)

            neighborhood = classified[r_min:r_max, c_min:c_max]
            patch_local = patch_mask[r_min:r_max, c_min:c_max]
            neighbors = neighborhood[(~patch_local) & (neighborhood > 0)]

            if len(neighbors) > 0:
                replacement = int(mode(neighbors, keepdims=False).mode)
                classified_clean[patch_mask] = replacement
                total_cleaned += patch_size

    classified_clean[nodata_mask] = 0
    print(f'Cleaned {total_cleaned} pixels from patches smaller than {MIN_PATCH_SIZE} pixels')

    output_clean_path = DATA_PATH + f'UNet_{site_name}_Classified_Clean.tif'
    with rasterio.open(output_clean_path, 'w', **out_profile) as dst:
        dst.write(classified_clean, 1)

    print(f'Post-processed map saved to {output_clean_path}')
    print(f'{site_name} complete.')

    return {
        'site': site_name,
        'status': 'complete',
        'raw_output': output_path,
        'clean_output': output_clean_path,
        'classes': {
            '0': 'NoData',
            '1': 'Cereals',
            '2': 'Canola',
            '3': 'Soybeans',
            '4': 'Corn'
        }
    }


if __name__ == '__main__':
    for site in get_sites():
        run_inference(site)

    print('\n' + '=' * 60)
    print('ALL SITES COMPLETE')
    print('=' * 60)
    print('Classes: 0=NoData, 1=Cereals, 2=Canola, 3=Soybeans, 4=Corn')
