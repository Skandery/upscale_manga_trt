import os
import time
import numpy as np
import random
import torch
import pyvips
from PIL import Image
from trt_utilities import Engine

from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2 # Added import for cv2
import torchvision.transforms as transforms
import timm
from tqdm import tqdm
import math # For isnan
import logging



# ==============================================================================
# Configuration (Moved inside function or passed as args where appropriate)
# ==============================================================================

# Determine script directory for relative paths (like ICC profiles)
CURRENT_FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# --- Classification Configuration ---
# These could be parameters of process_upscale if needed
CLASSIFICATION_MODEL_PATH = os.path.join(CURRENT_FILE_DIRECTORY, "./models/color_detector-97.62-regnety_320.swag_ft_in1k.pth")
CLASSIFICATION_MODEL_NAME = "regnety_320.swag_ft_in1k"
CLASSIFICATION_INPUT_SIZE = 384
CLASSIFICATION_NUM_CLASSES = 2
CLASSIFICATION_THRESHOLD_LOW = 0.46 # Lower bound for uncertain class (Prob Class 0)
CLASSIFICATION_THRESHOLD_HIGH = 0.53 # Upper bound for uncertain class (Prob Class 0)
ENABLE_TILING = True # Set to False to disable tiling globally if needed
TILE_SIZE = 1024     # Input tile size (adjust based on VRAM, must be <= 1280)
TILE_OVERLAP = 64    # Overlap between tiles (pixels on input)
TILE_MODEL_MAX_DIM = 1280 # Max input dimension the 4x engine supports

# --- Upscaling Model Definitions ---
# Manga 2x models (Provided by user)
MANGA_2X_MODELS = [
    {"name": "Manga 1200p 2x", "trt":  os.path.join(CURRENT_FILE_DIRECTORY, "./models/2x_MangaJaNai_1200p_V1_ESRGAN_70k.trt"), "input_height": 1200, "scale": 2},
    {"name": "Manga 1300p 2x", "trt":  os.path.join(CURRENT_FILE_DIRECTORY, "./models/2x_MangaJaNai_1300p_V1RC1_ESRGAN_75k.trt"), "input_height": 1300, "scale": 2},
    {"name": "Manga 1400p 2x", "trt":  os.path.join(CURRENT_FILE_DIRECTORY, "./models/2x_MangaJaNai_1400p_V1RC3_ESRGAN_70k.trt"), "input_height": 1400, "scale": 2},
    {"name": "Manga 1500p 2x", "trt":  os.path.join(CURRENT_FILE_DIRECTORY, "./models/2x_MangaJaNai_1500p_V1RC1_ESRGAN_90k.trt"), "input_height": 1500, "scale": 2},
    {"name": "Manga 1600p 2x", "trt":  os.path.join(CURRENT_FILE_DIRECTORY, "./models/2x_MangaJaNai_1600p_V1RC1_ESRGAN_90k.trt"), "input_height": 1600, "scale": 2},
    {"name": "Manga 1920p 2x", "trt":  os.path.join(CURRENT_FILE_DIRECTORY, "./models/2x_MangaJaNai_1920p_V1RC1_ESRGAN_70k.trt"), "input_height": 1920, "scale": 2},
    {"name": "Manga 2048p 2x", "trt":  os.path.join(CURRENT_FILE_DIRECTORY, "./models/2x_MangaJaNai_2048p_V1RC1_ESRGAN_95k.trt"), "input_height": 2048, "scale": 2},
]
# Create a set of TRT paths for quick lookup later
MANGA_2X_TRT_PATHS = {model['trt'] for model in MANGA_2X_MODELS}

# Other models (Illustrator 2x, Manga 4x Uncertain)
OTHER_MODELS = {
    "illust_2x": {
        "name": "Illustration 2x (Class 1)",
        "trt": os.path.join(CURRENT_FILE_DIRECTORY, "./models/2x_IllustrationJaNai_V1_ESRGAN_120k.trt"),
        "scale": 2,
        "group_type": "illust_2x" # Added identifier
    },
    "manga_4x": {
        "name": "Manga 4x (Uncertain)",
        "trt": os.path.join(CURRENT_FILE_DIRECTORY, "./models/4x_MangaJaNai_V0.5_100k.trt"),
        "scale": 4,
        "group_type": "manga_4x" # Added identifier
    }
}

# --- Pipeline Configuration ---
MAX_QUEUE_SIZE = 10 # Note: This isn't explicitly used with current ThreadPoolExecutor setup
NUM_LOAD_WORKERS = 4
NUM_SAVE_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Upscaling Dimension Limits ---
MIN_DIM = 64   # Lowered minimum dimension slightly
MAX_DIM = 4096 # Increased max dimension slightly for TRT profile flexibility

# --- Post-Upscaling Resize Configuration ---
TARGET_WIDTH_TALL = 1920
TARGET_WIDTH_WIDE = 3840

# --- ICC Profile Paths ---
GAMMA1ICC_PATH = os.path.join(CURRENT_FILE_DIRECTORY, "./resources/Custom Gray Gamma 1.0.icc")
DOTGAIN20ICC_PATH = os.path.join(CURRENT_FILE_DIRECTORY, "./resources/Dot Gain 20%.icc")

# ==============================================================================
# Helper Functions
# ==============================================================================

def get_image_height(image_path):
    """Quickly get image height using pyvips."""
    try:
        # Use 'sequential' access for potentially faster header reading
        image = pyvips.Image.new_from_file(image_path, access="sequential", fail=False)
        return image.height
    except pyvips.Error as e:
        logging.error(f"Warning: Could not get height for {os.path.basename(image_path)}: {str(e)}")
        return None

def find_nearest_manga_model(image_height, manga_models):
    """Finds the manga model config with the closest input_height."""
    if not manga_models:
        raise ValueError("Manga models list cannot be empty.")

    best_model = min(
        manga_models,
        key=lambda model: abs(model['input_height'] - image_height)
    )
    # Add group_type to the chosen manga model config dynamically
    best_model_copy = best_model.copy()
    best_model_copy['group_type'] = 'manga_2x'
    return best_model_copy


def resize_image_pyvips(vips_image, target_height):
    """Resizes a pyvips image to a target height, maintaining aspect ratio."""
    if vips_image.height == target_height:
        return vips_image # No resize needed
    scale_factor = target_height / vips_image.height
    # Using 'thumbnail' is often efficient for downscaling
    resized_image = vips_image.thumbnail_image(
        int(vips_image.width * scale_factor), # Target width based on height scale
        height=target_height,
        size='down' # Use 'down' for potentially better quality when downscaling
    )
    return resized_image

# ==============================================================================
# Classification Functions
# ==============================================================================

def load_classification_model(model_path, model_name, num_classes, device):
    """Loads the classification model."""
    logging.info(f"Loading classification model: {model_name} from {model_path}")
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device,weights_only=False))
        model.to(device)
        model.eval()
        logging.info("Classification model loaded successfully.")
        return model
    except FileNotFoundError:
        logging.info(f"Error: Classification model file not found at {model_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading classification model: {str(e)}")
        raise

def get_classification_transform(model, input_size):
    """Gets the appropriate transformations for the classification model."""
    cfg = model.default_cfg
    mean = cfg.get('mean', (0.485, 0.456, 0.406)) # Use defaults if not present
    std = cfg.get('std', (0.229, 0.224, 0.225))
    if 'mean' not in cfg or 'std' not in cfg:
         logging.error("Warning: Model default_cfg missing mean/std. Using ImageNet defaults.")

    return transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC), # Specify interpolation
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

@torch.no_grad()
def classify_image(image_path, model, transform, device):
    """Classifies a single image and returns probabilities for class 0 and class 1."""
    try:
        # Use PIL for compatibility with torchvision transforms
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        output = model(image)
        probs = torch.softmax(output, dim=1)
        prob_class0 = probs[0, 0].item()
        prob_class1 = probs[0, 1].item()
        # Check for NaN probabilities which can indicate issues
        if math.isnan(prob_class0) or math.isnan(prob_class1):
            logging.error(f"Warning: NaN probability encountered for {os.path.basename(image_path)}. Skipping classification.")
            return None, None
        return prob_class0, prob_class1
    except Exception as e:
        logging.error(f"Error classifying {os.path.basename(image_path)}: {str(e)}")
        return None, None

def classify_images_in_folder(folder_path, model, transform, device):
    """Classifies all images in a folder and returns a dictionary of results."""
    logging.info(f"\n--- Starting Classification Phase ---")
    t_start_classify = time.time()

    image_extensions = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")
    try:
        all_files = [f for f in os.listdir(folder_path)
                     if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(image_extensions)]
    except FileNotFoundError:
        logging.error(f"Error: Input folder not found: {folder_path}")
        return {}, 0.0, 0, 0 # Return empty results and zero counts/time
    except Exception as e:
        logging.error(f"Error listing files in {folder_path}: {str(e)}")
        return {}, 0.0, 0, 0

    if not all_files:
        logging.error(f"No compatible image files found in {folder_path} for classification.")
        return {}, 0.0, 0, 0

    logging.info(f"Found {len(all_files)} images for classification.")
    classification_results = {}
    classified_count = 0
    skipped_classification = 0

    for filename in tqdm(all_files, desc="Classifying images"):
        image_path = os.path.join(folder_path, filename)
        prob0, prob1 = classify_image(image_path, model, transform, device)
        if prob0 is not None and prob1 is not None:
            classification_results[filename] = (prob0, prob1)
            classified_count += 1
        else:
             skipped_classification +=1

    t_end_classify = time.time()
    classification_time = t_end_classify - t_start_classify
    logging.info(f"--- Classification Phase Complete ---")
    logging.info(f"Successfully classified: {classified_count}")
    logging.info(f"Skipped during classification: {skipped_classification}")
    logging.info(f"Classification time: {classification_time:.2f} seconds")
    return classification_results, classification_time, classified_count, skipped_classification

# ==============================================================================
# Black Level Adjustment Functions
# ==============================================================================

def convert_image_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB or RGBA image to grayscale using OpenCV."""
    if image.ndim == 3:
        if image.shape[2] == 3: # RGB
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4: # RGBA
            return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    elif image.ndim == 2: # Already grayscale
        return image
    else:
        raise ValueError(f"Unsupported image ndim for grayscale conversion: {image.ndim}")

def calculate_manga_black_level(image_paths: list[str], sample_percentage: float = 0.1) -> int | None:
    """
    Calculates the median 'most common dark pixel value' from a sample of manga images.

    Args:
        image_paths: A list of file paths for the manga images in a group.
        sample_percentage: The percentage of images to sample (0.0 to 1.0).

    Returns:
        The median most common dark pixel value (0-255), or None if calculation fails.
    """
    if not image_paths:
        logging.error("Warning: No image paths provided for black level calculation.")
        return None

    # Sample images
    sample_size = max(1, math.ceil(len(image_paths) * sample_percentage))
    # Ensure sample size doesn't exceed available images
    sample_size = min(sample_size, len(image_paths))
    try:
        sampled_files = random.sample(image_paths, sample_size)
    except ValueError as e:
        logging.error(f"Warning: Could not sample images for black level calculation: {str(e)}")
        return None

    logging.info(f"Calculating black level from a sample of {len(sampled_files)} manga images...")

    black_level_values = []
    errors = []

    for img_path in tqdm(sampled_files, desc="Analyzing Black Levels", leave=False):
        try:
            # Load image using OpenCV
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # Load as is first
            if img is None:
                raise IOError(f"cv2.imread failed to load image: {img_path}")

            # Convert to grayscale for analysis
            img_gray = convert_image_to_grayscale(img) # Use the helper function

            # Find the darkest 10% of pixels
            dark_threshold = np.percentile(img_gray, 10)
            # Ensure threshold is not excessively high (e.g., > 100) to avoid pure white areas
            dark_threshold = min(dark_threshold, 100)

            dark_pixels = img_gray[img_gray <= dark_threshold]

            if dark_pixels.size > 0:
                # Find the most common value among these dark pixels
                unique_values, counts = np.unique(dark_pixels, return_counts=True)
                most_common_dark = unique_values[np.argmax(counts)]
                black_level_values.append(int(most_common_dark)) # Store the 0-255 value
            else:
                 # Handle cases where no dark pixels are found below the threshold (e.g., mostly white image)
                 # Assigning 0 seems reasonable, assuming it should be black.
                 black_level_values.append(0)

        except Exception as e:
            logging.error(f"Error processing {os.path.basename(img_path)} for black level: {str(e)}")

    if errors:
        logging.error("Errors occurred during black level analysis:")
        for error in errors[:5]: # Print first few errors
            logging.error(f"  - {error}")
        if len(errors) > 5:
            logging.error(f"  ... and {len(errors) - 5} more errors.")

    if not black_level_values:
        logging.error("Warning: Could not determine black level from any sampled images.")
        return None

    # Calculate median black level across sampled images
    median_black_level = int(np.median(black_level_values))
    logging.info(f"Analysis complete. Determined Median Black Level for Manga Group: {median_black_level} (0-255 range)")

    if median_black_level > 90: # Example threshold
        logging.warning(f"Warning: Calculated median black level ({median_black_level}) seems high. Check image samples.")

    return median_black_level


def _apply_black_level_adjustment(image_array: np.ndarray, black_level: int) -> np.ndarray:
    """Applies black level adjustment to a NumPy array (handles grayscale or RGB)."""
    if black_level <= 0:
        return image_array # No adjustment needed
    if black_level >= 255:
        # If black level is 255, the entire image becomes black
        return np.zeros_like(image_array, dtype=np.uint8)

    # Promote to float32 for calculation precision
    img_float = image_array.astype(np.float32)

    # Subtract black level, ensuring values don't go below 0
    adjusted_float = np.maximum(img_float - black_level, 0)

    # Calculate scale factor to stretch the remaining range [0, 255-black_level] back to [0, 255]
    scale_factor = 255.0 / (255.0 - black_level)
    adjusted_float *= scale_factor

    # Clip to ensure values are within [0, 255] and convert back to uint8
    adjusted_uint8 = np.clip(adjusted_float, 0, 255).astype(np.uint8)

    return adjusted_uint8

# ==============================================================================
# Upscaling Core Functions
# ==============================================================================

def load_engine(trt_path):
    """Loads an existing TensorRT engine."""
    if not os.path.exists(trt_path):
        raise FileNotFoundError(f"TensorRT engine file not found: {trt_path}")

    logging.info(f"Loading existing TensorRT engine from {os.path.basename(trt_path)}...")
    try:
        engine = Engine(trt_path)
        engine.load()
        engine.activate() # Activates context
        logging.info("Engine loaded and activated.")
        return engine
    except Exception as e:
        logging.error(f"Error loading or activating TensorRT engine from {trt_path}: {str(e)}")
        raise # Cannot proceed without engine

def load_preprocess_task(input_image_path, output_image_path, force_target_height=None, apply_black_level=None):
    """Loads, optionally resizes, applies black level (if specified), and preprocesses one image for upscaling."""
    try:
        t_start = time.time()
        image_vips = pyvips.Image.new_from_file(input_image_path, access="sequential")

        # --- Optional Resize ---
        original_height = image_vips.height
        if force_target_height is not None and force_target_height > 0:
            image_vips = resize_image_pyvips(image_vips, force_target_height)
            # Update effective original shape for post-processing calculations
            H_orig = image_vips.height # Should be force_target_height
            W_orig = image_vips.width
        else:
            H_orig = image_vips.height
            W_orig = image_vips.width

        # --- Basic Validation & Color Conversion ---
        if image_vips.bands not in [1, 3, 4]:
             logging.error(f"Skipping {os.path.basename(input_image_path)}: Unsupported number of bands ({image_vips.bands})")
             return None

        if image_vips.bands == 1:
            # Convert grayscale to 3-channel sRGB for consistency with models
            image_vips = image_vips.colourspace('srgb')
        elif image_vips.bands == 4:
            # Remove alpha channel
            image_vips = image_vips.extract_band(0, n=3)

        # Convert to NumPy array (HWC, should be 3 bands now)
        image_np = np.ndarray(
            buffer=image_vips.write_to_memory(),
            dtype=np.uint8,
            shape=[image_vips.height, image_vips.width, image_vips.bands]
        )
        # Update H_orig, W_orig again just in case colourspace changed dimensions slightly
        H_orig, W_orig = image_np.shape[0], image_np.shape[1]

        # Check dimensions against limits *before* padding
        if H_orig < MIN_DIM or W_orig < MIN_DIM or H_orig > MAX_DIM or W_orig > MAX_DIM:
            logging.error(f"Skipping {os.path.basename(input_image_path)} due to unsupported size after potential resize: {H_orig}x{W_orig} (Range: {MIN_DIM}-{MAX_DIM})")
            return None

        # --- Apply Black Level Adjustment ---
        if apply_black_level is not None and apply_black_level > 0:
            logging.info(f"Applying black level adjustment ({apply_black_level}) to {os.path.basename(input_image_path)}")
            image_np = _apply_black_level_adjustment(image_np, apply_black_level)
            # Note: image_np remains 3-channel HWC uint8 after adjustment

        # --- Padding --- (Ensure divisibility by 2, adjust if models need more)
        pad_h_to = 2
        pad_w_to = 2
        pad_h = (pad_h_to - H_orig % pad_h_to) % pad_h_to
        pad_w = (pad_w_to - W_orig % pad_w_to) % pad_w_to

        # Convert to NCHW float tensor, normalize
        # Use the potentially adjusted image_np
        image_tensor = torch.from_numpy(image_np).float().permute(2, 0, 1) / 255.0 # CHW, [0, 1]
        image_tensor = image_tensor.unsqueeze(0) # NCHW

        if pad_h > 0 or pad_w > 0:
            image_tensor = torch.nn.functional.pad(image_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        H_padded, W_padded = image_tensor.shape[2], image_tensor.shape[3]

        # Final check on padded dimensions against MAX_DIM for TRT profile
        if H_padded > MAX_DIM or W_padded > MAX_DIM:
             logging.error(f"Skipping {os.path.basename(input_image_path)} due to PADDED size too large: {H_padded}x{W_padded} (Max: {MAX_DIM})")
             return None

        t_end = time.time()
        return {
            "input_tensor": image_tensor, # Keep on CPU
            "original_shape": (H_orig, W_orig), # Shape *after* potential resize, before padding
            "padded_shape": (H_padded, W_padded),
            "input_path": input_image_path,
            "output_path": output_image_path,
            "load_preprocess_time": t_end - t_start
        }
    except pyvips.Error as e:
        logging.error(f"Error loading/preprocessing {os.path.basename(input_image_path)} with pyvips: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Generic error loading/preprocessing {os.path.basename(input_image_path)}: {str(e)}")
        return None


def dotgain20_resize(image: np.ndarray, new_size: tuple[int, int]) -> np.ndarray:
    """Resize a grayscale image with ICC transforms using pyvips to mimic dot_original."""
    # Get original and target dimensions
    h, w = image.shape
    target_h, target_w = new_size

    # If size matches, return original
    if h == target_h and w == target_w:
        return image

    try:
        # Convert numpy array to pyvips image
        vips_image = pyvips.Image.new_from_array(image)

        # Apply Gaussian blur if upscaling in height
        size_ratio = h / target_h
        if size_ratio < 1:  # Upscaling when target_h > h
            blur_size = (1 / size_ratio - 1) / 3.5
            if blur_size >= 0.1:
                blur_size = min(blur_size, 250)
                vips_image = vips_image.gaussblur(blur_size)

        # First ICC transform: dotgain20 to gamma1
        vips_image = vips_image.icc_transform(
            GAMMA1ICC_PATH,
            input_profile=DOTGAIN20ICC_PATH,
            intent='perceptual'
        )

        # Convert to float32 [0, 1] for resizing, mimicking dot_original
        vips_image = vips_image / 255.0

        # Resize with separate scales to match exact target size
        hscale = target_w / w
        vscale = target_h / h
        vips_image = vips_image.resize(hscale, vscale=vscale, kernel='cubic')

        # Convert back to uint8 [0, 255] before second ICC transform
        vips_image = (vips_image * 255).cast('uchar')

        # Second ICC transform: gamma1 to dotgain20
        vips_image = vips_image.icc_transform(
            DOTGAIN20ICC_PATH,
            input_profile=GAMMA1ICC_PATH,
            intent='perceptual'
        )

        # Convert to numpy array
        final_image = vips_image.numpy()

        # Ensure output is 2D grayscale uint8
        if final_image.dtype != np.uint8:
            final_image = final_image.astype(np.uint8)
        if final_image.ndim == 3 and final_image.shape[2] == 1:
            final_image = final_image.squeeze(axis=2)
        elif final_image.ndim != 2:
            logging.warning(f"Warning: Unexpected shape {final_image.shape} after dotgain20_resize. Attempting conversion to grayscale.")
            if final_image.ndim == 3 and final_image.shape[2] == 3:
                final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2GRAY)
            else:
                logging.error("Error: Could not ensure grayscale output in dotgain20_resize. Returning original.")
                return image # Fallback to original unresized gray image

        return final_image

    except pyvips.Error as e:
        logging.error(f"pyvips error during dotgain20_resize: {str(e)}")
        logging.error("Falling back to OpenCV resize.")
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    except Exception as e:
        logging.error(f"Unexpected error during dotgain20_resize: {str(e)}")
        logging.error("Falling back to OpenCV resize.")
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

# NEW: Standard resize function using pyvips with Lanczos3
def standard_resize_pyvips(image: np.ndarray, new_size: tuple[int, int]) -> np.ndarray:
    """Resize an image using pyvips with Lanczos3 filter, ensuring RGB output."""
    # Get original and target dimensions
    h, w = image.shape[:2]  # Handle both 2D and 3D input
    target_h, target_w = new_size

    # If size matches, convert to RGB if needed and return
    if h == target_h and w == target_w:
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            return np.stack([image.squeeze()] * 3, axis=2)  # Convert grayscale to RGB
        elif image.ndim == 3 and image.shape[2] == 3:
            return image
        else:
            raise ValueError(f"Unexpected input shape: {image.shape}")

    try:
        # Convert input to RGB if it's grayscale
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            image = np.stack([image.squeeze()] * 3, axis=2)

        # Convert numpy array to pyvips image
        vips_image = pyvips.Image.new_from_array(image)

        # Resize with separate scales to match exact target size using Lanczos3
        hscale = target_w / w
        vscale = target_h / h
        vips_image = vips_image.resize(hscale, vscale=vscale, kernel='lanczos3')

        # Convert back to numpy array
        final_image = vips_image.numpy()

        # Ensure output is 3D RGB uint8
        if final_image.dtype != np.uint8:
            final_image = np.clip(final_image, 0, 255).astype(np.uint8)
        
        if final_image.ndim != 3 or final_image.shape[2] != 3:
            if final_image.ndim == 2:
                final_image = np.stack([final_image] * 3, axis=2)
            elif final_image.shape[2] == 1:
                final_image = np.stack([final_image.squeeze()] * 3, axis=2)
            else:
                logging.warning(f"Warning: Unexpected shape {final_image.shape} after resize. Forcing RGB.")
                final_image = cv2.cvtColor(final_image, cv2.COLOR_GRAY2RGB)

        return final_image

    except pyvips.Error as e:
        logging.error(f"pyvips error during standard_resize_pyvips: {str(e)}")
        logging.error("Falling back to OpenCV resize (Lanczos4).")
        # Fallback to OpenCV, ensuring RGB output
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        if resized.ndim == 2:
            return cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        return resized
    except Exception as e:
        logging.error(f"Unexpected error during standard_resize_pyvips: {str(e)}")
        logging.error("Falling back to OpenCV resize (Lanczos4).")
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        if resized.ndim == 2:
            return cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        return resized


# MODIFIED: Added group_type parameter
def postprocess_save_task(output_tensor_cpu, original_shape, scale_factor, output_image_path, group_type):
    """
    Postprocesses the upscaled output tensor, resizes based on group_type,
    and saves the final image.
    """
    try:
        t_start = time.time()
        H_orig, W_orig = original_shape # Shape *before* padding, *after* potential initial resize
        H_out_scaled_padded = output_tensor_cpu.shape[2]
        W_out_scaled_padded = output_tensor_cpu.shape[3]

        H_expected = H_orig * scale_factor
        W_expected = W_orig * scale_factor

        # Crop padding
        if H_expected > H_out_scaled_padded or W_expected > W_out_scaled_padded:
            logging.warning(f"Warning: Cropping issue for {os.path.basename(output_image_path)}. Expected {H_expected}x{W_expected}, got tensor {H_out_scaled_padded}x{W_out_scaled_padded}. Clamping crop.")
            H_expected = min(H_expected, H_out_scaled_padded)
            W_expected = min(W_expected, W_out_scaled_padded)
        output_tensor_cpu = output_tensor_cpu[:, :, :H_expected, :W_expected]

        # Clamp, permute, scale, convert to numpy
        output_tensor_cpu = output_tensor_cpu.clamp_(0, 1)
        output_image_np_rgb = (output_tensor_cpu.squeeze(0).permute(1, 2, 0) * 255.0).round_().byte().cpu().numpy()

        # Determine target resize dimensions based on *original* aspect ratio (before any forcing)
        aspect_ratio = H_orig / W_orig if W_orig > 0 else 1.0

        if aspect_ratio > 1.0: # Tall image
            target_width = TARGET_WIDTH_TALL
            target_height = round(target_width * aspect_ratio)
        else: # Wide or square image
            target_width = TARGET_WIDTH_WIDE
            target_height = round(target_width * aspect_ratio)

        # --- Apply Resize based on Group Type ---
        current_h, current_w, _ = output_image_np_rgb.shape
        if current_h != target_height or current_w != target_width:
            logging.info(f"Resizing {os.path.basename(output_image_path)} from {current_w}x{current_h} to {target_width}x{target_height} using '{group_type}' method.")
            if group_type == 'manga_2x':
                # Use DotGain20 resize for manga 2x group
                output_image_gray = convert_image_to_grayscale(output_image_np_rgb)
                output_image_resized = dotgain20_resize(output_image_gray, (target_height, target_width))
                save_profile = DOTGAIN20ICC_PATH # Embed DotGain20 profile for manga
            else: # illust_2x or manga_4x (or any other future type)
                # Use Standard Lanczos resize for other groups
                output_image_resized = standard_resize_pyvips(output_image_np_rgb, (target_height, target_width))
                save_profile = None # Don't embed specific profile for standard resize
        else:
             output_image_resized = output_image_gray
             # Determine profile even if no resize occurred
             save_profile = DOTGAIN20ICC_PATH if group_type == 'manga_2x' else None
             logging.info(f"Skipping resize for {os.path.basename(output_image_path)}, dimensions already match target.")


        # --- Save the final image ---
        lossy_compression_quality = 92
        save_options = {"Q": int(lossy_compression_quality), "strip": True}
        if save_profile:
            save_options["profile"] = save_profile # Embed profile if specified

        pyvips_image_to_save = pyvips.Image.new_from_array(output_image_resized)
        pyvips_image_to_save.write_to_file(output_image_path, **save_options)

        t_end = time.time()
        return t_end - t_start

    except pyvips.Error as e:
        logging.error(f"Error during postprocessing/saving {os.path.basename(output_image_path)} with pyvips: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Generic error during postprocessing/saving {os.path.basename(output_image_path)}: {str(e)}")
        return None

def infer_tiled(engine, input_tensor_cpu, scale_factor, tile_size, overlap, max_tile_dim, device):
    """
    Performs inference by splitting the input into tiles, processing each,
    and stitching the results back together with overlap blending.

    Args:
        engine: The loaded TensorRT engine.
        input_tensor_cpu: The full preprocessed input tensor (NCHW) on CPU.
        scale_factor: The upscaling factor (e.g., 4).
        tile_size: The dimension of the square tiles for input.
        overlap: The overlap between adjacent tiles on the input.
        max_tile_dim: The maximum dimension the engine supports for a tile.
        device: The torch device (e.g., "cuda").

    Returns:
        torch.Tensor: The upscaled and stitched output tensor on CPU, or None if error.
    """
    if tile_size > max_tile_dim:
        logging.error(f"Error: TILE_SIZE ({tile_size}) cannot be greater than TILE_MODEL_MAX_DIM ({max_tile_dim}).")
        return None
    if overlap >= tile_size:
        logging.error(f"Error: TILE_OVERLAP ({overlap}) must be smaller than TILE_SIZE ({tile_size}).")
        return None

    n, c, h_in, w_in = input_tensor_cpu.shape
    h_out = h_in * scale_factor
    w_out = w_in * scale_factor
    tile_step = tile_size - overlap

    # Create buffers for the full output and a counter for blending
    output_tensor_full_cpu = torch.zeros((n, c, h_out, w_out), dtype=torch.float32, device='cpu')
    output_counts = torch.zeros((n, 1, h_out, w_out), dtype=torch.int16, device='cpu') # Use int16 for counts

    logging.info(f"Tiling input {w_in}x{h_in} into {tile_size}x{tile_size} tiles with {overlap} overlap...")
    num_tiles_h = math.ceil(h_in / tile_step)
    num_tiles_w = math.ceil(w_in / tile_step)
    total_tiles = num_tiles_h * num_tiles_w
    processed_tiles = 0

    try:
        for i in range(num_tiles_h): # Iterate over rows
            for j in range(num_tiles_w): # Iterate over columns
                # --- 1. Calculate Tile Coordinates (Input) ---
                y_start = i * tile_step
                x_start = j * tile_step
                y_end = min(y_start + tile_size, h_in)
                x_end = min(x_start + tile_size, w_in)
                y_start = max(0, y_end - tile_size) # Adjust start if tile goes past edge
                x_start = max(0, x_end - tile_size) # Adjust start if tile goes past edge

                tile_h_in = y_end - y_start
                tile_w_in = x_end - x_start

                # Skip tiny tiles that might occur at edges if logic is complex
                if tile_h_in < MIN_DIM or tile_w_in < MIN_DIM:
                     logging.info(f"Skipping tiny tile at ({i},{j}) with size {tile_w_in}x{tile_h_in}")
                     continue

                # --- 2. Extract Tile ---
                input_tile_cpu = input_tensor_cpu[:, :, y_start:y_end, x_start:x_end]

                # --- 3. Pad Tile if necessary (e.g., for divisibility by 2, though main padding should handle this) ---
                # Basic padding check (similar to main preprocessing)
                pad_h_to = 2
                pad_w_to = 2
                pad_h = (pad_h_to - tile_h_in % pad_h_to) % pad_h_to
                pad_w = (pad_w_to - tile_w_in % pad_w_to) % pad_w_to
                if pad_h > 0 or pad_w > 0:
                    input_tile_cpu = torch.nn.functional.pad(input_tile_cpu, (0, pad_w, 0, pad_h), mode='reflect')
                    tile_h_in_padded, tile_w_in_padded = input_tile_cpu.shape[2], input_tile_cpu.shape[3]
                else:
                    tile_h_in_padded, tile_w_in_padded = tile_h_in, tile_w_in

                # Final check on tile size against max dim
                if tile_h_in_padded > max_tile_dim or tile_w_in_padded > max_tile_dim:
                    logging.error(f"Error: Padded tile size {tile_w_in_padded}x{tile_h_in_padded} exceeds max dim {max_tile_dim}. Check TILE_SIZE/OVERLAP. Skipping tile.")
                    continue # Skip this tile

                # --- 4. Inference on Tile ---
                input_tile_gpu = input_tile_cpu.to(device)
                tile_input_shape = (n, c, tile_h_in_padded, tile_w_in_padded)
                tile_output_shape = (n, c, tile_h_in_padded * scale_factor, tile_w_in_padded * scale_factor)

                tile_shape_dict = {
                    "input": {"shape": tile_input_shape},
                    "output": {"shape": tile_output_shape},
                }

                output_tile_gpu = None # Ensure defined in outer scope
                try:
                    engine.allocate_buffers(shape_dict=tile_shape_dict)
                    stream_ptr = torch.cuda.current_stream(device).cuda_stream if device.type == 'cuda' else None
                    tile_infer_input = {"input": input_tile_gpu}
                    tile_result_dict = engine.infer(tile_infer_input, stream_ptr)
                    output_tile_gpu = tile_result_dict["output"]

                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    output_tile_cpu = output_tile_gpu.cpu()

                except Exception as e_tile:
                    logging.error(f"Error during inference for tile ({i},{j}) (Shape: {tile_input_shape}): {e_tile}. Skipping tile.")
                    # Clean up GPU memory for this failed tile
                    del input_tile_gpu
                    if output_tile_gpu is not None: del output_tile_gpu
                    if device.type == 'cuda': torch.cuda.empty_cache()
                    continue # Skip to next tile
                finally:
                    # Clean up GPU memory for this successful tile
                    del input_tile_gpu
                    if output_tile_gpu is not None: del output_tile_gpu
                    # Don't empty cache here, might be too slow

                # --- 5. Calculate Output Coordinates and Crop Padding ---
                y_start_out = y_start * scale_factor
                x_start_out = x_start * scale_factor
                # Expected size *without* tile padding
                tile_h_out_expected = tile_h_in * scale_factor
                tile_w_out_expected = tile_w_in * scale_factor
                # Crop the padding added to the tile before inference
                output_tile_cpu = output_tile_cpu[:, :, :tile_h_out_expected, :tile_w_out_expected]

                y_end_out = y_start_out + output_tile_cpu.shape[2]
                x_end_out = x_start_out + output_tile_cpu.shape[3]

                # --- 6. Add Tile to Full Output (Averaging Blend) ---
                output_tensor_full_cpu[:, :, y_start_out:y_end_out, x_start_out:x_end_out] += output_tile_cpu
                output_counts[:, :, y_start_out:y_end_out, x_start_out:x_end_out] += 1
                processed_tiles += 1
                logging.info(f"\rProcessed tile {processed_tiles}/{total_tiles}", end="")

        logging.info("Stitching complete. Finalizing output...")

        # --- 7. Final Averaging ---
        # Avoid division by zero where counts are 0 (shouldn't happen in theory)
        output_counts = torch.where(output_counts == 0, torch.ones_like(output_counts), output_counts)
        output_tensor_full_cpu /= output_counts.float() # Divide by counts to average

        # Clamp final result
        output_tensor_full_cpu.clamp_(0, 1)

        # Clean up count tensor
        del output_counts
        if device.type == 'cuda': torch.cuda.empty_cache() # Clean cache once after tiling

        return output_tensor_full_cpu

    except Exception as e_main:
        logging.error(f"\nFATAL Error during tiled inference: {e_main}")
        # Clean up potentially large tensors
        del output_tensor_full_cpu
        del output_counts
        if device.type == 'cuda': torch.cuda.empty_cache()
        return None

# ==============================================================================
# Group Processing Function (Handles one specific model)
# ==============================================================================

def process_image_group(image_paths, model_config, output_folder_path, device, force_target_height=None, apply_black_level=None):
    """
    Processes a list of images using a specific upscaling model TRT engine.
    Uses tiling for the 4x model if images are large enough.
    Optionally applies black level adjustment before upscaling.
    Uses group_type for post-processing decisions.
    """
    group_processed_count = 0
    group_skipped_count = 0
    group_timings = { "load_preprocess": [], "inference": [], "postprocess_save": [], "total_per_image": [] }

    if not image_paths:
        logging.info(f"No images to process for group: {model_config['name']}.")
        return group_processed_count, group_skipped_count, group_timings

    # Get group_type from model_config (should be added in process_upscale)
    group_type = model_config.get('group_type', 'unknown') # Default if missing
    scale_factor = model_config['scale'] # Get scale factor early

    logging.info(f"\n--- Processing Group: {model_config['name']} ({len(image_paths)} images) ---")
    logging.info(f"Group Type (for resize): {group_type}")
    logging.info(f"Using TRT: {os.path.basename(model_config['trt'])}")
    logging.info(f"Scale Factor: {scale_factor}")
    if force_target_height:
        logging.info(f"Forcing input height to: {force_target_height} (if applicable to this group)")
    if apply_black_level is not None and apply_black_level > 0:
        logging.info(f"Applying Black Level Adjustment: {apply_black_level} (for this group)")

    # Check if tiling should be considered for this group
    use_tiling_for_group = ENABLE_TILING and scale_factor == 4
    if use_tiling_for_group:
        logging.info(f"Tiling ENABLED for this group (Scale={scale_factor}). Max Tile Dim: {TILE_MODEL_MAX_DIM}, Tile Size: {TILE_SIZE}, Overlap: {TILE_OVERLAP}")
    else:
         logging.info(f"Tiling DISABLED for this group (Scale={scale_factor} or ENABLE_TILING=False).")


    # --- Load and Activate Engine for this group ---
    engine = None
    try:
        engine = load_engine(model_config['trt'])
    except Exception as e:
        logging.error(f"FATAL: Could not load engine for group {model_config['name']}. Skipping this group. Error: {str(e)}")
        group_skipped_count = len(image_paths)
        return group_processed_count, group_skipped_count, group_timings

    # --- Setup Thread Pools ---
    preload_executor = ThreadPoolExecutor(max_workers=NUM_LOAD_WORKERS, thread_name_prefix=f'LoadPre_{model_config["name"][:6]}')
    save_executor = ThreadPoolExecutor(max_workers=NUM_SAVE_WORKERS, thread_name_prefix=f'SavePost_{model_config["name"][:6]}')
    load_futures = []
    save_futures_data = [] # Store tuple: (future, load_time, inference_time)

    try:
        # --- Submit Load Tasks ---
        for input_path in image_paths:
            filename = os.path.basename(input_path)
            output_filename = os.path.splitext(filename)[0] + ".jpg" # Save as jpg
            output_path = os.path.join(output_folder_path, output_filename)

            # Pass force_target_height and apply_black_level to the load task
            future = preload_executor.submit(
                load_preprocess_task,
                input_path,
                output_path,
                force_target_height,
                apply_black_level # Pass the black level value for this group
            )
            load_futures.append(future)

        # --- Process Loaded Images (Inference + Submit Save) ---
        logging.info(f"Submitted {len(load_futures)} images for loading/preprocessing...")

        # ================================================================
        # MODIFIED SECTION: Handle Tiling vs Non-Tiling Inference
        # ================================================================
        for future in tqdm(as_completed(load_futures), total=len(load_futures), desc=f"Inferring ({model_config['name']})"):
            preprocess_result = future.result()

            if preprocess_result is None:
                group_skipped_count += 1
                continue # Skip image if loading/preprocessing failed

            t_infer_start = time.time()
            input_tensor_cpu = preprocess_result["input_tensor"]
            padded_shape = preprocess_result["padded_shape"]
            H_pad, W_pad = padded_shape
            output_tensor_cpu = None # Initialize output tensor variable
            inference_failed = False

            # --- Determine if tiling is needed FOR THIS IMAGE ---
            needs_tiling = (
                use_tiling_for_group and
                (H_pad > TILE_MODEL_MAX_DIM or W_pad > TILE_MODEL_MAX_DIM)
            )

            if needs_tiling:
                # --- Tiled Inference Path ---
                logging.info(f"\nImage {os.path.basename(preprocess_result['input_path'])} ({W_pad}x{H_pad}) requires tiling.")
                try:
                    output_tensor_cpu = infer_tiled(
                        engine, input_tensor_cpu, scale_factor,
                        TILE_SIZE, TILE_OVERLAP, TILE_MODEL_MAX_DIM, device
                    )
                    if output_tensor_cpu is None:
                        logging.info(f"\nTiled inference failed for {os.path.basename(preprocess_result['input_path'])}. Skipping image.")
                        inference_failed = True
                except Exception as e_tile_main:
                    logging.error(f"\nError during tiled inference call for {os.path.basename(preprocess_result['input_path'])}: {e_tile_main}. Skipping image.")
                    inference_failed = True
                finally:
                    pass # engine.reset() might be too drastic here.

            else:

                input_shape_nchw = (1, 3, H_pad, W_pad)
                output_shape_nchw = (1, 3, H_pad * scale_factor, W_pad * scale_factor)
                shape_dict = {
                    "input": {"shape": input_shape_nchw},
                    "output": {"shape": output_shape_nchw},
                }
                input_tensor_gpu = None # Ensure defined
                output_tensor_gpu = None # Ensure defined

                try:
                    input_tensor_gpu = input_tensor_cpu.to(device)
                    engine.allocate_buffers(shape_dict=shape_dict) # Allocate for the whole image
                    stream_ptr = torch.cuda.current_stream(device).cuda_stream if device.type == 'cuda' else None
                    infer_input_dict = {"input": input_tensor_gpu}
                    result_dict = engine.infer(infer_input_dict, stream_ptr)
                    output_tensor_gpu = result_dict["output"] # Match output name

                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    output_tensor_cpu = output_tensor_gpu.cpu() # Get result to CPU

                except Exception as e_std:
                    logging.error(f"\nError during standard inference for {os.path.basename(preprocess_result['input_path'])} (Shape: {padded_shape}): {e_std}. Skipping image.")
                    inference_failed = True
                finally:
                    # Clean up GPU memory for standard inference
                    if input_tensor_gpu is not None: del input_tensor_gpu
                    if output_tensor_gpu is not None: del output_tensor_gpu
                    # Don't empty cache here in the loop

            t_infer_end = time.time()
            inference_time = t_infer_end - t_infer_start
            # --- End Inference Stage ---

            # --- Submit Postprocess/Save Task (if inference succeeded) ---
            if not inference_failed and output_tensor_cpu is not None:
                save_future = save_executor.submit(
                    postprocess_save_task,
                    output_tensor_cpu, # Pass the potentially tiled result
                    preprocess_result["original_shape"], # Pass shape after potential resize
                    scale_factor,
                    preprocess_result["output_path"],
                    group_type # Pass the group identifier
                )
                save_futures_data.append((save_future, preprocess_result["load_preprocess_time"], inference_time))
            else:
                group_skipped_count += 1 # Increment skip count if inference failed
                if output_tensor_cpu is not None: del output_tensor_cpu

            # --- Clean up input tensor ---
            del input_tensor_cpu
            if device.type == 'cuda':
                 pass

        # ================================================================
        # End of Modified Section
        # ================================================================


        # --- Collect Save Results ---
        logging.info(f"\nInference complete for group {model_config['name']}. Waiting for save tasks...")
        for future, load_time, infer_time in tqdm(save_futures_data, desc=f"Saving ({model_config['name']})"):
             try:
                save_time = future.result(timeout=300) # Increased timeout slightly
                if save_time is not None:
                    group_timings["load_preprocess"].append(load_time)
                    group_timings["inference"].append(infer_time)
                    group_timings["postprocess_save"].append(save_time)
                    group_timings["total_per_image"].append(load_time + infer_time + save_time)
                    group_processed_count += 1
                else:
                     group_skipped_count += 1
             except TimeoutError:
                 logging.error("Error: A save task timed out.")
                 group_skipped_count += 1
             except Exception as e:
                logging.error(f"Error collecting save result: {str(e)}")
                group_skipped_count += 1

        logging.info(f"Finished processing group: {model_config['name']}.")

    finally:
        # --- Cleanup ---
        logging.info(f"Shutting down executors for group: {model_config['name']}...")
        preload_executor.shutdown(wait=False) # Don't wait for loading if exiting
        save_executor.shutdown(wait=True) # Wait for saving to finish
        logging.info(f"Resetting engine context for group: {model_config['name']}...")
        if engine:
            # Instead of del engine, just ensure context is released if needed
            # The engine object itself might be reused if script structure changes
            if engine.context:
                 del engine.context # Explicitly delete context
                 engine.context = None
            # del engine # Keep engine object if needed later, just release context/buffers
        if device.type == 'cuda':
            torch.cuda.empty_cache() # Final cleanup for the group

    return group_processed_count, group_skipped_count, group_timings


# ==============================================================================
# Main Processing Function
# ==============================================================================

def process_upscale(input_folder_path, output_folder_path, force_image_height=None):
    logfilename = os.path.basename(input_folder_path) + ".log"
    logging.basicConfig(level=logging.INFO, filename=logfilename)
    """
    Classifies images, calculates black level for manga, selects appropriate
    upscaling models (dynamic for manga), applies black level adjustment to manga,
    processes them using TensorRT, and applies group-specific resizing.

    Args:
        input_folder_path (str): Path to the folder containing input images.
        output_folder_path (str): Path to the folder where upscaled images will be saved.
        force_image_height (int, optional): If provided, manga images will be resized
                                             to this height *before* selecting the
                                             nearest manga model and upscaling. Defaults to None.
    """
    logging.info(f"Starting upscale process.")
    logging.info(f"Input Folder: {input_folder_path}")
    logging.info(f"Output Folder: {output_folder_path}")
    logging.info(f"Using Device: {DEVICE}")
    if force_image_height:
        logging.info(f"Forcing Manga Input Height (before model selection): {force_image_height}")

    os.makedirs(output_folder_path, exist_ok=True)

    # --- Timing and Counters ---
    overall_start_time = time.time()
    overall_timings = {
        "classification": 0.0,
        "black_level_calc": 0.0,
        "load_preprocess": [],
        "inference": [],
        "postprocess_save": [],
        "total_per_image": [],
    }
    total_processed_count = 0
    total_skipped_count = 0
    total_classified_count = 0
    total_images_found_initial = 0

    # --- 1. Classification Phase ---
    classification_model = None
    classification_transform = None
    classification_results = {}
    try:
        classification_model = load_classification_model(
            CLASSIFICATION_MODEL_PATH, CLASSIFICATION_MODEL_NAME, CLASSIFICATION_NUM_CLASSES, DEVICE
        )
        classification_transform = get_classification_transform(classification_model, CLASSIFICATION_INPUT_SIZE)
        classification_results, cls_time, cls_count, skip_cls = classify_images_in_folder(
            input_folder_path, classification_model, classification_transform, DEVICE
        )
        overall_timings["classification"] = cls_time
        total_classified_count = cls_count
        total_skipped_count += skip_cls
        total_images_found_initial = cls_count + skip_cls

    except Exception as e:
        logging.error(f"FATAL: Could not perform classification phase. Error: {str(e)}")
        return
    finally:
        del classification_model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    if not classification_results:
        logging.error("No images were successfully classified. Exiting.")
        return

    # --- 2. Group Images Based on Classification ---
    logging.info("\n--- Grouping Images Based on Classification ---")
    illust_group = []
    uncertain_group = []
    manga_candidates = []
    skipped_grouping = 0

    for filename, (prob0, prob1) in classification_results.items():
        input_path = os.path.join(input_folder_path, filename)
        if not os.path.exists(input_path):
             logging.warning(f"Warning: Classified image file not found during grouping: {filename}. Skipping.")
             skipped_grouping += 1
             continue

        if prob0 > CLASSIFICATION_THRESHOLD_HIGH:           # Class 0 -> Manga (candidate)
            manga_candidates.append(input_path)
        elif prob0 < CLASSIFICATION_THRESHOLD_LOW:          # Class 1 -> Illust 2x
            illust_group.append(input_path)
        elif CLASSIFICATION_THRESHOLD_LOW <= prob0 <= CLASSIFICATION_THRESHOLD_HIGH: # Uncertain -> Manga 4x
            uncertain_group.append(input_path)
        else:
            logging.info(f"Skipping {filename}: Classification probability {prob0:.3f} falls outside defined thresholds.")
            skipped_grouping += 1

    total_skipped_count += skipped_grouping

    # --- 2b. Calculate Manga Black Level ---
    manga_median_black_level = None
    if manga_candidates:
        logging.info("\n--- Calculating Black Level for Manga Group ---")
        t_bl_start = time.time()
        manga_median_black_level = calculate_manga_black_level(manga_candidates)
        t_bl_end = time.time()
        overall_timings["black_level_calc"] = t_bl_end - t_bl_start
        if manga_median_black_level is None:
            logging.warning("Warning: Failed to calculate median black level for manga. Adjustment will be skipped.")
            manga_median_black_level = 0
        elif manga_median_black_level <= 0:
             logging.info("Median black level is 0 or less. No adjustment needed.")
             manga_median_black_level = 0
        else:
            logging.info(f"Black level calculation time: {overall_timings['black_level_calc']:.2f} seconds")
    else:
        logging.info("\nNo manga candidates found, skipping black level calculation.")


    # --- 2c. Sub-group Manga Images Based on Height ---
    logging.info("\n--- Selecting Manga Models Based on Height ---")
    manga_sub_groups = {} # Key: trt_path, Value: dict {'paths': [], 'model_config': {}}
    skipped_manga_height_selection = 0

    if manga_candidates:
        for img_path in tqdm(manga_candidates, desc="Selecting manga models"):
            height = get_image_height(img_path)
            if height is None:
                logging.info(f"Skipping {os.path.basename(img_path)}: Could not determine height.")
                skipped_manga_height_selection += 1
                continue

            selection_height = force_image_height if force_image_height else height

            try:
                # find_nearest_manga_model now returns the config with 'group_type' added
                chosen_model_config = find_nearest_manga_model(selection_height, MANGA_2X_MODELS)
                trt_path = chosen_model_config['trt']

                if trt_path not in manga_sub_groups:
                    manga_sub_groups[trt_path] = {'paths': [], 'model_config': chosen_model_config}
                manga_sub_groups[trt_path]['paths'].append(img_path)

            except ValueError as e:
                 logging.error(f"Error selecting manga model for {os.path.basename(img_path)}: {str(e)}. Skipping.")
                 skipped_manga_height_selection += 1
            except Exception as e:
                 logging.error(f"Unexpected error selecting manga model for {os.path.basename(img_path)}: {str(e)}. Skipping.")
                 skipped_manga_height_selection += 1

    total_skipped_count += skipped_manga_height_selection

    logging.info("Image grouping complete:")
    logging.info(f"  Group '{OTHER_MODELS['illust_2x']['name']}': {len(illust_group)} images (Resize: standard)")
    logging.info(f"  Group '{OTHER_MODELS['manga_4x']['name']}': {len(uncertain_group)} images (Resize: standard)")
    for trt_path, group_data in manga_sub_groups.items():
         model_name = group_data['model_config'].get('name', "Unknown Manga Model")
         logging.info(f"  Group '{model_name}': {len(group_data['paths'])} images (Resize: dotgain20)")
    if skipped_grouping > 0 or skipped_manga_height_selection > 0:
        logging.info(f"  Skipped during grouping/selection (thresholds, height error, etc.): {skipped_grouping + skipped_manga_height_selection}")


    # --- 3. Process Each Group ---
    # Process Illustrators (Standard Resize, No black level adjustment)
    if illust_group:
        processed, skipped, timings = process_image_group(
            illust_group, OTHER_MODELS['illust_2x'], output_folder_path, DEVICE,
            force_target_height=force_image_height, apply_black_level=None # group_type is in model_config
        )
        total_processed_count += processed
        total_skipped_count += skipped
        for key in timings: overall_timings[key].extend(timings[key])

    # Process Uncertain (Manga 4x) (Standard Resize, No black level adjustment)
    if uncertain_group:
        processed, skipped, timings = process_image_group(
            uncertain_group, OTHER_MODELS['manga_4x'], output_folder_path, DEVICE,
            force_target_height=force_image_height, apply_black_level=None # group_type is in model_config
        )
        total_processed_count += processed
        total_skipped_count += skipped
        for key in timings: overall_timings[key].extend(timings[key])

    # Process Manga Sub-Groups (DotGain20 Resize, Apply calculated black level)
    for trt_path, group_data in manga_sub_groups.items():
        image_list = group_data['paths']
        model_config = group_data['model_config'] # Contains 'group_type': 'manga_2x'
        if model_config:
            processed, skipped, timings = process_image_group(
                image_list, model_config, output_folder_path, DEVICE,
                force_target_height=force_image_height,
                apply_black_level=manga_median_black_level # Pass the calculated value
            )
            total_processed_count += processed
            total_skipped_count += skipped
            for key in timings: overall_timings[key].extend(timings[key])
        else:
            # This case should ideally not happen due to how manga_sub_groups is built
            logging.error(f"Error: Could not find model config for TRT path {trt_path}. Skipping {len(image_list)} images.")
            total_skipped_count += len(image_list)


    # --- 4. Final Performance Summary ---
    overall_end_time = time.time()
    total_time_elapsed = overall_end_time - overall_start_time

    logging.info("\n\n--- Overall Performance Summary ---")
    logging.info(f"Total images found in input folder: {total_images_found_initial}")
    logging.info(f"Successfully classified: {total_classified_count}")
    logging.info(f"Successfully processed (upscaled & saved): {total_processed_count}")
    logging.info(f"Total skipped or errored (classification + grouping + processing): {total_skipped_count}")
    logging.info(f"Sanity Check: Processed ({total_processed_count}) + Skipped ({total_skipped_count}) = {total_processed_count + total_skipped_count} vs Found ({total_images_found_initial})")


    logging.info(f"\nTotal execution time: {total_time_elapsed:.2f} seconds")
    logging.info(f"  - Classification time: {overall_timings['classification']:.2f} seconds")
    if overall_timings['black_level_calc'] > 0:
        logging.info(f"  - Manga Black Level Calc time: {overall_timings['black_level_calc']:.2f} seconds")

    if total_processed_count > 0:
        avg_load = np.mean(overall_timings['load_preprocess']) if overall_timings['load_preprocess'] else 0
        avg_infer = np.mean(overall_timings['inference']) if overall_timings['inference'] else 0
        avg_save = np.mean(overall_timings['postprocess_save']) if overall_timings['postprocess_save'] else 0
        avg_total_per_image = np.mean(overall_timings['total_per_image']) if overall_timings['total_per_image'] else 0

        # Calculate upscaling time excluding classification and black level calc
        upscaling_time = total_time_elapsed - overall_timings['classification'] - overall_timings['black_level_calc']
        img_per_sec = total_processed_count / upscaling_time if upscaling_time > 0 else 0

        logging.info(f"\nAverage time per SUCCESSFULLY processed image (Upscaling Stages):")
        logging.info(f"  Load & Preprocess: {avg_load:.4f} seconds")
        logging.info(f"  Inference:         {avg_infer:.4f} seconds")
        logging.info(f"  Postprocess & Save:{avg_save:.4f} seconds")
        logging.info(f"  -----------------------------")
        logging.info(f"  Total (per image): {avg_total_per_image:.4f} seconds")
        logging.info(f"\nOverall upscaling throughput: {img_per_sec:.2f} images/second (excluding classification & black level calc time)")
    else:
        logging.info("\nNo images were successfully processed (upscaled).")

    logging.info("\nProcessing finished.")
    return total_time_elapsed, total_processed_count, total_skipped_count, skip_cls


#if __name__ == "__main__":
#    # Configure paths here for direct execution
#    INPUT_DIR = r"C:\Paprika2\Projet 2 - Copie (3)\Auto rename\The Shiunji Family Children T01 (Miyajima) (2025) [Digital-1200] [Manga FR] (Paprika+)"
#    OUTPUT_DIR = r"C:\Paprika2\Projet 2 - Copie (3)\Auto rename\sortietrt_classified_dynamic_blacklevel_groupresize" # Changed output dir name
#
#    # --- Optional: Set a height to force manga images to before upscaling ---
#    # FORCE_HEIGHT = None
#    FORCE_HEIGHT = 1200 # Example: Force to 1200p input
#
#    # --- Run the processing ---
#    try:
#        total_time_elapsed, total_processed_count, total_skipped_count, skip_cls =process_upscale(INPUT_DIR, OUTPUT_DIR, force_image_height=FORCE_HEIGHT)
#        logging.info(total_time_elapsed, total_processed_count, total_skipped_count, skip_cls)
#    except ImportError as e:
#         logging.error(f"Import Error: {str(e)}. Please ensure all dependencies including 'trt_utilities' and 'opencv-python' are installed and accessible.")
#    except FileNotFoundError as e:
#         logging.error(f"File Not Found Error: {str(e)}. Check model paths and input/output directories.")
#    except Exception as e:
#         logging.error(f"An unexpected error occurred: {str(e)}")
#         # Optionally add traceback printing for debugging
#         import traceback
#         traceback.print_exc()
