import argparse
import os
import random
import concurrent.futures
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image

def get_image_paths(directory, limit=None):
    """Recursively find all images in the directory."""
    supported_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    paths = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_extensions:
                paths.append(os.path.join(root, file))
                
    # Sort for consistent ordering, especially useful when using --limit (-n)
    paths.sort()
    
    if limit is not None and limit > 0:
        return paths[:limit]
    return paths

def apply_social_media_compression(img_path, quality_range=(50, 70)):
    """
    Simulates WhatsApp/Instagram compression by re-encoding to JPEG at a lower quality,
    introducing block artifacts and color subsampling.
    """
    try:
        # Read the image
        img = Image.open(img_path).convert('RGB')
        
        # Save to an in-memory buffer with random quality in the specified range
        buffer = BytesIO()
        quality = random.randint(quality_range[0], quality_range[1])
        
        # JPEG compression is typically what causes these specific artifacts
        img.save(buffer, format="JPEG", quality=quality)
        
        # Re-open from the buffer
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        
        # Save back to the original format (keeping the original extension as requested)
        # Note: We are overwriting the original file to apply the distortion in-place.
        # If the original was PNG, it will now contain JPEG-like artifacts saved as PNG.
        ext = os.path.splitext(img_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            # We already have the JPEG byte stream, just write it
            with open(img_path, 'wb') as f:
                f.write(buffer.getvalue())
        else:
             # Need to save the degraded Image object back to its respective format
             compressed_img.save(img_path)
             
        return True
    except Exception as e:
        print(f"Error compressing {img_path}: {e}")
        return False

def generate_moire_pattern(shape, frequency=None, angle=None):
    """
    Generates a synthetic, subtle moire pattern.
    """
    height, width = shape[:2]
    
    # Randomize parameters to prevent the model from learning a single static overlay
    if frequency is None:
        frequency = random.uniform(20.0, 50.0) # Higher frequency -> finer lines
    if angle is None:
        angle = random.uniform(0, np.pi)
        
    y, x = np.mgrid[0:height, 0:width]
    
    # Generate wave pattern
    wave = np.sin((x * np.cos(angle) + y * np.sin(angle)) * frequency)
    
    # Normalize to 0-1
    pattern = (wave + 1) / 2.0
    
    # Make it a 3-channel image if necessary
    if len(shape) == 3 and shape[2] == 3:
         pattern = np.stack([pattern]*3, axis=-1)
         
    return pattern

def apply_moire_effect(img_path, alpha_range=(0.05, 0.15)):
    """
    Applies a subtle Moire effect to the image to simulate a digital screen capture.
    """
    try:
        # Load image via Pillow instead of OpenCV
        img_pil = Image.open(img_path).convert('RGB')
        img_np = np.array(img_pil)
            
        # Convert to float for blending
        img_float = img_np.astype(np.float32) / 255.0
        
        # Generate pattern
        pattern = generate_moire_pattern(img_np.shape)
        
        # Randomize blending strength so it remains subtle
        alpha = random.uniform(alpha_range[0], alpha_range[1])
        
        # Use an overlay or soft-light style blend. 
        # A simple linear blend (alpha blending) might dull the image too much.
        # Here we use a 'multiply' blend modified by alpha to add dark fringes, 
        # typical of moire interference.
        blended = img_float * (1.0 - alpha) + (img_float * pattern) * alpha
        
        # Ensure values are within valid range and convert back
        blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
        
        # Overwrite the original
        blended_pil = Image.fromarray(blended)
        blended_pil.save(img_path)
        return True
    except Exception as e:
        print(f"Error applying moire to {img_path}: {e}")
        return False

def process_image(args):
    """Worker function for threading."""
    img_path, distortion_type = args
    
    if distortion_type == 1: # Compression Only
        return apply_social_media_compression(img_path)
    elif distortion_type == 2: # Moire Only
        return apply_moire_effect(img_path)
    elif distortion_type == 3: # Moire + Compression
        # Apply moire first (simulating capture), then compression (simulating transmission)
        success_moire = apply_moire_effect(img_path)
        if success_moire:
            return apply_social_media_compression(img_path)
        return False
    else:
        return True # Type 4 - No effect maybe? Or user didn't specify.

def main():
    parser = argparse.ArgumentParser(description="Apply distortions to deepfake dataset images.")
    parser.add_argument("directory", type=str, help="Path to the dataset directory (e.g., 'dataset/Data Set 1')")
    parser.add_argument("type", type=int, choices=[1, 2, 3, 4], 
                        help="Distortion type: 1=Compression, 2=Moire, 3=Both, 4=None(Pass)")
    parser.add_argument("-n", "--limit", type=int, default=None, 
                        help="Limit the number of images to process (useful for testing)")
    parser.add_argument("-t", "--threads", type=int, default=os.cpu_count() or 4,
                        help="Number of threads to use for parallel processing")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="Random seed for reproducibility")
                        
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    if args.type == 4:
         print("Type 4 selected: No distortions will be applied.")
         return

    print(f"Scanning target directory: {args.directory}")
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' not found.")
        return
        
    image_paths = get_image_paths(args.directory, args.limit)
    total_images = len(image_paths)
    
    if total_images == 0:
        print("No images found in the specified directory.")
        return
        
    print(f"Found {total_images} images to process.")
    print(f"Applying distortion Type {args.type} using {args.threads} threads...")
    
    # Prepare arguments for the thread pool
    worker_args = [(path, args.type) for path in image_paths]
    
    success_count = 0
    # Use ThreadPoolExecutor since file I/O and PIL operations often release the GIL
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        # submit all tasks and track completion
        futures = {executor.submit(process_image, arg): arg for arg in worker_args}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            if result:
                success_count += 1
                
            # Print progress
            if (i + 1) % max(1, (total_images // 20)) == 0 or (i + 1) == total_images:
                print(f"Progress: {i + 1}/{total_images} ({(i + 1)/total_images * 100:.1f}%)")
                
    print(f"Done. Successfully processed {success_count} out of {total_images} images.")

if __name__ == "__main__":
    main()
