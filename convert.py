# Usage:
# docker run --rm -it -v D:\projects\neuralangelo\datasets:/data neuralangelo bash
# Then inside the container:
# cd /data/torso
# python convert.py

import json
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_POSES = "/data/torso/poses.json"
INPUT_INTRINSICS = "/data/torso/intrinsics_rgb.json"
OUTPUT_JSON = "/data/torso/transforms.json"

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    poses_data = load_json(INPUT_POSES)
    intrinsics = load_json(INPUT_INTRINSICS)
    
    # 1. Extract Camera Parameters
    # Neuralangelo expects fl_x, fl_y, cx, cy, w, h
    output_data = {
        "fl_x": intrinsics["fx"],
        "fl_y": intrinsics["fy"],
        "cx": intrinsics["cx"],
        "cy": intrinsics["cy"],
        "w": intrinsics["width"],
        "h": intrinsics["height"],
        "sk_x": 0.0,
        "sk_y": 0.0,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "is_fisheye": False,
        "sphere_center": [0.0, 0.0, 0.0],
        "sphere_radius": 1.0,
        "aabb_scale": 1,  # Tells Neuralangelo the scene scale is 1 (unit sphere)
        "frames": []
    }

    # 2. Collect all camera positions to calculate Scene Center & Scale
    all_c2w = []
    frames_buffer = []

    print(f"Processing {len(poses_data['frames'])} frames...")

    for frame in poses_data["frames"]:
        # Extract 4x4 matrix
        c2w = np.array(frame["world_from_camera"])
        
        # Store for normalization calculation
        all_c2w.append(c2w)
        
        # Keep track of file paths (ensure extension matches your actual files)
        # Your JSON says "images/frame_00000.png", ensure this relative path is correct
        frames_buffer.append({
            "file_path": frame["rgb_path"], 
            "original_matrix": c2w
        })

    # 3. NORMALIZE THE SCENE (Crucial for Neuralangelo)
    # We need the object of interest to be at (0,0,0) and cameras within a reasonable radius.
    
    all_c2w = np.array(all_c2w)
    cam_centers = all_c2w[:, :3, 3] # Extract translation (x,y,z) columns
    
    # Calculate center of all cameras (assuming object is roughly in the middle of the camera ring)
    # Ideally, if you know the center of the Torso, hardcode it here.
    # For a surround view, the average of camera positions is usually the object center.
    center = np.mean(cam_centers, axis=0)
    
    # Calculate distance from center to furthest camera
    max_dist = np.max(np.linalg.norm(cam_centers - center, axis=1))
    
    # Scale factor: We want cameras to be roughly at distance 2.0 to 3.0 from origin 
    # so the object (radius ~0.5-1.0) fits in the unit sphere.
    scale_factor = 2.0 / max_dist 

    print(f"Auto-centering at {center}")
    print(f"Auto-scaling by {scale_factor} (Max camera dist was {max_dist:.2f})")

    # 4. Apply Transformation and Save
    for i, frame_buf in enumerate(frames_buffer):
        c2w = frame_buf["original_matrix"]
        
        # Apply normalization:
        # 1. Translate cameras so scene center is 0,0,0
        c2w[:3, 3] -= center
        # 2. Scale the world
        c2w[:3, 3] *= scale_factor
        
        # Neuralangelo expects flat lists for matrices
        output_data["frames"].append({
            "file_path": frame_buf["file_path"],
            "transform_matrix": c2w.tolist()
        })

    with open(OUTPUT_JSON, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)
    
    print(f"Success! Saved {OUTPUT_JSON}")
    print("NOTE: Ensure your 'images' folder is in the same directory as this json.")

if __name__ == "__main__":
    main()