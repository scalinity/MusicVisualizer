
import numpy as np
import math

def test_terrain_math():
    print("Testing Terrain Math...")
    
    # Mock Constants
    ROWS = 180
    COLS = 64
    WIDTH_STEP = 1.0
    DEPTH_STEP = 0.4
    HEIGHT_SCALE = 1.5
    
    # Mock History (mimic init)
    history = [np.zeros(COLS) for _ in range(ROWS)]
    
    # Fill with some random data (simulating audio)
    # Audio data is usually non-negative in Visualizer (abs(fft))
    # But let's test edge cases: 0, small positive, large positive.
    for i in range(ROWS):
        history[i] = np.random.random(COLS) * 100.0
        # Add some zeros
        history[i][0] = 0.0
        
    print(f"History initialized. Shape: ({ROWS}, {COLS})")
    
    # 1. Calculate heights
    history_array = np.array(history)
    print(f"History Array Shape: {history_array.shape}")
    
    # Check for NaNs/Infs in input
    if not np.all(np.isfinite(history_array)):
        print("FAIL: Input history contains non-finite values!")
        return
        
    heights = (history_array ** 0.7) * HEIGHT_SCALE
    print("Heights calculated.")
    if not np.all(np.isfinite(heights)):
        print("FAIL: Heights contain non-finite values!")
        print(heights[~np.isfinite(heights)])
        return
        
    # 2. Gradient
    dz_arr, dx_arr = np.gradient(heights)
    print("Gradients calculated.")
    
    # 3. Normals
    normals = np.stack([-dx_arr, np.ones_like(heights), -dz_arr], axis=-1)
    norm_norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / (norm_norms + 1e-6)
    
    if not np.all(np.isfinite(normals)):
        print("FAIL: Normals contain non-finite values!")
        return
    print("Normals calculated and clean.")
    
    # 4. Strip generation logic
    num_strips = ROWS - 1
    
    # Check shape logic for VBO
    # v_top stack
    x_coords = np.arange(COLS) * WIDTH_STEP
    z_coords = np.arange(ROWS) * DEPTH_STEP
    
    top_slice = slice(0, ROWS-1)
    bot_slice = slice(1, ROWS)
    
    h_top = heights[top_slice]
    c_top = np.zeros((*h_top.shape, 3)) # Mock colors
    n_top = normals[top_slice]
    
    x_top = np.tile(x_coords, (num_strips, 1))
    z_top = np.tile(z_coords[top_slice][:, np.newaxis], (1, COLS))
    
    v_top = np.stack([x_top, h_top, z_top, 
                      c_top[...,0], c_top[...,1], c_top[...,2],
                      n_top[...,0], n_top[...,1], n_top[...,2]], axis=-1)
                      
    print(f"v_top shape: {v_top.shape} (Expected: {num_strips}, {COLS}, 9)")
    
    # Final buffer size check
    # Visualizer says: max_vertices = (rows - 1) * cols * 2 * 9 (floats)
    total_floats = (ROWS - 1) * COLS * 2 * 9
    print(f"Expected float count: {total_floats}")
    
    combined = np.zeros((num_strips, COLS, 2, 9), dtype=np.float32)
    # If this assignment works, shapes match
    try:
        combined[:, :, 0, :] = v_top
        # Mock bot
        combined[:, :, 1, :] = v_top 
    except ValueError as e:
        print(f"FAIL: Shape mismatch during assignment: {e}")
        return

    final_mesh = combined.reshape(num_strips, COLS * 2, 9)
    vertex_data = final_mesh.flatten().astype(np.float32)
    
    print(f"Final data size: {vertex_data.size}")
    
    if vertex_data.size != total_floats:
        print(f"FAIL: Size mismatch! Got {vertex_data.size}, expected {total_floats}")
    else:
        print("SUCCESS: Math logic seems valid and consistent.")

if __name__ == "__main__":
    test_terrain_math()
