import pygame
from pygame.locals import *
from OpenGL.GL import *
# from OpenGL.GLU import * # GLU is missing on some systems
import numpy as np
import math
import ctypes
import colorsys
from src.audio_analyzer import AudioAnalyzer

def gluPerspective(fovy, aspect, zNear, zFar):
    ymax = zNear * math.tan(fovy * math.pi / 360.0)
    xmax = ymax * aspect
    glFrustum(-xmax, xmax, -ymax, ymax, zNear, zFar)

def gluOrtho2D(left, right, bottom, top):
    glOrtho(left, right, bottom, top, -1, 1)

def gluLookAt(eyex, eyey, eyez, centerx, centery, centerz, upx, upy, upz):
    forward = np.array([centerx - eyex, centery - eyey, centerz - eyez])
    forward = forward / np.linalg.norm(forward)
    
    up = np.array([upx, upy, upz])
    
    side = np.cross(forward, up)
    side = side / np.linalg.norm(side)
    
    new_up = np.cross(side, forward)
    # new_up is already normalized
    
    # Create the view matrix
    # OpenGL is column-major, but we can construct row-major and transpose or load appropriately
    # We'll use glMultMatrixf with the transpose of the rotation part
    
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = side
    m[1, :3] = new_up
    m[2, :3] = -forward
    
    glMultMatrixf(m.T)
    glTranslatef(-eyex, -eyey, -eyez)


class Visualizer:
    # Display constants
    DEFAULT_WIDTH = 1280
    DEFAULT_HEIGHT = 720
    TARGET_FPS = 60

    # Visualization modes
    MODE_BARS = 0
    MODE_TERRAIN = 1
    MODE_CIRCULAR = 2
    MODE_STEREO = 3

    # Camera constants
    DEFAULT_CAMERA_DISTANCE = 30
    CAMERA_ZOOM_STEP = 2.0
    CAMERA_MIN_DISTANCE = 10
    CAMERA_MAX_DISTANCE = 200
    ROTATION_SPEED = 0.3
    MOUSE_ROTATION_SENSITIVITY = 0.5

    # Sensitivity constants
    DEFAULT_SENSITIVITY = 5.0
    SENSITIVITY_STEP = 0.1
    MIN_SENSITIVITY = 0.1

    # History for terrain mode
    TERRAIN_HISTORY_LENGTH = 180
    TERRAIN_WIDTH_STEP = 1.0  # Spacing between frequency bands
    TERRAIN_DEPTH_STEP = 0.4  # Spacing between history rows
    TERRAIN_HEIGHT_SCALE = 1.5  # Vertical scaling for audio amplitude
    TERRAIN_Y_OFFSET = -5  # Vertical position offset

    # Performance monitoring
    FPS_SAMPLE_WINDOW = 60

    def __init__(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
        self.width = width
        self.height = height
        self.running = True
        self.mode = self.MODE_BARS
        self.paused = False
        self.sensitivity = self.DEFAULT_SENSITIVITY
        
        # Camera parameters
        self.camera_distance = self.DEFAULT_CAMERA_DISTANCE
        self.camera_rotation = 0
        self.camera_pitch = 0
        self.auto_rotate = True

        # History for Terrain Mode
        self.history_len = self.TERRAIN_HISTORY_LENGTH
        self.history = [np.zeros(64) for _ in range(self.history_len)]

        # Performance monitoring
        self.fps = 0
        self.frame_times = []
        self.frame_time_window = self.FPS_SAMPLE_WINDOW

        # VBOs - will be initialized in init_gl
        self.cube_vbo = None
        self.terrain_vbo = None
        self.terrain_vertex_count = 0

        # Fullscreen state
        self.fullscreen = False

        # Initialize PyGame
        pygame.init()
        pygame.display.set_caption("3D Music Visualizer")
        
        # Enable Multisampling (MSAA) to fix aliasing/banding at distance
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        
        self.screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL | RESIZABLE)

        # Audio device management
        self.available_devices = AudioAnalyzer.list_input_devices()
        self.current_device_idx = 0  # Index in available_devices list

        # Initialize Audio Analyzer with first device
        device_index = self.available_devices[0]['index'] if self.available_devices else None
        self.analyzer = AudioAnalyzer(device_index=device_index)

        # OpenGL Setup (must be after pygame display init and history_len set)
        self.init_gl()

    def init_gl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_BLEND)
        glEnable(GL_NORMALIZE) # Fix lighting for scaled objects
        glEnable(GL_MULTISAMPLE) # Enable MSAA in OpenGL
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glShadeModel(GL_SMOOTH) # Ensure smooth shading
        
        # Disable Specular Highlights (Matte Finish) to prevent "washed out" white glare on sides
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0.0, 0.0, 0.0, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0)

        # Light 0: Centered for Symmetry
        glLightfv(GL_LIGHT0, GL_POSITION,  (0, 20, 15, 0))
        # "Rich Color" Lighting: Moderate Ambient (0.5) + Moderate Diffuse (0.5)
        # This prevents the "Pastel/Flat" look of high ambient, while keeping enough light for depth
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.5, 0.5, 0.5, 1.0)) 
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))

        # Setup viewport and perspective
        self._setup_viewport()

        # Create VBOs for rendering
        self._create_cube_vbo()
        self._create_terrain_vbo()

    def _setup_viewport(self):
        """Setup OpenGL viewport and perspective based on current window size"""
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # Increased zNear from 0.1 to 1.0 to improve depth buffer precision at distance
        gluPerspective(45, (self.width / self.height), 1.0, 300.0)
        glMatrixMode(GL_MODELVIEW)

    def _toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode"""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            # Get current display info for fullscreen resolution
            display_info = pygame.display.Info()
            self.width = display_info.current_w
            self.height = display_info.current_h
            self.screen = pygame.display.set_mode((self.width, self.height),
                                                  DOUBLEBUF | OPENGL | FULLSCREEN)
        else:
            # Return to windowed mode with default size
            self.width = self.DEFAULT_WIDTH
            self.height = self.DEFAULT_HEIGHT
            self.screen = pygame.display.set_mode((self.width, self.height),
                                                  DOUBLEBUF | OPENGL | RESIZABLE)
        self._setup_viewport()

    def _switch_audio_device(self):
        """Cycle to the next available audio input device"""
        if not self.available_devices:
            print("No audio devices available")
            return

        self.current_device_idx = (self.current_device_idx + 1) % len(self.available_devices)
        new_device = self.available_devices[self.current_device_idx]
        print(f"Switching to: {new_device['name']}")
        self.analyzer.switch_device(new_device['index'])

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.mode = (self.mode + 1) % 4
                elif event.key == pygame.K_UP:
                    self.camera_distance = max(self.CAMERA_MIN_DISTANCE,
                                               self.camera_distance - self.CAMERA_ZOOM_STEP)
                elif event.key == pygame.K_DOWN:
                    self.camera_distance = min(self.CAMERA_MAX_DISTANCE,
                                               self.camera_distance + self.CAMERA_ZOOM_STEP)
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.auto_rotate = not self.auto_rotate
                elif event.key == pygame.K_RIGHT:
                    self.sensitivity += self.SENSITIVITY_STEP
                elif event.key == pygame.K_LEFT:
                    self.sensitivity = max(self.MIN_SENSITIVITY, self.sensitivity - self.SENSITIVITY_STEP)
                elif event.key == pygame.K_f or event.key == pygame.K_F11:
                    self._toggle_fullscreen()
                elif event.key == pygame.K_d:
                    self._switch_audio_device()
            elif event.type == pygame.MOUSEWHEEL:
                # Mouse wheel zoom: scroll up = zoom in, scroll down = zoom out
                zoom_delta = event.y * self.CAMERA_ZOOM_STEP
                self.camera_distance = max(self.CAMERA_MIN_DISTANCE,
                                          min(self.CAMERA_MAX_DISTANCE,
                                              self.camera_distance - zoom_delta))
            elif event.type == pygame.VIDEORESIZE:
                # Handle window resize
                self.width = event.w
                self.height = event.h
                self.screen = pygame.display.set_mode((self.width, self.height),
                                                      DOUBLEBUF | OPENGL | RESIZABLE)
                self._setup_viewport()
            elif event.type == pygame.MOUSEMOTION:
                if not self.auto_rotate and event.buttons[0]:  # Left mouse button held
                    self.camera_rotation += event.rel[0] * self.MOUSE_ROTATION_SENSITIVITY
                    self.camera_pitch += event.rel[1] * self.MOUSE_ROTATION_SENSITIVITY
                    
                    # Optional: Clamp pitch to avoid flipping upside down
                    self.camera_pitch = max(-90.0, min(90.0, self.camera_pitch))
                    
    def update(self):
        if self.paused:
            return self.history[-1] if self.history else np.zeros(64)
            
        # Get audio data - stereo (2, 64)
        raw_stereo = self.analyzer.read_audio() * self.sensitivity
        
        # Force Symmetry: Average Left and Right channels
        # This solves the "one side more active" issue for panned audio
        stereo_mean = np.mean(raw_stereo, axis=0)
        raw_stereo[0] = stereo_mean
        raw_stereo[1] = stereo_mean
        
        # Smooth stereo bands
        bands_stereo = np.zeros_like(raw_stereo)
        
        for ch in range(2):
            raw_bands = raw_stereo[ch]
            # Simple 3-point moving average for each channel
            smoothed = np.zeros_like(raw_bands)
            for i in range(len(raw_bands)):
                prev = raw_bands[i-1] if i > 0 else raw_bands[i]
                curr = raw_bands[i]
                next_val = raw_bands[i+1] if i < len(raw_bands)-1 else raw_bands[i]
                smoothed[i] = (prev + curr * 2 + next_val) / 4
            bands_stereo[ch] = smoothed

        # Create mono mix for history and standard modes
        mono_bands = np.mean(bands_stereo, axis=0)

        # Update history with mono bands (for terrain)
        self.history.pop(0)
        self.history.append(mono_bands)
        
        if self.mode == self.MODE_STEREO:
            return bands_stereo
        else:
            return mono_bands
        
    def draw_text_overlay(self):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self.width, 0, self.height)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)

        mode_names = ["Bars", "Terrain", "Vortex", "Stereo"]
        device_name = self.analyzer.get_current_device_name()
        # Truncate device name if too long
        if len(device_name) > 25:
            device_name = device_name[:22] + "..."
        status = f"Mode: {mode_names[self.mode]} | Device: {device_name} | FPS: {self.fps:.1f} | Sens: {self.sensitivity:.1f} | [SPACE] Mode [D]evice [P]ause [R]otate [F]ullscreen"
        pygame.display.set_caption(f"3D Music Visualizer - {status}")

        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def _enable_cube_vbo(self):
        if self.cube_vbo is None: return
        glBindBuffer(GL_ARRAY_BUFFER, self.cube_vbo)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        stride = 6 * 4
        glVertexPointer(3, GL_FLOAT, stride, None)
        glNormalPointer(GL_FLOAT, stride, ctypes.c_void_p(3 * 4))

    def _disable_cube_vbo(self):
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw_cube_fast(self):
        glDrawArrays(GL_QUADS, 0, 24)

    def draw_bars(self, bands):
        num_bars = len(bands)
        spacing = 0.8 # Adjusted spacing for more bands
        total_width = num_bars * spacing
        start_x = -total_width / 2
        
        self._enable_cube_vbo()

        # Lighting Enabled (Standard)

        # Draw reflection
        glPushMatrix()
        glScalef(1, -1, 1)
        glTranslatef(0, 0.1, 0) # Slight offset
        for i, band in enumerate(bands):
            height = (band ** 0.7) * 1.5 
            if height < 0.2: height = 0.2
            x = start_x + i * spacing
            
            glPushMatrix()
            glTranslatef(x, 0, 0)
            # Fainter reflection
            glColor4f(i / num_bars, 1 - (i / num_bars), 0.5 + height/10, 0.3)
            glScalef(0.6, height, 0.6)
            glTranslatef(0, 0.5, 0)
            self.draw_cube_fast()
            glPopMatrix()
        glPopMatrix()
        
        # Main Bars
        for i, band in enumerate(bands):
            height = (band ** 0.7) * 1.5 
            if height < 0.2: height = 0.2
            x = start_x + i * spacing
            
            glPushMatrix()
            glTranslatef(x, 0, 0)
            glColor3f(i / num_bars, 1 - (i / num_bars), 0.5 + height/10)
            glScalef(0.6, height, 0.6)
            glTranslatef(0, 0.5, 0)
            self.draw_cube_fast()
            glPopMatrix()

        self._disable_cube_vbo()

    def _create_terrain_vbo(self):
        """Create VBO for terrain (dynamic, updated each frame)"""
        rows = self.history_len
        cols = 64  # Number of bands

        # Calculate max vertices: (rows-1) strips * cols * 2 vertices per strip * 6 floats (3 pos + 3 color + 3 normal)
        # Added normals for proper lighting on smooth terrain
        max_vertices = (rows - 1) * cols * 2 * 9
        self.terrain_vertex_count = (rows - 1) * cols * 2

        # Generate VBO with dynamic draw hint
        self.terrain_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.terrain_vbo)
        glBufferData(GL_ARRAY_BUFFER, max_vertices * 4, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw_terrain(self):
        """Draw terrain using VBO for better performance (Vectorized)"""
        if self.terrain_vbo is None:
            return

        rows = len(self.history)
        cols = len(self.history[0]) # 64
        width_step = self.TERRAIN_WIDTH_STEP
        depth_step = self.TERRAIN_DEPTH_STEP

        # Create grid coordinates
        # x_coords: (cols,) -> [0, 0.5, 1.0, ...]
        x_coords = np.arange(cols) * width_step
        # z_coords: (rows,) -> [0, 0.4, 0.8, ...]
        z_coords = np.arange(rows) * depth_step
        
        # 1. Calculate heights (Y) for the entire grid
        # self.history: list of arrays. Convert to 2D array (rows, cols)
        history_array = np.array(self.history) # shape (rows, cols)
        
        # Apply height scaling
        heights = (history_array ** 0.7) * self.TERRAIN_HEIGHT_SCALE # shape (rows, cols)
        
        # 2. Calculate Normals Vectorized
        # We need gradients in X and Z directions
        # gradient returns [df/dz, df/dx] for 2D array
        dz_arr, dx_arr = np.gradient(heights)
        
        # Normal vector (-dx, 1, -dz) normalized
        # Stack to shape (rows, cols, 3)
        normals = np.stack([-dx_arr, np.ones_like(heights), -dz_arr], axis=-1)
        
        # Normalize
        norm_norms = np.linalg.norm(normals, axis=2, keepdims=True)
        normals = normals / (norm_norms + 1e-6) # prevent divide by zero
        
        # 3. Calculate Colors Vectorized
        # Hue varies by position: hue = (x/cols + z/rows * 0.2) % 1.0
        # Create X and Z grids for calc
        X_grid, Z_grid = np.meshgrid(np.arange(cols), np.arange(rows)) # shape (rows, cols)
        
        hues = ((X_grid / cols) + (Z_grid / rows) * 0.2) % 1.0
        
        # HSV to RGB vectorization is tricky without a library or complex numpy logic.
        # Since we just need RGB, let's implement a simple fast hsv_to_rgb or use a lookup?
        # Or just use a simpler color ramp for speed.
        # But let's try to stick to the original look.
        # Vectorized HSV to RGB:
        def vectorized_hsv_to_rgb(h, s, v):
            i = (h * 6).astype(int)
            f = (h * 6) - i
            p = v * (1 - s)
            q = v * (1 - f * s)
            t = v * (1 - (1 - f) * s)
            i = i % 6
            
            r = np.select([i==0, i==1, i==2, i==3, i==4, i==5], [v, q, p, p, t, v], default=v)
            g = np.select([i==0, i==1, i==2, i==3, i==4, i==5], [t, v, v, q, p, p], default=p)
            b = np.select([i==0, i==1, i==2, i==3, i==4, i==5], [p, p, t, v, v, q], default=t)
            
            return np.stack([r, g, b], axis=-1)

        colors = vectorized_hsv_to_rgb(hues, 1.0, 1.0) # shape (rows, cols, 3)
        
        # 4. Construct Triangle Strips
        # We need to construct vertices for GL_TRIANGLE_STRIP
        # Each strip connects row z with row z+1
        # Strip i uses rows i and i+1
        # Vertices sequence: (x0, z), (x0, z+1), (x1, z), (x1, z+1), ...
        
        # We handle rows 0 to rows-2
        num_strips = rows - 1
        
        # Prepare indices for extraction
        # We need to interleave row i and row i+1
        # Let's create an array of shape (num_strips, cols * 2, 9)
        # 9 floats: x, y, z, r, g, b, nx, ny, nz
        
        strip_data = np.zeros((num_strips, cols * 2, 9), dtype=np.float32)
        
        # Pre-expand X coordinates for all strips
        # shape (cols,)
        
        # Row 1 (Top of strip) -> indices 0, 2, 4... -> even indices
        # Row 2 (Bottom of strip) -> indices 1, 3, 5... -> odd indices
        
        # Positions
        # X is constant for each column index in the strip
        # Even indices: x [0, 0, 1, 1, 2, 2 ...] -> repeat each el twice ? No.
        # Strip sequence: (col 0, row i), (col 0, row i+1), (col 1, row i), (col 1, row i+1)...
        
        # We can construct the full vertex array using intelligent slicing/stacking
        
        # Slice for "Current Row" (Top vertices)
        top_slice = slice(0, rows-1)
        # Slice for "Next Row" (Bottom vertices)
        bot_slice = slice(1, rows)
        
        # Extract data for top and bottom rows of all strips
        h_top = heights[top_slice]
        h_bot = heights[bot_slice]
        
        n_top = normals[top_slice]
        n_bot = normals[bot_slice]
        
        c_top = colors[top_slice]
        c_bot = colors[bot_slice]
        
        # X coords (broadcasted)
        x_top = np.tile(x_coords, (num_strips, 1))
        x_bot = x_top
        
        # Z coords
        z_top = np.tile(z_coords[top_slice][:, np.newaxis], (1, cols))
        z_bot = np.tile(z_coords[bot_slice][:, np.newaxis], (1, cols))
        
        # Stack everything into (num_strips, cols, 9) for top and bottom separately
        # 9 comps: x, y, z, r, g, b, nx, ny, nz
        
        # Top vertices data: (num_strips, cols, 9)
        v_top = np.stack([x_top, h_top, z_top, 
                          c_top[...,0], c_top[...,1], c_top[...,2],
                          n_top[...,0], n_top[...,1], n_top[...,2]], axis=-1)
                          
        # Bottom vertices data: (num_strips, cols, 9)
        v_bot = np.stack([x_bot, h_bot, z_bot, 
                          c_bot[...,0], c_bot[...,1], c_bot[...,2],
                          n_bot[...,0], n_bot[...,1], n_bot[...,2]], axis=-1)
        
        # Interleave top and bottom
        # Result shape: (num_strips, cols * 2, 9)
        # We can use dstack or simple assignment
        combined = np.zeros((num_strips, cols, 2, 9), dtype=np.float32)
        combined[:, :, 0, :] = v_top
        combined[:, :, 1, :] = v_bot
        
        # Flatten the cols and 2 (top/bot) dimensions -> (num_strips, cols*2, 9)
        final_mesh = combined.reshape(num_strips, cols * 2, 9)
        
        # Flatten all strips for buffer upload -> (num_strips * cols * 2 * 9)
        # Actually, draw_terrain loop draws strips individually, but we put all in one buffer
        
        vertex_data = final_mesh.flatten().astype(np.float32)

        # Update VBO
        glBindBuffer(GL_ARRAY_BUFFER, self.terrain_vbo)
        # Check if buffer needs resizing (shouldn't, but safer to use BufferData if size changed, BufferSubData if same)
        # We initialized with max_vertices. Let's check if our size exceeds it?
        # max_vertices calculation in init was: (rows - 1) * cols * 2 * 9 (floats)
        # Our calculation produces exactly this size.
        
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertex_data.nbytes, vertex_data)

        # Render
        start_x = -(cols * width_step) / 2
        start_z = -(rows * depth_step) / 2

        glPushMatrix()
        glTranslatef(start_x, self.TERRAIN_Y_OFFSET, start_z)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)

        stride = 9 * 4 # 3 pos + 3 color + 3 normal
        # Offsets (in bytes)
        # 0: x, y, z
        # 12: r, g, b
        # 24: nx, ny, nz
        glVertexPointer(3, GL_FLOAT, stride, None)
        glColorPointer(3, GL_FLOAT, stride, ctypes.c_void_p(3 * 4))
        glNormalPointer(GL_FLOAT, stride, ctypes.c_void_p(6 * 4))

        # Draw each strip separately
        vertices_per_strip = cols * 2
        for i in range(num_strips):
            glDrawArrays(GL_TRIANGLE_STRIP, i * vertices_per_strip, vertices_per_strip)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        
        glPopMatrix()

    def draw_circular(self, bands):
        # Edge-to-Edge Vortex Mode
        # Instead of a small circle, we draw multiple concentric rings or a spiral
        # that expands outwards, filling the screen.
        
        self._enable_cube_vbo()
        
        num_bars = len(bands)
        max_rings = 5
        base_radius = 10
        
        # Draw multiple rings for a "Vortex" effect
        for ring in range(max_rings):
            radius = base_radius + ring * 5
            
            # Fade out outer rings
            alpha = 1.0 - (ring / max_rings)
            
            for i, band in enumerate(bands):
                # Map full 360 degrees
                angle = (i / num_bars) * 360 + (ring * 10) # Spiral twist
                rad_angle = math.radians(angle)
                
                # Height dampens with distance
                height = ((band ** 0.7) * 1.5) / (1 + ring * 0.2)
                if height < 0.2: height = 0.2
                
                x = math.cos(rad_angle) * radius
                z = math.sin(rad_angle) * radius
                
                glPushMatrix()
                glTranslatef(x, 0, z)
                glRotatef(-angle, 0, 1, 0)
                
                # Color cycle based on ring and band
                hue = (i / num_bars + ring * 0.1) % 1.0
                r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
                
                glColor4f(r, g, b, alpha)
                
                glScalef(0.6 + ring * 0.1, height, 0.6 + ring * 0.1)
                glTranslatef(0, 0.5, 0)
                self.draw_cube_fast()
                glPopMatrix()

        self._disable_cube_vbo()

    def draw_stereo_bars(self, stereo_bands):
        # Stereo Mode: Left Channel on Left, Right Channel on Right
        # stereo_bands shape: (2, 64)
        
        
        left_bands = stereo_bands[0]
        right_bands = stereo_bands[1]
        
        # Lighting Enabled (Standard)
        
        num_bars = len(left_bands)
        spacing = 0.8
        
        # Calculate total width for one channel
        channel_width = num_bars * spacing
        
        # Offset to separate channels
        center_gap = 2.0
        
        self._enable_cube_vbo()
        
        # --- Reflection Pass (Draw First) ---
        glPushMatrix()
        glScalef(1, -1, 1)
        glTranslatef(0, 0.1, 0) # Slight offset
        
        # Unified Center Gap (minimal)
        center_gap = 0.0
        
        # Reflection: Left Channel
        start_x_left = -center_gap / 2
        for i, band in enumerate(left_bands):
            height = (band ** 0.7) * 1.5
            if height < 0.2: height = 0.2
            x = start_x_left - i * spacing
            
            glPushMatrix()
            glTranslatef(x, 0, 0)
            
            # Gradient: Center Purple (0.5, 0.0, 1.0) -> Edge Cyan (0.0, 1.0, 1.0)
            t = i / num_bars
            r = 0.5 * (1 - t)
            g = t
            b = 1.0
            
            glColor4f(r, g, b, 0.3) # Fainter reflection
            glScalef(0.6, height, 0.6)
            glTranslatef(0, 0.5, 0)
            self.draw_cube_fast()
            glPopMatrix()

        # Reflection: Right Channel
        start_x_right = center_gap / 2
        for i, band in enumerate(right_bands):
            height = (band ** 0.7) * 1.5
            if height < 0.2: height = 0.2
            x = start_x_right + i * spacing
            
            glPushMatrix()
            glTranslatef(x, 0, 0)
            
            # Gradient: Center Purple (0.5, 0.0, 1.0) -> Edge Orange (1.0, 0.5, 0.0)
            t = i / num_bars
            r = 0.5 + 0.5 * t
            g = 0.5 * t
            b = 1.0 * (1 - t)
            
            glColor4f(r, g, b, 0.3) # Fainter reflection
            glScalef(0.6, height, 0.6)
            glTranslatef(0, 0.5, 0)
            self.draw_cube_fast()
            glPopMatrix()
            
        glPopMatrix()
        
        # --- End Reflection Pass ---
        
        # Draw Left Channel (Mirrored: Bass at Center, Treble at Left)
        # Unified Center
        start_x_left = -center_gap / 2
        
        for i, band in enumerate(left_bands):
            height = (band ** 0.7) * 1.5
            if height < 0.2: height = 0.2
            # Go left: -x
            x = start_x_left - i * spacing
            
            glPushMatrix()
            glTranslatef(x, 0, 0)
            
            # Gradient: Center Purple (0.5, 0.0, 1.0) -> Edge Cyan (0.0, 1.0, 1.0)
            t = i / num_bars
            r = 0.5 * (1 - t)
            g = t
            b = 1.0
            
            glColor3f(r, g, b)
            glScalef(0.6, height, 0.6)
            glTranslatef(0, 0.5, 0)
            self.draw_cube_fast()
            glPopMatrix()

        # Draw Right Channel (Standard from center-gap)
        start_x_right = center_gap / 2
        
        for i, band in enumerate(right_bands):
            height = (band ** 0.7) * 1.5
            if height < 0.2: height = 0.2
            x = start_x_right + i * spacing
            
            glPushMatrix()
            glTranslatef(x, 0, 0)
            
            # Gradient: Center Purple (0.5, 0.0, 1.0) -> Edge Orange (1.0, 0.5, 0.0)
            t = i / num_bars
            r = 0.5 + 0.5 * t
            g = 0.5 * t
            b = 1.0 * (1 - t)
            
            glColor3f(r, g, b) 
            glScalef(0.6, height, 0.6)
            glTranslatef(0, 0.5, 0) # pivot at bottom
            self.draw_cube_fast()
            glPopMatrix()
            
        self._disable_cube_vbo()

    def _create_cube_vbo(self):
        """Create Vertex Buffer Object for cube with normals"""
        # Cube vertices with normals (6 faces, 4 vertices each, 3 pos + 3 normal)
        vertices = np.array([
            # Top face (y = 0.5)
            0.5, 0.5, -0.5,  0, 1, 0,
            -0.5, 0.5, -0.5,  0, 1, 0,
            -0.5, 0.5, 0.5,  0, 1, 0,
            0.5, 0.5, 0.5,  0, 1, 0,
            # Bottom face (y = -0.5)
            0.5, -0.5, 0.5,  0, -1, 0,
            -0.5, -0.5, 0.5,  0, -1, 0,
            -0.5, -0.5, -0.5,  0, -1, 0,
            0.5, -0.5, -0.5,  0, -1, 0,
            # Front face (z = 0.5)
            0.5, 0.5, 0.5,  0, 0, 1,
            -0.5, 0.5, 0.5,  0, 0, 1,
            -0.5, -0.5, 0.5,  0, 0, 1,
            0.5, -0.5, 0.5,  0, 0, 1,
            # Back face (z = -0.5)
            0.5, -0.5, -0.5,  0, 0, -1,
            -0.5, -0.5, -0.5,  0, 0, -1,
            -0.5, 0.5, -0.5,  0, 0, -1,
            0.5, 0.5, -0.5,  0, 0, -1,
            # Left face (x = -0.5)
            -0.5, 0.5, 0.5,  -1, 0, 0,
            -0.5, 0.5, -0.5,  -1, 0, 0,
            -0.5, -0.5, -0.5,  -1, 0, 0,
            -0.5, -0.5, 0.5,  -1, 0, 0,
            # Right face (x = 0.5)
            0.5, 0.5, -0.5,  1, 0, 0,
            0.5, 0.5, 0.5,  1, 0, 0,
            0.5, -0.5, 0.5,  1, 0, 0,
            0.5, -0.5, -0.5,  1, 0, 0,
        ], dtype=np.float32)

        # Generate VBO
        self.cube_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw_cube(self):
        """Legacy wrapper method, prefer using _enable_cube_vbo and draw_cube_fast"""
        self._enable_cube_vbo()
        self.draw_cube_fast()
        self._disable_cube_vbo()

    def run(self):
        clock = pygame.time.Clock()

        while self.running:
            frame_start = pygame.time.get_ticks()

            self.handle_input()
            bands = self.update()

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()

            gluLookAt(0, 15, self.camera_distance, 0, 0, 0, 0, 1, 0)

            if self.auto_rotate:
                self.camera_rotation += self.ROTATION_SPEED

            glRotatef(self.camera_pitch, 1, 0, 0)
            glRotatef(self.camera_rotation, 0, 1, 0)

            if self.mode == 0:
                self.draw_bars(bands)
            elif self.mode == 1:
                self.draw_terrain()
            elif self.mode == 2:
                self.draw_circular(bands)
            elif self.mode == 3:
                self.draw_stereo_bars(bands)

            self.draw_text_overlay()

            pygame.display.flip()

            # FPS calculation
            frame_end = pygame.time.get_ticks()
            frame_time = frame_end - frame_start
            self.frame_times.append(frame_time)
            if len(self.frame_times) > self.frame_time_window:
                self.frame_times.pop(0)

            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0

            clock.tick(self.TARGET_FPS)

        self.analyzer.close()
        pygame.quit()
