from tkinter import ttk, Canvas
import cv2
import numpy as np
from PIL import Image, ImageTk



class HandGestureComponent(ttk.Frame):
    def __init__(self, parent, label_text, button_command=None, key_mapping=None, isMouse=False, coords=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.label_text = label_text
        self.button_command = button_command
        self.key_mapping = key_mapping
        
        # Configure TTK styles
        style = ttk.Style()
        style.configure('TButton', padding=6, relief='raised', background='#ccc')
        style.configure('Hover.TButton', padding=6, relief='raised', background='#ddd')
        
        # Process coordinates if they're provided as a flat list
        if coords and isinstance(coords, str):
            coords = [float(x) for x in coords.split()]
            self.coords = list(zip(coords[::2], coords[1::2]))
        else:
            self.coords = coords
        
        # Initialize canvas with white background
        self.canvas = Canvas(self, width=70, height=100, bg="white", highlightthickness=1, highlightbackground="black")
        self.canvas.grid(row=1, column=1, pady=(10, 10), padx=(10, 10))
        
        # Create container for labels and buttons
        container = ttk.Frame(self)
        container.grid(row=1, column=0)
        self.container = container

        # Store the PhotoImage as an instance variable
        self.photo = None

        # Initialize UI elements
        if not isMouse:
            if self.key_mapping and self.key_mapping.get(label_text) and self.key_mapping[label_text][1]:
                self.label = ttk.Label(container, text=f"{self.label_text}: {self.key_mapping[label_text][1]}", 
                                     font=("Arial", 10, 'bold'), foreground="green", background="")
            else:         
                self.label = ttk.Label(container, text=f"{self.label_text}: Not assigned", 
                                     font=("Arial", 10, 'bold'), foreground="red", background="")
            self.label.grid(pady=(10,5))
            
            if button_command:
                self.button = ttk.Button(container, text="Click to record", style='TButton',
                                       command=lambda g=self.label_text, l=self.label: self.button_command(g, l))
                self.button.grid(pady=(10,5))
                
                def on_enter(event):
                    self.button.configure(style='Hover.TButton')
                
                def on_leave(event):
                    self.button.configure(style='TButton')
                
                self.button.bind("<Enter>", on_enter)
                self.button.bind("<Leave>", on_leave)
        else:
            self.label = ttk.Label(container, text=f"{self.label_text}", 
                                 font=("Arial", 10, 'bold'), foreground="green", background="")
            self.label.grid(pady=(10,5))
        
        # Draw hand gesture if coordinates are available
        if self.coords:
            self.draw_hand_gesture()
            
    def normalize_coordinates(self, coords):
        """Normalize coordinates to fit within the canvas"""
        # Find min and max values
        x_coords = [x for x, y in coords]
        y_coords = [y for x, y in coords]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Calculate scaling factors
        x_range = x_max - x_min
        y_range = y_max - y_min
        max_range = max(x_range, y_range)
        
        # Scale and translate coordinates
        normalized_coords = []
        for x, y in coords:
            x_norm = ((x - x_min) / max_range) * 130 + 10  # Leave 10px padding
            y_norm = ((y - y_min) / max_range) * 130 + 10
            normalized_coords.append([int(x_norm), int(y_norm)])
        
        return normalized_coords
        
    def draw_label_wrapper(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1, x2, y1 + radius,
            x2, y2 - radius,
            x2, y2, x2 - radius, y2,
            x1 + radius, y2,
            x1, y2, x1, y2 - radius,
            x1, y1 + radius,
            x1, y1,
        ]
        self.canvas.create_polygon(points, smooth=True, **kwargs)
        
    def update_wrapper(self):
        self.canvas.delete("all")
        self.update_idletasks()
        
        container_x, container_y, container_width, container_height  =(
            self.container.winfo_x(),
            self.container.winfo_y(),
            self.container.winfo_width(),
            self.container.winfo_height(),
        )
        
        padding = 5
        x1 = container_x - padding
        y1 = container_y - padding
        x2 = container_x + container_width + padding
        y2 = container_y + container_height + padding
        
        self.draw_label_wrapper(x1, y1, x2, y2, radius=10, outline="blue", width=2)

    def draw_hand_gesture(self):
        """Draws a hand gesture using OpenCV and displays it in the canvas"""
        if not self.coords or len(self.coords) != 21:
            print(f"Invalid coordinates length: {len(self.coords) if self.coords else 0}")
            return

        # Create a white background image
        image = np.ones((150, 150, 3), dtype=np.uint8) * 255
        
        # Normalize coordinates to fit in the canvas
        landmark_point = self.normalize_coordinates(self.coords)

        # Draw connecting lines with white overlay
        connections = [
            # Thumb
            (2, 3), (3, 4),
            # Index finger
            (5, 6), (6, 7), (7, 8),
            # Middle finger
            (9, 10), (10, 11), (11, 12),
            # Ring finger
            (13, 14), (14, 15), (15, 16),
            # Pinky
            (17, 18), (18, 19), (19, 20),
            # Palm
            (0, 1), (1, 2), (2, 5), (5, 9),
            (9, 13), (13, 17), (17, 0)
        ]

        # Draw lines
        for start_idx, end_idx in connections:
            start_point = tuple(landmark_point[start_idx])
            end_point = tuple(landmark_point[end_idx])
            # Draw black outline
            cv2.line(image, start_point, end_point, (0, 0, 0), 6)
            # Draw white inner line
            cv2.line(image, start_point, end_point, (255, 255, 255), 2)

        # Draw landmark points
        for point in landmark_point:
            cv2.circle(image, tuple(point), 3, (0, 0, 255), -1)

        image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))  # Resize to half

        # Convert OpenCV image to PhotoImage
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        self.photo = ImageTk.PhotoImage(image=pil_image)
        
        # Clear canvas and display the new image
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)