from tkinter import ttk, Canvas
import numpy as np
import cv2
from PIL import Image
from PIL import ImageFont, ImageTk
class HandGestureComponent(ttk.Frame):
    def __init__(self, parent, label_text, button_command=None, delete_button_command=None, key_mapping = None, coords=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.label_text = label_text
        self.button_command = button_command
        self.delete_button_command = delete_button_command
        self.key_mapping = key_mapping
        
        # self.canvas = tk.Canvas(self, width=100, height = 50, highlightthickness=0)
        # self.canvas = tk.Canvas(self, highlightthickness=0)
        # self.canvas.grid(pady=(10,5), padx=(10,5))
        # self.draw_label_wrapper(5,5,175,45,10,fill="white",outline="red",width=2)
        
        self.config(width=200, height=200)
        
        container = ttk.Frame(self)
        container.place(relx=0.5, rely=0.5, anchor="center")
        self.container = container

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
        
        if label_text in self.key_mapping and self.key_mapping[label_text]:
            print(f"{label_text} exists in key_mapping")        
            if self.key_mapping[label_text][1]:
                print(f"{label_text} has keybind")
                self.label = ttk.Label(container, text=f"{self.label_text}: {self.key_mapping[label_text][1]}", font=("Venite Adoremus", 10, 'bold'), foreground="green")
            else:
                print(f"{label_text} has no keybind")         
                self.label = ttk.Label(container, text=f"{self.label_text}: Not assigned", font=("Venite Adoremus", 10, 'bold'), foreground="red")
        else:
            print(f"{label_text} does not exist in key_mapping")
            self.label = ttk.Label(container, text=f"{self.label_text}: Not assigned", font=("Venite Adoremus", 10, 'bold'), foreground="red")
        self.label.grid(pady=(10,5))
        
        self.button = ttk.Button(container, text="Click to record", command=lambda g=self.label_text, l=self.label: self.button_command(g, l))
        self.button.grid(pady=(10,5))
        
        self.delete_button = ttk.Button(container, text="Delete", command=lambda g=label_text: self.delete_button_command(g, self))
        self.delete_button.grid(pady=(10,5))
        
        def hoverButtonEffect(customHGName):
            # Add hover effect to buttons
            def on_enter(event, b=self.button):
                b.config(style='Hover.TButton')
                print(f"Hover over button for customHG {customHGName}")
                    
            def on_leave(event, b=self.button):
                b.config(style='TButton')
                    
            return on_enter, on_leave
                 
        record_on_enter, record_on_leave = hoverButtonEffect(label_text)    
        self.button.bind("<Enter>", record_on_enter)
        self.button.bind("<Leave>", record_on_leave)
        delete_on_enter, delete_on_leave = hoverButtonEffect(label_text)    
        self.delete_button.bind("<Enter>", delete_on_enter)
        self.delete_button.bind("<Leave>", delete_on_leave)
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