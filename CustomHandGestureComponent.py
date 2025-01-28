from tkinter import ttk


class HandGestureComponent(ttk.Frame):
    def __init__(self, parent, label_text, button_command=None, delete_button_command=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.label_text = label_text
        self.button_command = button_command
        self.delete_button_command = delete_button_command
        
        # self.canvas = tk.Canvas(self, width=100, height = 50, highlightthickness=0)
        # self.canvas = tk.Canvas(self, highlightthickness=0)
        # self.canvas.grid(pady=(10,5), padx=(10,5))
        # self.draw_label_wrapper(5,5,175,45,10,fill="white",outline="red",width=2)
        
        self.config(width=200, height=200)
        
        container = ttk.Frame(self)
        container.place(relx=0.5, rely=0.5, anchor="center")
        self.container = container
                
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
        
        # self.update_wrapper()
        
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