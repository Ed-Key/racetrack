import tkinter as tk
import numpy as np
import pickle
from tkinter import filedialog, messagebox

class TrackEditor:
    """Simple GUI for manually creating racetracks."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Racetrack Editor")
        
        # Track parameters
        self.height = 30
        self.width = 32
        self.cell_size = 20
        
        # Track representation
        # 0 = track, 1 = wall, 2 = start line, 3 = finish line
        self.track = np.ones((self.height, self.width), dtype=int)
        
        # Current drawing mode
        self.mode = tk.IntVar(value=0)
        
        # UI setup
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(pady=10, padx=10)
        
        # Canvas for track drawing
        self.canvas_frame = tk.Frame(main_frame)
        self.canvas_frame.pack(side=tk.LEFT)
        
        self.canvas = tk.Canvas(
            self.canvas_frame, 
            width=self.width * self.cell_size,
            height=self.height * self.cell_size,
            bg="white"
        )
        self.canvas.pack()
        
        # Draw grid
        self.draw_grid()
        
        # Control panel
        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.RIGHT, padx=10)
        
        # Mode selection
        mode_frame = tk.LabelFrame(control_frame, text="Drawing Mode")
        mode_frame.pack(pady=10, fill=tk.X)
        
        modes = [
            ("Track (White)", 0),
            ("Wall (Black)", 1),
            ("Start Line (Red)", 2),
            ("Finish Line (Green)", 3)
        ]
        
        for text, value in modes:
            tk.Radiobutton(
                mode_frame, 
                text=text, 
                variable=self.mode, 
                value=value
            ).pack(anchor=tk.W)
        
        # Buttons
        button_frame = tk.Frame(control_frame)
        button_frame.pack(pady=10, fill=tk.X)
        
        tk.Button(
            button_frame, 
            text="Clear Track", 
            command=self.clear_track
        ).pack(fill=tk.X, pady=2)
        
        tk.Button(
            button_frame, 
            text="Fill With Walls", 
            command=self.fill_walls
        ).pack(fill=tk.X, pady=2)
        
        tk.Button(
            button_frame, 
            text="Save Track", 
            command=self.save_track
        ).pack(fill=tk.X, pady=2)
        
        tk.Button(
            button_frame, 
            text="Load Track", 
            command=self.load_track
        ).pack(fill=tk.X, pady=2)
        
        # Resize controls
        resize_frame = tk.LabelFrame(control_frame, text="Resize Track")
        resize_frame.pack(pady=10, fill=tk.X)
        
        tk.Label(resize_frame, text="Width:").grid(row=0, column=0, sticky=tk.W)
        self.width_var = tk.StringVar(value=str(self.width))
        tk.Entry(resize_frame, textvariable=self.width_var, width=5).grid(row=0, column=1)
        
        tk.Label(resize_frame, text="Height:").grid(row=1, column=0, sticky=tk.W)
        self.height_var = tk.StringVar(value=str(self.height))
        tk.Entry(resize_frame, textvariable=self.height_var, width=5).grid(row=1, column=1)
        
        tk.Button(
            resize_frame, 
            text="Resize", 
            command=self.resize_track
        ).grid(row=2, column=0, columnspan=2, pady=5)
        
        # Instructions
        instruction_frame = tk.LabelFrame(control_frame, text="Instructions")
        instruction_frame.pack(pady=10, fill=tk.X)
        
        instructions = """
        Click to draw/erase cells.
        
        Color codes:
        - White: Track
        - Black: Wall
        - Red: Start line
        - Green: Finish line
        
        Save your track when done.
        """
        tk.Label(instruction_frame, text=instructions, justify=tk.LEFT).pack(anchor=tk.W)
        
        # Bind canvas events
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)
        
        # Draw initial track
        self.update_canvas()
    
    def draw_grid(self):
        """Draw the grid on the canvas."""
        # Vertical lines
        for i in range(0, self.width + 1):
            x = i * self.cell_size
            self.canvas.create_line(
                x, 0, x, self.height * self.cell_size, 
                fill="gray", width=1
            )
        
        # Horizontal lines
        for i in range(0, self.height + 1):
            y = i * self.cell_size
            self.canvas.create_line(
                0, y, self.width * self.cell_size, y, 
                fill="gray", width=1
            )
    
    def update_canvas(self):
        """Update the canvas to reflect the current track state."""
        # Clear canvas
        self.canvas.delete("cell")
        
        # Draw cells
        for y in range(self.height):
            for x in range(self.width):
                color = self.get_color(self.track[y, x])
                self.canvas.create_rectangle(
                    x * self.cell_size, y * self.cell_size,
                    (x + 1) * self.cell_size, (y + 1) * self.cell_size,
                    fill=color, outline="gray", tags="cell"
                )
        
        # Bring grid to front
        self.canvas.tag_raise("grid")
    
    def get_color(self, value):
        """Convert cell value to color."""
        colors = {
            0: "white",  # Track
            1: "black",  # Wall
            2: "red",    # Start line
            3: "green"   # Finish line
        }
        return colors.get(value, "gray")
    
    def draw(self, event):
        """Handle drawing on the canvas."""
        # Convert canvas coordinates to grid coordinates
        x = event.x // self.cell_size
        y = event.y // self.cell_size
        
        # Ensure coordinates are within grid
        if 0 <= x < self.width and 0 <= y < self.height:
            # Update track with current mode
            self.track[y, x] = self.mode.get()
            
            # Update canvas
            color = self.get_color(self.mode.get())
            self.canvas.create_rectangle(
                x * self.cell_size, y * self.cell_size,
                (x + 1) * self.cell_size, (y + 1) * self.cell_size,
                fill=color, outline="gray", tags="cell"
            )
    
    def clear_track(self):
        """Clear the track (set all cells to walls)."""
        self.track = np.ones((self.height, self.width), dtype=int)
        self.update_canvas()
    
    def fill_walls(self):
        """Fill the track with walls."""
        self.track = np.ones((self.height, self.width), dtype=int)
        self.update_canvas()
    
    def resize_track(self):
        """Resize the track based on user input."""
        try:
            new_width = int(self.width_var.get())
            new_height = int(self.height_var.get())
            
            if new_width < 5 or new_height < 5:
                messagebox.showerror("Error", "Width and height must be at least 5.")
                return
            
            # Create new track with the resized dimensions
            new_track = np.ones((new_height, new_width), dtype=int)
            
            # Copy as much of the old track as possible
            h = min(self.height, new_height)
            w = min(self.width, new_width)
            new_track[:h, :w] = self.track[:h, :w]
            
            # Update track dimensions
            self.width = new_width
            self.height = new_height
            self.track = new_track
            
            # Resize canvas
            self.canvas.config(
                width=self.width * self.cell_size,
                height=self.height * self.cell_size
            )
            
            # Redraw grid and track
            self.canvas.delete("all")
            self.draw_grid()
            self.update_canvas()
            
        except ValueError:
            messagebox.showerror("Error", "Width and height must be integers.")
    
    def save_track(self):
        """Save the track to a file."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(self.track, f)
                messagebox.showinfo("Success", f"Track saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save track: {e}")
    
    def load_track(self):
        """Load a track from a file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    track = pickle.load(f)
                
                # Update track dimensions
                self.height, self.width = track.shape
                self.track = track
                
                # Update dimension variables
                self.width_var.set(str(self.width))
                self.height_var.set(str(self.height))
                
                # Resize canvas
                self.canvas.config(
                    width=self.width * self.cell_size,
                    height=self.height * self.cell_size
                )
                
                # Redraw grid and track
                self.canvas.delete("all")
                self.draw_grid()
                self.update_canvas()
                
                messagebox.showinfo("Success", f"Track loaded from {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load track: {e}")

def main():
    root = tk.Tk()
    app = TrackEditor(root)
    root.mainloop()

if __name__ == "__main__":
    main()
