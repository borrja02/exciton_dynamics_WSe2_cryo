import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading, time, serial

# === Global Configuration ===
default_fontsize = 20  # Font size for all GUI elements

# === Main Application Class ===
class MonochromatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Random Number Generator')
        self.root.geometry('1000x1000')

        # Enable resizable layout
        for i in range(3):
            self.root.columnconfigure(i, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)

        font_large = ('Arial', default_fontsize)
        font_button = ('Arial', default_fontsize, 'bold')

        # === Acquisition Time Input ===
        self.label_time = tk.Label(root, text='Acquisition time (s):', font=font_large)
        self.label_time.grid(row=0, column=0, sticky='w', padx=10, pady=5)

        self.time_entry = tk.Entry(root, font=font_large, width=15)
        self.time_entry.insert(0, '3')  # Default value
        self.time_entry.grid(row=0, column=1, sticky='ew', padx=10, pady=5)

        # === Start/Stop Button ===
        self.start_button = tk.Button(root, text='Start', command=self.toggle_running, font=font_button)
        self.start_button.grid(row=0, column=2, padx=10, pady=5)

        # === Plot Setup ===
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.ax.set_facecolor('azure')
        self.ax.set_title('Random Number Histogram', fontsize=default_fontsize, fontweight='bold')
        self.ax.set_xlabel('Value', fontsize=default_fontsize)
        self.ax.set_ylabel('Frequency', fontsize=default_fontsize)
        self.ax.grid(True, color='gray')

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=3, sticky='nsew', padx=10, pady=10)

        # === Save Button ===
        self.save_button = tk.Button(root, text='Save', command=self.save_plot, font=font_button)
        self.save_button.grid(row=2, column=1, pady=10)

        # === State Variables ===
        self.running = False
        self.started_once = False
        self.mean = 10  # Mean of generated random numbers

        # === Arduino Serial Connection ===
        try:
            self.arduino = serial.Serial('COM3', 9600, timeout=1)  # Update COM port as needed
            time.sleep(2)  # Wait for Arduino initialization
        except Exception as e:
            print(f'[ERROR] Could not connect to Arduino: {e}')
            self.arduino = None

    def toggle_running(self):
        """Start or stop acquisition."""
        if self.running:
            self.running = False
            self.start_button.config(text='Restart')
        else:
            self.running = True
            self.start_button.config(text='Stop')

            if not self.started_once:
                self.started_once = True

            # Start acquisition in a separate thread
            threading.Thread(target=self.generate_numbers, daemon=True).start()

    def generate_numbers(self):
        """Main acquisition loop: communicates with Arduino and updates histogram."""
        numbers = []

        while self.running:
            try:
                acquisition_time = float(self.time_entry.get())
                if acquisition_time < 0:
                    acquisition_time = 0
            except ValueError:
                acquisition_time = 1  # Fallback to 1 second

            # Send step command to Arduino
            if self.arduino:
                try:
                    self.arduino.write(b'S')  # Command Arduino to step
                    line = self.arduino.readline().decode().strip()
                    while line != 'DONE':
                        line = self.arduino.readline().decode().strip()
                except Exception as e:
                    print(f'[ERROR] Arduino communication: {e}')

            # Simulate data acquisition
            time.sleep(acquisition_time)
            numbers.append(np.random.normal(self.mean, 1))
            self.update_plot(numbers)

    def update_plot(self, numbers):
        """Clear and redraw histogram with updated data."""
        self.ax.clear()
        self.ax.grid(True, color='gray', zorder=0)
        self.ax.hist(numbers, bins=20, color='cornflowerblue', edgecolor='k', density=True, zorder=2)
        self.ax.set_title('Random Numbers Histogram', fontsize=default_fontsize, fontweight='bold')
        self.ax.set_xlabel('Value', fontsize=default_fontsize)
        self.ax.set_ylabel('Frequency', fontsize=default_fontsize)
        self.canvas.draw()

    def save_plot(self):
        """Save the histogram as a PNG file."""
        file_path = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG files', '*.png'), ('All Files', '*.*')],
            title='Save plot as...'
        )
        if file_path:
            self.fig.savefig(file_path)
            print(f'[INFO] Plot saved at: {file_path}')

# === Application Launch ===
if __name__ == '__main__':
    root = tk.Tk()
    app = MonochromatorApp(root)
    root.mainloop()
