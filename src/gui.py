import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
import queue
import webbrowser
import os

# To make the GUI aware of the main logic, we need to import it.
# This might require restructuring main.py slightly if it's not already modular.
# For now, we assume we can import the core functions.
from src.pipeline import generate_new_plays, update_with_new_results

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("SHIOLPlus v1.4 - Lottery Intelligence")
        self.geometry("700x550")

        # --- Style Configuration (Dark Theme) ---
        self.style = ttk.Style(self)
        self.style.theme_use('clam') # 'clam' is a good base for custom styling

        # Colors
        BG_COLOR = "#2E2E2E"
        FG_COLOR = "#FFFFFF"
        BTN_COLOR = "#4A4A4A"
        BTN_FOCUS_COLOR = "#5A5A5A"

        self.configure(background=BG_COLOR)
        self.style.configure('.', background=BG_COLOR, foreground=FG_COLOR)
        self.style.configure('TButton', background=BTN_COLOR, foreground=FG_COLOR, borderwidth=1)
        self.style.map('TButton', background=[('active', BTN_FOCUS_COLOR)])
        self.style.configure('TLabel', background=BG_COLOR, foreground=FG_COLOR)
        self.style.configure('TFrame', background=BG_COLOR)

        # --- UI Widgets ---
        self.main_frame = ttk.Frame(self, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Control Frame
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(fill=tk.X, pady=5)

        self.generate_button = ttk.Button(control_frame, text="Generate Plays", command=self.run_generate_thread)
        self.generate_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.update_button = ttk.Button(control_frame, text="Update from CSV", command=self.run_update_thread)
        self.update_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Results Text Area
        self.results_text = scrolledtext.ScrolledText(self.main_frame, wrap=tk.WORD, height=20, state='disabled',
                                                      bg="#1C1C1C", fg=FG_COLOR)
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Status Bar
        status_frame = ttk.Frame(self.main_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Ready", anchor='w')
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.manual_link = ttk.Label(status_frame, text="User Manual", foreground="cyan", cursor="hand2")
        self.manual_link.pack(side=tk.RIGHT)
        self.manual_link.bind("<Button-1>", self.open_manual)

        # --- Threading and Queue Setup ---
        self.task_queue = queue.Queue()
        self.after(100, self.process_queue)

    def set_status(self, text):
        self.status_label.config(text=text)

    def set_buttons_state(self, state):
        self.generate_button.config(state=state)
        self.update_button.config(state=state)

    def display_results(self):
        self.results_text.config(state='normal')
        self.results_text.delete('1.0', tk.END)
        try:
            with open('outputs/predictions_personal.csv', 'r') as f:
                personal_plays = f.read()
            with open('outputs/predictions_syndicate.csv', 'r') as f:
                syndicate_plays = f.read()
            
            results = "--- Personal Plays ---\n" + personal_plays + "\n\n"
            results += "--- Syndicate Plays ---\n" + syndicate_plays
            self.results_text.insert(tk.END, results)
        except FileNotFoundError:
            self.results_text.insert(tk.END, "Could not find prediction files. Please run generation.")
        self.results_text.config(state='disabled')

    def open_manual(self, event=None):
        # Assumes README.md is in the root directory
        manual_path = os.path.abspath("README.md")
        if os.path.exists(manual_path):
            webbrowser.open('file://' + manual_path)
        else:
            self.set_status("Error: README.md not found.")

    def run_generate_thread(self):
        self.set_status("Generating... please wait.")
        self.set_buttons_state('disabled')
        self.task_thread = threading.Thread(target=self.generate_worker)
        self.task_thread.start()

    def generate_worker(self):
        try:
            generate_new_plays()
            self.task_queue.put(("success", "Plays generated successfully!"))
        except Exception as e:
            self.task_queue.put(("error", f"Error: {e}"))

    def run_update_thread(self):
        file_path = filedialog.askopenfilename(
            title="Select a CSV file with new draws",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if not file_path:
            return
        
        self.set_status(f"Updating with {os.path.basename(file_path)}...")
        self.set_buttons_state('disabled')
        self.task_thread = threading.Thread(target=self.update_worker, args=(file_path,))
        self.task_thread.start()

    def update_worker(self, file_path):
        try:
            update_with_new_results(file_path)
            self.task_queue.put(("success", "Update and retraining complete!"))
        except Exception as e:
            self.task_queue.put(("error", f"Error: {e}"))

    def process_queue(self):
        try:
            status, message = self.task_queue.get_nowait()
            self.set_status(message)
            self.set_buttons_state('normal')
            if status == "success":
                self.display_results()
        except queue.Empty:
            pass
        self.after(100, self.process_queue)

def launch_gui():
    app = App()
    app.mainloop()