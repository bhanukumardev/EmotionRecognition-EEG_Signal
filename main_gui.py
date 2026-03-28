"""
EEG Emotion Recognition System - GUI Application
A modern minimalist interface for emotion recognition using EEG signals.

Contributors:
- Bhanu Kumar Dev (2328162)
- Adarsh Kumar (2328063)
- Aman Sinha (2306096)
- Srijan (2328235)
- Kanishka (2306118)
- Ashish Yadav (2328157)
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import threading
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from predict import EEGEmotionPredictor, generate_sample_eeg


class EEGEmotionGUI:
    """Modern minimalist GUI for EEG Emotion Recognition."""
    
    # Color scheme - Modern Minimalist
    BG_COLOR = "#f8f9fa"
    PRIMARY_COLOR = "#2c3e50"
    ACCENT_COLOR = "#3498db"
    SUCCESS_COLOR = "#27ae60"
    WARNING_COLOR = "#f39c12"
    DANGER_COLOR = "#e74c3c"
    TEXT_COLOR = "#2c3e50"
    SUBTEXT_COLOR = "#7f8c8d"
    CARD_BG = "#ffffff"
    
    # Emotion colors
    EMOTION_COLORS = {
        'Neutral': '#95a5a6',
        'Positive': '#27ae60',
        'Negative': '#e74c3c'
    }
    
    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("EEG Emotion Recognition System")
        self.root.geometry("900x700")
        self.root.configure(bg=self.BG_COLOR)
        self.root.minsize(800, 600)
        
        # Center window on screen
        self.center_window()
        
        # Initialize predictor
        self.predictor = None
        self.current_emotion = None
        
        # Create UI elements
        self.create_styles()
        self.create_header()
        self.create_contributors_section()
        self.create_control_section()
        self.create_result_section()
        self.create_status_bar()
        
        # Initialize predictor in background
        self.initialize_predictor()
    
    def center_window(self):
        """Center the window on the screen."""
        self.root.update_idletasks()
        width = 900
        height = 700
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_styles(self):
        """Create custom ttk styles for modern look."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles
        style.configure(
            'Modern.TFrame',
            background=self.BG_COLOR
        )
        
        style.configure(
            'Card.TFrame',
            background=self.CARD_BG
        )
        
        style.configure(
            'Title.TLabel',
            font=('Helvetica', 24, 'bold'),
            foreground=self.PRIMARY_COLOR,
            background=self.BG_COLOR
        )
        
        style.configure(
            'Subtitle.TLabel',
            font=('Helvetica', 12),
            foreground=self.SUBTEXT_COLOR,
            background=self.BG_COLOR
        )
        
        style.configure(
            'CardTitle.TLabel',
            font=('Helvetica', 14, 'bold'),
            foreground=self.PRIMARY_COLOR,
            background=self.CARD_BG
        )
        
        style.configure(
            'Contributor.TLabel',
            font=('Helvetica', 11),
            foreground=self.TEXT_COLOR,
            background=self.CARD_BG
        )
        
        style.configure(
            'ID.TLabel',
            font=('Helvetica', 10),
            foreground=self.SUBTEXT_COLOR,
            background=self.CARD_BG
        )
        
        style.configure(
            'Modern.TButton',
            font=('Helvetica', 12, 'bold'),
            foreground='white',
            background=self.ACCENT_COLOR,
            padding=(20, 10)
        )
        style.map(
            'Modern.TButton',
            background=[('active', '#2980b9'), ('pressed', '#1f5f8b')]
        )
        
        style.configure(
            'Emotion.TButton',
            font=('Helvetica', 11),
            padding=(15, 8)
        )
    
    def create_header(self):
        """Create the header section with project title."""
        header_frame = ttk.Frame(self.root, style='Modern.TFrame')
        header_frame.pack(fill='x', padx=30, pady=(30, 10))
        
        # Title
        title_label = ttk.Label(
            header_frame,
            text="Emotion Recognition using EEG Signals",
            style='Title.TLabel'
        )
        title_label.pack()
        
        # Subtitle
        subtitle_label = ttk.Label(
            header_frame,
            text="Machine Learning System for Human Emotion Classification",
            style='Subtitle.TLabel'
        )
        subtitle_label.pack(pady=(5, 0))
        
        # Separator
        separator = ttk.Separator(self.root, orient='horizontal')
        separator.pack(fill='x', padx=30, pady=15)
    
    def create_contributors_section(self):
        """Create the contributors section with all team members."""
        # Container frame
        container = ttk.Frame(self.root, style='Modern.TFrame')
        container.pack(fill='x', padx=30, pady=10)
        
        # Section title
        section_title = ttk.Label(
            container,
            text="Project Contributors",
            style='CardTitle.TLabel'
        )
        section_title.pack(anchor='w', pady=(0, 15))
        
        # Contributors card
        card_frame = tk.Frame(container, bg=self.CARD_BG, bd=0, relief='flat')
        card_frame.pack(fill='x')
        
        # Add shadow effect
        card_frame.configure(highlightbackground='#e0e0e0', highlightthickness=1)
        
        # Contributors data
        contributors = [
            ("Bhanu Kumar Dev", "2328162"),
            ("Adarsh Kumar", "2328063"),
            ("Aman Sinha", "2306096"),
            ("Srijan", "2328235"),
            ("Kanishka", "2306118"),
            ("Ashish Yadav", "2328157")
        ]
        
        # Create 2 rows of 3 contributors each
        for row in range(2):
            row_frame = tk.Frame(card_frame, bg=self.CARD_BG)
            row_frame.pack(fill='x', padx=20, pady=10)
            
            for col in range(3):
                idx = row * 3 + col
                if idx < len(contributors):
                    name, id_num = contributors[idx]
                    
                    # Contributor box
                    contrib_box = tk.Frame(
                        row_frame,
                        bg=self.CARD_BG,
                        bd=1,
                        relief='solid'
                    )
                    contrib_box.pack(side='left', expand=True, fill='both', padx=10, pady=5)
                    contrib_box.configure(highlightbackground='#e0e0e0', highlightthickness=1)
                    
                    # Name label
                    name_label = tk.Label(
                        contrib_box,
                        text=name,
                        font=('Helvetica', 12, 'bold'),
                        fg=self.PRIMARY_COLOR,
                        bg=self.CARD_BG
                    )
                    name_label.pack(pady=(10, 2))
                    
                    # ID label
                    id_label = tk.Label(
                        contrib_box,
                        text=f"ID: {id_num}",
                        font=('Helvetica', 10),
                        fg=self.SUBTEXT_COLOR,
                        bg=self.CARD_BG
                    )
                    id_label.pack(pady=(0, 10))
    
    def create_control_section(self):
        """Create the control section with emotion selection and predict button."""
        # Container frame
        container = ttk.Frame(self.root, style='Modern.TFrame')
        container.pack(fill='x', padx=30, pady=20)
        
        # Section title
        section_title = ttk.Label(
            container,
            text="Emotion Classification",
            style='CardTitle.TLabel'
        )
        section_title.pack(anchor='w', pady=(0, 15))
        
        # Control card
        card_frame = tk.Frame(container, bg=self.CARD_BG, bd=0, relief='flat')
        card_frame.pack(fill='x')
        card_frame.configure(highlightbackground='#e0e0e0', highlightthickness=1)
        
        # Emotion selection frame
        selection_frame = tk.Frame(card_frame, bg=self.CARD_BG)
        selection_frame.pack(fill='x', padx=20, pady=15)
        
        # Label
        select_label = tk.Label(
            selection_frame,
            text="Select Sample Emotion:",
            font=('Helvetica', 12),
            fg=self.TEXT_COLOR,
            bg=self.CARD_BG
        )
        select_label.pack(side='left', padx=(0, 15))
        
        # Emotion variable
        self.selected_emotion = tk.StringVar(value='Neutral')
        
        # Emotion buttons
        emotions = ['Neutral', 'Positive', 'Negative']
        for emotion in emotions:
            btn = tk.Radiobutton(
                selection_frame,
                text=emotion,
                variable=self.selected_emotion,
                value=emotion,
                font=('Helvetica', 11),
                fg=self.TEXT_COLOR,
                bg=self.CARD_BG,
                selectcolor=self.CARD_BG,
                activebackground=self.CARD_BG,
                cursor='hand2'
            )
            btn.pack(side='left', padx=10)
        
        # Predict button frame
        button_frame = tk.Frame(card_frame, bg=self.CARD_BG)
        button_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        # Predict button
        self.predict_btn = tk.Button(
            button_frame,
            text="🧠 Predict Emotion",
            font=('Helvetica', 14, 'bold'),
            fg='white',
            bg=self.ACCENT_COLOR,
            activebackground='#2980b9',
            activeforeground='white',
            cursor='hand2',
            bd=0,
            padx=30,
            pady=12,
            command=self.on_predict
        )
        self.predict_btn.pack()
        
        # Loading indicator (hidden by default)
        self.loading_label = tk.Label(
            button_frame,
            text="Processing...",
            font=('Helvetica', 11),
            fg=self.ACCENT_COLOR,
            bg=self.CARD_BG
        )
    
    def create_result_section(self):
        """Create the result display section."""
        # Container frame
        container = ttk.Frame(self.root, style='Modern.TFrame')
        container.pack(fill='both', expand=True, padx=30, pady=(10, 20))
        
        # Section title
        section_title = ttk.Label(
            container,
            text="Prediction Result",
            style='CardTitle.TLabel'
        )
        section_title.pack(anchor='w', pady=(0, 15))
        
        # Result card
        self.result_card = tk.Frame(container, bg=self.CARD_BG, bd=0, relief='flat')
        self.result_card.pack(fill='both', expand=True)
        self.result_card.configure(highlightbackground='#e0e0e0', highlightthickness=1)
        
        # Initial message
        self.result_content = tk.Frame(self.result_card, bg=self.CARD_BG)
        self.result_content.pack(expand=True)
        
        self.result_icon = tk.Label(
            self.result_content,
            text="🤔",
            font=('Helvetica', 48),
            bg=self.CARD_BG
        )
        self.result_icon.pack(pady=(40, 10))
        
        self.result_text = tk.Label(
            self.result_content,
            text="Click 'Predict Emotion' to analyze EEG signals",
            font=('Helvetica', 14),
            fg=self.SUBTEXT_COLOR,
            bg=self.CARD_BG
        )
        self.result_text.pack(pady=(0, 40))
    
    def create_status_bar(self):
        """Create the status bar at the bottom."""
        self.status_bar = tk.Label(
            self.root,
            text="Initializing...",
            font=('Helvetica', 10),
            fg=self.SUBTEXT_COLOR,
            bg=self.BG_COLOR,
            anchor='w',
            padx=30,
            pady=10
        )
        self.status_bar.pack(fill='x', side='bottom')
    
    def initialize_predictor(self):
        """Initialize the EEG predictor in a background thread using local models."""
        def load_model():
            try:
                # Explicitly use local model directory - no downloads
                model_dir = os.path.join(os.path.dirname(__file__), 'model')
                self.root.after(0, lambda: self.status_bar.config(
                    text="Loading model from local storage...",
                    fg=self.ACCENT_COLOR
                ))
                # Use real-data model if available
                use_real = os.path.exists(os.path.join(model_dir, 'model_info_real.pkl'))
                self.predictor = EEGEmotionPredictor(model_dir=model_dir, use_real_model=use_real)
                model_source = "REAL EEG DATA" if use_real else "Synthetic Data"
                self.root.after(0, lambda: self.status_bar.config(
                    text=f"✓ Model loaded ({model_source}): {self.predictor.model_type.upper()} | Ready",
                    fg=self.SUCCESS_COLOR
                ))
            except Exception as e:
                self.root.after(0, lambda: self.status_bar.config(
                    text=f"✗ Error loading model: {str(e)}",
                    fg=self.DANGER_COLOR
                ))

        threading.Thread(target=load_model, daemon=True).start()
    
    def on_predict(self):
        """Handle predict button click."""
        if self.predictor is None:
            messagebox.showwarning(
                "Model Not Ready",
                "Please wait for the model to finish loading."
            )
            return
        
        # Disable button and show loading
        self.predict_btn.config(state='disabled', text="Processing...")
        self.loading_label.pack(pady=(10, 0))
        self.status_bar.config(text="Analyzing EEG signals...", fg=self.ACCENT_COLOR)
        
        # Run prediction in background thread
        threading.Thread(target=self.run_prediction, daemon=True).start()
    
    def run_prediction(self):
        """Run the prediction in a background thread."""
        try:
            # Get selected emotion
            selected = self.selected_emotion.get()
            
            # Generate synthetic EEG data
            eeg_segment = generate_sample_eeg(
                emotion=selected,
                n_channels=self.predictor.n_channels,
                sampling_rate=self.predictor.sampling_rate
            )
            
            # Make prediction
            result = self.predictor.predict(eeg_segment)
            
            # Update UI in main thread
            self.root.after(0, lambda: self.show_result(result, selected))
            
        except Exception as e:
            self.root.after(0, lambda: self.show_error(str(e)))
    
    def show_result(self, result, input_emotion):
        """Display the prediction result."""
        # Clear previous result
        for widget in self.result_content.winfo_children():
            widget.destroy()
        
        predicted_emotion = result['emotion_label']
        confidence = result['confidence']
        probabilities = result['probabilities']
        
        # Determine if prediction matches input
        is_correct = predicted_emotion == input_emotion
        
        # Result icon based on prediction
        icons = {
            'Neutral': '😐',
            'Positive': '😊',
            'Negative': '😔'
        }
        
        # Result color
        result_color = self.EMOTION_COLORS.get(predicted_emotion, self.TEXT_COLOR)
        
        # Icon
        icon_label = tk.Label(
            self.result_content,
            text=icons.get(predicted_emotion, '🤔'),
            font=('Helvetica', 64),
            bg=self.CARD_BG
        )
        icon_label.pack(pady=(30, 10))
        
        # Prediction label
        pred_label = tk.Label(
            self.result_content,
            text=f"Predicted Emotion: {predicted_emotion}",
            font=('Helvetica', 20, 'bold'),
            fg=result_color,
            bg=self.CARD_BG
        )
        pred_label.pack()
        
        # Confidence
        conf_label = tk.Label(
            self.result_content,
            text=f"Confidence: {confidence:.2%}",
            font=('Helvetica', 14),
            fg=self.TEXT_COLOR,
            bg=self.CARD_BG
        )
        conf_label.pack(pady=(5, 15))
        
        # Probabilities frame
        prob_frame = tk.Frame(self.result_content, bg=self.CARD_BG)
        prob_frame.pack(fill='x', padx=50, pady=10)
        
        # Probability bars
        for emotion, prob in probabilities.items():
            # Emotion row
            row = tk.Frame(prob_frame, bg=self.CARD_BG)
            row.pack(fill='x', pady=3)
            
            # Label
            lbl = tk.Label(
                row,
                text=f"{emotion}:",
                font=('Helvetica', 11),
                fg=self.TEXT_COLOR,
                bg=self.CARD_BG,
                width=10,
                anchor='e'
            )
            lbl.pack(side='left', padx=(0, 10))
            
            # Progress bar container
            bar_container = tk.Frame(row, bg='#e0e0e0', width=200, height=20)
            bar_container.pack(side='left')
            bar_container.pack_propagate(False)
            
            # Progress bar fill
            bar_color = self.EMOTION_COLORS.get(emotion, self.TEXT_COLOR)
            bar_fill = tk.Frame(
                bar_container,
                bg=bar_color,
                width=int(200 * prob)
            )
            bar_fill.place(x=0, y=0, relheight=1)
            
            # Percentage label
            pct_label = tk.Label(
                row,
                text=f"{prob:.1%}",
                font=('Helvetica', 10),
                fg=self.TEXT_COLOR,
                bg=self.CARD_BG,
                width=8,
                anchor='w'
            )
            pct_label.pack(side='left', padx=(10, 0))
        
        # Input info
        info_text = f"Input: {input_emotion} emotion sample | Model: {self.predictor.model_type.upper()}"
        info_label = tk.Label(
            self.result_content,
            text=info_text,
            font=('Helvetica', 10),
            fg=self.SUBTEXT_COLOR,
            bg=self.CARD_BG
        )
        info_label.pack(pady=(20, 30))
        
        # Reset button state
        self.predict_btn.config(state='normal', text="🧠 Predict Emotion")
        self.loading_label.pack_forget()
        self.status_bar.config(
            text=f"✓ Prediction complete: {predicted_emotion} ({confidence:.1%} confidence)",
            fg=self.SUCCESS_COLOR
        )
    
    def show_error(self, error_message):
        """Display error message."""
        for widget in self.result_content.winfo_children():
            widget.destroy()
        
        error_icon = tk.Label(
            self.result_content,
            text="⚠️",
            font=('Helvetica', 48),
            bg=self.CARD_BG
        )
        error_icon.pack(pady=(40, 10))
        
        error_label = tk.Label(
            self.result_content,
            text="Prediction Error",
            font=('Helvetica', 16, 'bold'),
            fg=self.DANGER_COLOR,
            bg=self.CARD_BG
        )
        error_label.pack()
        
        error_detail = tk.Label(
            self.result_content,
            text=error_message,
            font=('Helvetica', 11),
            fg=self.TEXT_COLOR,
            bg=self.CARD_BG,
            wraplength=500
        )
        error_detail.pack(pady=(10, 40))
        
        self.predict_btn.config(state='normal', text="🧠 Predict Emotion")
        self.loading_label.pack_forget()
        self.status_bar.config(text=f"✗ Error: {error_message}", fg=self.DANGER_COLOR)


def main():
    """Main entry point for the GUI application."""
    import logging
    
    # Setup logging for debugging
    log_file = os.path.join(os.path.dirname(__file__), 'gui_debug.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("EEG Emotion Recognition GUI - Starting")
    logger.info("=" * 60)
    
    try:
        # Check for display availability (Linux/headless environments)
        if os.environ.get('DISPLAY') is None and sys.platform != 'win32':
            logger.warning("No DISPLAY environment variable found.")
            logger.info("Attempting to use virtual display (Xvfb)...")
            
            # Try to use Xvfb for virtual display
            try:
                import subprocess
                import time
                # Start Xvfb on display :99
                xvfb_proc = subprocess.Popen(
                    ['Xvfb', ':99', '-screen', '0', '1024x768x24', '-ac'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                os.environ['DISPLAY'] = ':99'
                # Wait for Xvfb to be ready
                time.sleep(1)
                logger.info("Virtual display started on :99")
            except FileNotFoundError:
                logger.error("Xvfb not found. Cannot run GUI in headless environment.")
                logger.error("\n" + "=" * 60)
                logger.error("GUI ERROR: No display available")
                logger.error("=" * 60)
                logger.error("\nThis error occurs when running the GUI in a headless")
                logger.error("environment (like a server or sandbox).")
                logger.error("\nTo fix this on Windows:")
                logger.error("  1. Ensure you have a monitor connected")
                logger.error("  2. Run the script normally: python main_gui.py")
                logger.error("\nTo fix this on Linux (with display):")
                logger.error("  export DISPLAY=:0")
                logger.error("  python main_gui.py")
                logger.error("\nTo run without GUI (command-line mode):")
                logger.error("  python src/predict.py")
                logger.error("=" * 60)
                # Only use input() in interactive mode
                if sys.stdin.isatty():
                    input("\nPress Enter to exit...")  # Keep window open
                sys.exit(1)
        
        # Create main window
        logger.info("Creating Tkinter root window...")
        root = tk.Tk()
        logger.info("Tkinter root window created successfully")
        
        # Set DPI awareness for Windows
        if sys.platform == 'win32':
            try:
                from ctypes import windll
                windll.shcore.SetProcessDpiAwareness(1)
                logger.info("Windows DPI awareness enabled")
            except Exception as e:
                logger.debug(f"Could not set DPI awareness: {e}")
        
        # Create application
        logger.info("Initializing EEG Emotion GUI...")
        app = EEGEmotionGUI(root)
        logger.info("GUI initialized successfully")
        
        # Start main loop
        logger.info("Starting main event loop...")
        root.mainloop()
        
    except tk.TclError as e:
        logger.error(f"\n{'=' * 60}")
        logger.error(f"GUI ERROR: {str(e)}")
        logger.error(f"{'=' * 60}")
        logger.error("\nThis error typically occurs when:")
        logger.error("  1. Running on a system without a graphical display")
        logger.error("  2. Running via SSH without X11 forwarding")
        logger.error("  3. Running in a container/VM without display access")
        logger.error("\nTroubleshooting steps:")
        logger.error("  - Windows: Ensure you're running locally with a monitor")
        logger.error("  - Linux: Run 'export DISPLAY=:0' before starting")
        logger.error("  - Remote: Use X11 forwarding (ssh -X)")
        logger.error(f"{'=' * 60}")
        # Only use input() in interactive mode
        if sys.stdin.isatty():
            input("\nPress Enter to exit...")  # Keep window open
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\n{'=' * 60}")
        logger.error(f"UNEXPECTED ERROR: {str(e)}")
        logger.error(f"{'=' * 60}")
        import traceback
        logger.error(traceback.format_exc())
        logger.error(f"{'=' * 60}")
        # Only use input() in interactive mode
        if sys.stdin.isatty():
            input("\nPress Enter to exit...")  # Keep window open
        sys.exit(1)
    
    finally:
        logger.info("GUI application ended")


if __name__ == '__main__':
    main()