import customtkinter as ctk
from tkinter import ttk, filedialog
import tkinter as tk
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy import stats
import math

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\SpinkC\OneDrive - Kennedy Krieger\Desktop\build\build\build\assets\frame0")

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

PURPLE = "#400287"
WHITE = "#FFFFFF"

cells = {}

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def validate_numeric_input(self, P):
    # Allow empty string or only digits (including decimal point and minus sign)
    return P == "" or P.replace('.', '').replace('-', '', 1).isdigit()
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("CDC Calc")
        self.geometry("1200x800")
        self.resizable(True, True)

        self.configure(fg_color=WHITE)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.main_frame = ctk.CTkFrame(self, corner_radius=10, fg_color=WHITE,
                                       border_color=PURPLE, border_width=2)
        self.main_frame.grid(row=0, column=0, rowspan=2, padx=20, pady=20, 
                             sticky="nsew")

        self.create_input_section()
        self.create_table_section()
        self.create_output_section()
        self.bind_scroll_events()

        # Initialize the list to store spreadsheet entries
        self.spreadsheet_entries = []

    def create_input_section(self):
        input_frame = ctk.CTkFrame(self.main_frame, width=300, fg_color=WHITE)
        input_frame.pack(pady=20, padx=20, fill="y", expand=False)

        inputs = [
            ("Graph Title:", 24.0, 41.0),
            ("Graph Subtitle:", 24.0, 85.0),
            ("Y-Axis Label:", 24.0, 129.0),
            ("Condition Names (comma-separated):", 24.0, 173.0),
            ("# of Phases:", 24.0, 217.0),
            ("# of Sessions", 24.0, 261.0)
        ]

        self.entries = []
        for idx, (text, x, y) in enumerate(inputs):
            label = ctk.CTkLabel(input_frame, text=text, font=("Arial", 14),
                                 text_color=PURPLE)
            label.pack(pady=(10, 0) if idx == 0 else 10, anchor="w")

            entry = ctk.CTkEntry(input_frame, width=220, height=32,
                                font=("Arial", 12), border_width=1,
                                corner_radius=6, fg_color=WHITE,
                                border_color=PURPLE, text_color=PURPLE)
            entry.pack(pady=5)
            self.entries.append(entry)

        self.create_table_btn = ctk.CTkButton(
            input_frame,
            text="Generate Table",
            command=self.on_create_table,
            height=40,
            font=("Arial", 14, "bold"),
            corner_radius=8,
            fg_color=PURPLE,
            hover_color="#600387"
        )
        self.create_table_btn.pack(pady=10)

        self.generate_graph_btn = ctk.CTkButton(
            input_frame,
            text="Generate Graph",
            command=self.on_generate_graph,
            height=40,
            font=("Arial", 14, "bold"),
            corner_radius=8,
            fg_color=PURPLE,
            hover_color="#600387"
        )
        self.generate_graph_btn.pack(pady=10)

        self.clear_all_btn = ctk.CTkButton(
            input_frame,
            text="Clear All",
            command=self.on_clear_all,
            height=40,
            font=("Arial", 14, "bold"),
            corner_radius=8,
            fg_color=PURPLE,
            hover_color="#600387"
        )
        self.clear_all_btn.pack(pady=10)

    def create_table_section(self):
        table_container = ctk.CTkFrame(self, corner_radius=10, fg_color=WHITE,
                                       border_color=PURPLE, border_width=2)
        table_container.grid(row=0, column=1, rowspan=2, padx=20,
                             pady=20, sticky="nsew")

        self.canvas = ctk.CTkCanvas(table_container, bg=WHITE,
                                    highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.v_scroll = ctk.CTkScrollbar(table_container, orientation="vertical", 
                                         command=self.canvas.yview)
        self.v_scroll.pack(side="right", fill="y")

        self.h_scroll = ctk.CTkScrollbar(self, orientation="horizontal", 
                                         command=self.canvas.xview)
        self.h_scroll.grid(row=2, column=1, sticky="ew", padx=20)

        self.canvas.configure(
            yscrollcommand=self.v_scroll.set,
            xscrollcommand=self.h_scroll.set
        )

        self.spreadsheet_frame = ctk.CTkFrame(self.canvas, fg_color=WHITE)
        self.canvas.create_window((0, 0), window=self.spreadsheet_frame, 
                                  anchor="nw")

        self.spreadsheet_frame.bind("<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
    def create_output_section(self):
        output_frame = ctk.CTkFrame(self, width=400, corner_radius=10, 
                                    fg_color=WHITE, border_color=PURPLE, 
                                    border_width=2)
        output_frame.grid(row=0, column=2, rowspan=2, 
                          padx=20, pady=20, sticky="nsew")

        title = ctk.CTkLabel(output_frame, text="Output Console",
                           font=("Arial", 16, "bold"), text_color=PURPLE)
        title.pack(pady=10)

        self.output_text = ctk.CTkTextbox(output_frame, wrap="word",
                                        font=("Consolas", 12),
                                        width=380, height=600,
                                        activate_scrollbars=False,
                                        fg_color=WHITE, text_color=PURPLE)
        self.output_text.pack(padx=10, pady=10, fill="both", expand=True)

        output_v_scroll = ctk.CTkScrollbar(output_frame, 
                                           command=self.output_text.yview)
        output_v_scroll.pack(side="right", fill="y")
        self.output_text.configure(yscrollcommand=output_v_scroll.set)

    def bind_scroll_events(self):
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind_all("<Shift-MouseWheel>", self.on_mousewheel)
        self.bind("<Up>", lambda e: self.canvas.yview_scroll(-1, "units"))
        self.bind("<Down>", lambda e: self.canvas.yview_scroll(1, "units"))
        self.bind("<Left>", lambda e: self.canvas.xview_scroll(-1, "units"))
        self.bind("<Right>", lambda e: self.canvas.xview_scroll(1, "units"))

    def on_mousewheel(self, event):
        shift_pressed = event.state & 0x1
        scroll_dir = -1 if event.delta > 0 else 1
        if shift_pressed:
            self.canvas.xview_scroll(scroll_dir, "units")
        else:
            self.canvas.yview_scroll(scroll_dir, "units")

    def get_condition_names(self):
        return [name.strip() for name in self.entries[3].get().split(',') if name.strip()]

    def create_dynamic_table(self):
        num_phases = int(self.entries[4].get()) if self.entries[4].get().isdigit() else 0
        num_sessions = int(self.entries[5].get()) if self.entries[5].get().isdigit() else 0
    
        if num_phases == 0 or num_sessions == 0:
            self.log_message("Error: Please enter valid numbers for # of Sessions and # of Phases")
            return
    
        # Clear existing widgets in spreadsheet frame
        for widget in self.spreadsheet_frame.winfo_children():
            widget.destroy()
    
        # Clear existing spreadsheet entries
        self.spreadsheet_entries = []
    
        condition_names = self.get_condition_names()
        cell_width = 9  # Consistent width for all cells
    
        for i in range(num_sessions + 1):
            row_entries = []
            for j in range(num_phases + 1):
                if i == 0:  # Header row
                    if j == 0:
                        header = ctk.CTkLabel(self.spreadsheet_frame, 
                                              text="Session", width=cell_width*10, 
                                              fg_color=WHITE, text_color=PURPLE)
                        header.grid(row=i, column=j, padx=2,
                                    pady=2, sticky='nsew')
                        row_entries.append(header)
                    else:
                        frame = ctk.CTkFrame(self.spreadsheet_frame, fg_color=WHITE)
                        frame.grid(row=i, column=j, padx=2, pady=2, sticky='nsew')

                        combo = ctk.CTkOptionMenu(
                                    frame,
                                    width=cell_width*8,
                                    values=condition_names,
                                    fg_color=WHITE,
                                    text_color=PURPLE,
                                    button_color=PURPLE,
                                    button_hover_color="#600387"
                                    )              
                        combo.set(f"Phase {j}" if not condition_names else condition_names[0] if condition_names else "")
                        combo.pack(side="left", expand=True, fill="x")
                        if j > 1:
                            plus_minus_combo = ctk.CTkOptionMenu(frame,
                                                        width=cell_width*2,
                                                        values=["+", "-"],  
                                                        fg_color=WHITE,
                                                        text_color=PURPLE,
                                                        button_color=PURPLE,
                                                        button_hover_color="#600387"
                                                        )
                            plus_minus_combo.set("+")
                            plus_minus_combo.pack(side="left")
                            row_entries.append((combo, plus_minus_combo))
                        else:
                            row_entries.append(combo)
                        
                elif j == 0:  # First column (session labels)
                    entry = ctk.CTkEntry(
                        self.spreadsheet_frame,
                        width=cell_width*10,
                        fg_color=WHITE,
                        text_color=PURPLE,
                        border_color=PURPLE
                    )
                    entry.insert(0, f"Session {i}")
                    entry.grid(row=i, column=j, padx=2, pady=2, sticky='nsew')
                    row_entries.append(entry)
                else:  # Data cells
                    frame = ctk.CTkFrame(self.spreadsheet_frame,
                                         fg_color=WHITE)
                    frame.grid(row=i, column=j, padx=2, pady=2, sticky='nsew')
                    
                    entry = ctk.CTkEntry(
                        frame,
                        width=cell_width*8,
                        fg_color=WHITE,
                        text_color=PURPLE,
                        border_color=PURPLE
                    )
                    entry.pack(side="left", expand=True, fill="x")
                    entry.bind('<Control-v>', lambda e, row=i,
                               col=j: self.paste_from_clipboard(e, row, col))
                    
                    row_entries.append((entry))
    
            self.spreadsheet_entries.append(row_entries)
    
        # Configure grid resizing
        for i in range(num_sessions + 1):
            self.spreadsheet_frame.grid_rowconfigure(i, weight=1)
        for j in range(num_phases + 1):
            self.spreadsheet_frame.grid_columnconfigure(j, weight=1)
    
        self.spreadsheet_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
        # Configure grid resizing
        for i in range(num_sessions + 1):
            self.spreadsheet_frame.grid_rowconfigure(i, weight=1)
        for j in range(num_phases + 1):
            self.spreadsheet_frame.grid_columnconfigure(j, weight=1)
    
        self.spreadsheet_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def log_message(self, message):
        self.output_text.configure(state="normal")
        self.output_text.insert("end", message + "\n")
        self.output_text.see("end")
        self.output_text.configure(state="disabled")

    def on_create_table(self):
        # Ensure `self.entries` exists and contains valid CTkEntry objects
        if not isinstance(self.entries, list) or len(self.entries) < 6:
            self.log_message("Error: Input fields are missing. Resetting...")
            self.entries = [None] * 6  # Reset the list with placeholders
            return  # Stop execution to prevent further errors

        # Define field names for better debugging messages
        field_names = ["Graph Title", "Graph Subtitle", "Y-Axis Label", 
                       "Condition Names", "# of Phases", "# of Sessions"]

        # Ensure each input field exists and is a CTkEntry
        for i in range(6):
            if not isinstance(self.entries[i], ctk.CTkEntry):
                self.log_message(f"Error: {field_names[i]} is missing or invalid.")
                return  # Stop execution if an invalid entry is found

        try:
            # Validate # of Phases and # of Sessions
            num_phases = int(self.entries[4].get().strip()) if self.entries[4].get().strip().isdigit() else 1
            num_sessions = int(self.entries[5].get().strip()) if self.entries[5].get().strip().isdigit() else 1

            if num_phases < 1 or num_sessions < 1:
                raise ValueError("Phases and sessions must be at least 1.")

        except (IndexError, ValueError, AttributeError):
            self.log_message("Error: Please enter a valid number for # of Phases and # of Sessions.")
            return  # Stop execution if invalid input is found

        # Log successful table creation
        self.log_message("=== New Table Created ===")

        self.create_dynamic_table()

    def on_generate_graph(self):
        phase_names, data, predicted_directions = self.get_entered_values()
        
        #Adds a '0' to predicted directions to match shape, has no effect on readings
        predicted_directions.append(0)
                
        self.treatment_phase_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                     15, 16, 17, 18, 19, 20, 21, 22, 23]
        self.requirement_list = [3, 4, 5, 6, 6, 7, 8, 8, 9, 9, 10, 11, 12,
                                 12, 12, 13, 13, 14, 14, 15, 15]
        
        if not data:
            self.log_message("Error: No data entered.")
            return
            
        numerical_data = []
        for row in data:
            numerical_row = []
            for value in row[1:]:
                try:
                    numerical_row.append(float(value) if value.strip() else None)
                except ValueError:
                    numerical_row.append(None)
            numerical_data.append(numerical_row)
    
        phase_lines = self.calculate_phase_lines()
        fig, ax = plt.subplots(figsize=(12, 8))
        sessions = [int(row[0].split()[1]) for row in data]
    
        # Store phase analytics for comparison
        phase_analytics = []
    
        for phase_idx in range(len(phase_names)):
            phase_data = [float(row[phase_idx + 1]) for row in data if row[phase_idx + 1]]
            phase_sessions = [s for s, row in zip(sessions, data) if row[phase_idx + 1]]
    
            if phase_data:
                # Plot data points
                ax.plot(phase_sessions, phase_data, 'o-', color='black',
                        markerfacecolor='black', markeredgecolor='black')
    
                mean_val = np.mean(phase_data)
                std_dev = np.std(phase_data)
                direction = predicted_directions[phase_idx]
                
                # Determine direction adjustment multiplier (+1 or -1)
                direction_mult = 1 if direction == "+" else -1
    
                # Apply adjustment based on direction for phase
                adjustment = 0.25 * std_dev * direction_mult
                adjusted_mean = mean_val + adjustment
    
                if phase_idx < len(phase_names) - 1:
                    line_start = phase_lines[phase_idx] + 0.5
                    line_end = phase_lines[phase_idx + 1] + 0.5 if phase_idx + 1 < len(phase_lines) else max(sessions)
    
                    # Adjusted level line (mean)
                    ax.hlines(adjusted_mean, line_start, line_end, colors='blue', linestyles='dashed', linewidth=2)
                    
                    # Calculate and draw trend line separately
                    slope, intercept = None, None
                    if len(phase_sessions) > 1:
                        # Calculate basic trend line
                        slope, intercept = np.polyfit(phase_sessions, phase_data, 1)
                        
                        # Apply the same direction-based adjustment to the trend line
                        adjusted_intercept = intercept + adjustment
                        
                        # Generate trend line points
                        trend_x = np.linspace(line_start, line_end, 100)
                        trend_y = slope * trend_x + adjusted_intercept
                        
                        # Plot the adjusted trend line
                        ax.plot(trend_x, trend_y, color='red', linestyle='dotted', linewidth=2)
    
                # Store phase analytics with direction-specific adjustments
                analytics = {
                    'raw_mean': mean_val,
                    'adjusted_mean': adjusted_mean,
                    'sessions': phase_sessions,
                    'data': phase_data,
                    'slope': slope,
                    'intercept': intercept,  # Store raw intercept
                    'adjusted_intercept': adjusted_intercept if slope is not None else None,
                    'direction': direction,
                    'adjustment': adjustment,
                    'total_points': len(phase_data)
                }
                phase_analytics.append(analytics)
                        
        for phase_idx in range(1, len(phase_analytics)):
            prev = phase_analytics[phase_idx-1]
            current = phase_analytics[phase_idx]
            
            # Separate counts for level and trend comparisons
            below_level_count = 0
            below_trend_count = 0
            below_both_count = 0
            above_level_count = 0
            above_trend_count = 0
            above_both_count = 0
            total_points = current['total_points']
            phase_pair = f"{phase_names[phase_idx-1]} and {phase_names[phase_idx]}"
        
            # Compare against both separately for better clarity
            for x, y in zip(current['sessions'], current['data']):
                # Check against previous phase's level line (mean) - using adjusted mean
                if y < prev['adjusted_mean']:
                    below_level_count += 1
                else:
                    above_level_count += 1
                    
                # Check against previous phase's trend line - using adjusted trend
                if prev['slope'] is not None:
                    prev_trend = prev['slope'] * x + prev['adjusted_intercept']
                    if y < prev_trend:
                        below_trend_count += 1
                    else:
                        above_trend_count += 1
                        
                    # Track points above/below both lines
                    if y < prev['adjusted_mean'] and y < prev_trend:
                        below_both_count += 1
                    if y > prev['adjusted_mean'] and y > prev_trend:
                        above_both_count += 1
        
            # Direct requirement list lookup based on point count
            req_index = total_points - 3  # First requirement (3) applies to 3 data points
            
            # Fix: Use array indexing (not function call) for predicted_directions
            if predicted_directions[phase_idx-1] == "+":
                # Direction is positive, check points ABOVE both lines
                if 0 <= req_index < len(self.requirement_list):
                    required = self.requirement_list[req_index]
                
                    self.log_message(f"Phase comparison: {phase_pair}")
                    self.log_message(f"  Points above both lines: {above_both_count}")
                
                    # Use above_both_count for final determination with positive direction
                    if above_both_count >= required:
                        self.log_message(f"✓ Systematic difference detected between {phase_pair}")
                    else:
                        self.log_message(f"✗ No systematic difference between {phase_pair}")
                else:
                    self.log_message(f"Undefined requirement for {phase_pair}: "
                                     f"{total_points} data points")
                    
            elif predicted_directions[phase_idx-1] == "-":
                # Direction is negative, check points BELOW both lines
                if 0 <= req_index < len(self.requirement_list):
                    required = self.requirement_list[req_index]
                
                    self.log_message(f"Phase comparison: {phase_pair}")
                    self.log_message(f"  Points below both lines: {below_both_count}")
                
                    # Use below_both_count for final determination with negative direction
                    if below_both_count >= required:
                        self.log_message(f"✓ Systematic difference detected between {phase_pair}")
                    else:
                        self.log_message(f"✗ No systematic difference between {phase_pair}")
                    self.log_message(f"Undefined requirement for {phase_pair}: "
                                     f"{total_points} data points")
    
        # Add vertical phase lines and labels
        for line_pos in phase_lines:
            ax.axvline(x=line_pos + 0.5, color='black', linestyle='--', linewidth=1)
    
        for i in range(len(phase_names)):
            start = min(sessions) if i == 0 else phase_lines[i - 1] + 1
            end = max(sessions) if i == len(phase_names) - 1 else phase_lines[i]
            mid_point = (start + end) / 2
            label_text = f"{phase_names[i]}"
            ax.text(mid_point, self.y_max * 0.9, label_text,
                    horizontalalignment='center', verticalalignment='bottom',
                    fontsize=14)
    
        ax.set_xlabel('Session Number', fontsize=18)
        ax.set_ylabel(self.entries[2].get(), fontsize=18)
        plt.suptitle(self.entries[0].get(), fontsize=20)
        plt.title(self.entries[1].get(), fontsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, self.y_max)
        
        # Display graph
        graph_window = tk.Toplevel(self)
        graph_window.title("Graph")
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
    
        # Save button
        save_button = ctk.CTkButton(graph_window, text="Save Graph",
                                    command=lambda: self.save_graph(fig),
                                    height=40, font=("Arial", 14, "bold"),
                                    corner_radius=8, fg_color="#400287",
                                    hover_color="#600387")
        save_button.pack(pady=10)
    
        canvas.draw()
        plt.tight_layout()
    
        y_max = max(max(float(row[i]) for row in data if row[i])
                    for i in range(1, len(phase_names) + 1))
        ax.set_ylim(bottom=0, top=y_max * 1.1)
    
        plt.show()
        
    def on_clear_all(self):
        # Ensure `self.entries` exists and is a list
        if not hasattr(self, "entries") or not isinstance(self.entries, list):
            return  # Prevents accidental errors

        # Ensure that input fields are still linked in `self.entries`
        if len(self.entries) < 6:
            self.log_message("Error: Input fields are missing. Resetting...")
            self.entries = [None] * 6  # Reset the list with placeholders

        # Clear input fields (Graph Title, Subtitle, etc.), but keep them in `self.entries`
        for i in range(6):
            if isinstance(self.entries[i], ctk.CTkEntry):
                self.entries[i].delete(0, 'end')

        # Ensure Phases and Sessions fields exist and are repopulated with default values
        if isinstance(self.entries[4], ctk.CTkEntry):
            self.entries[4].insert(0, "")  # Default to 1 phase
        if isinstance(self.entries[5], ctk.CTkEntry):
            self.entries[5].insert(0, "")  # Default to 1 session

        # Clear spreadsheet entries list
        self.spreadsheet_entries = []

        # Clear the spreadsheet frame
        for widget in self.spreadsheet_frame.winfo_children():
            widget.destroy()

        # Clear output console
        self.output_text.configure(state="normal")
        self.output_text.delete(1.0, 'end')
        self.output_text.configure(state="disabled")

        # Reset scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def paste_from_clipboard(self, event, row, col):
        try:
            # Get raw data from the clipboard
            clipboard_data = self.clipboard_get()
            
            # Check if it's multi-cell content (contains tabs or newlines)
            if '\t' in clipboard_data or '\n' in clipboard_data:
                # Normalize line endings
                clipboard_data = clipboard_data.replace('\r\n', '\n')
                
                # Split into rows
                rows = clipboard_data.split('\n')
                
                # Remove any trailing empty rows
                while rows and not rows[-1]:
                    rows.pop()
                
                # Process each row
                for r_offset, row_data in enumerate(rows):
                    # Split the row into cells
                    cols = row_data.split('\t')
                    
                    # Process each cell
                    for c_offset, value in enumerate(cols):
                        target_row = row + r_offset
                        target_col = col + c_offset
                        
                        # Make sure we're within grid bounds
                        if 0 <= target_row < len(self.spreadsheet_entries) and 0 <= target_col < len(self.spreadsheet_entries[0]):
                            # Get the target cell widget
                            cell_widget = self.spreadsheet_entries[target_row][target_col]
                            
                            if isinstance(cell_widget, ctk.CTkEntry):
                                # Clear existing content and insert new value
                                cell_widget.delete(0, 'end')
                                cell_widget.insert(0, value)
                                
                                # Immediately update the cell to ensure changes are visible
                                cell_widget.update()
                
                # Return early to prevent the default paste behavior
                return "break"
            
            # If it's not multi-cell content, let the default paste behavior handle it
            return None
            
        except Exception as e:
            error_msg = f"Paste error: {str(e)}"
            self.log_message(error_msg)
            print(f"Error details: {repr(e)}")
            return "break"  # Prevent default paste behavior on error
         
    def get_entered_values(self):
        entered_values = []
        phase_names = []
        predicted_directions = []
        all_numerical_values = []  # To store all numerical values for finding the maximum
        
        for i, row in enumerate(self.spreadsheet_entries):
            row_values = []
            for j, cell in enumerate(row):
                if i == 0:  # Header row
                    if j == 0:
                        continue  # Skip session header
                    if isinstance(cell, tuple):
                        phase_names.append(cell[0].get())
                        predicted_directions.append(cell[1].get())
                    else:
                        phase_names.append(cell.get())
                else:
                    if j == 0:  # Session number
                        row_values.append(cell.get())
                    else:  # Data values
                        value = cell.get()
                        row_values.append(value)
                        
                        # Try to convert to float and add to our list of numerical values
                        if value.strip():  # Skip empty cells
                            try:
                                numerical_value = float(value)
                                all_numerical_values.append(numerical_value)
                            except ValueError:
                                # Not a valid number, skip it
                                pass
            
            if i > 0:
                entered_values.append(row_values)
        
        # Set the maximum y value if we have numerical data
        if all_numerical_values:
            self.y_max = max(all_numerical_values) * 1.1
        else:
            self.y_max = 10  # Default if no valid data
        
        return phase_names, entered_values, predicted_directions
    
    def calculate_phase_lines(self):
        phase_names, data, predicted_directions = self.get_entered_values()
        phase_lines = []
        current_phase = 0
        
        for i, row in enumerate(data):
            for j in range(1, len(row)):
                if row[j] and current_phase != j:
                    if i > 0:  # Don't add a line before the first data point
                        phase_lines.append(i)
                    current_phase = j
                    break
    
        return phase_lines
    
    def save_graph(self, fig):
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"),
                                                            ("All files", "*.*")])
        if file_path:
            fig.savefig(file_path)
            self.log_message(f"Graph saved successfully to {file_path}")
        
if __name__ == "__main__":
    app = App()
    app.mainloop()