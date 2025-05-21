"""
Styles module for the Trade Calculator application.
Contains UI theme definitions and styling constants.
"""

import FreeSimpleGUI as sg

# Color constants
BLACK = '#000000'
WHITE = '#FFFFFF'
DARK_GRAY = '#1C1C1C'
BLUE = '#0D47A1'
AMBER = '#FFBF00'
RED = '#800000'
GREEN = '#00FF00'
LIGHT_BLUE = '#00BFFF'

# Font defaults
DEFAULT_FONT = 'Helvetica'
BOLD_FONT = ('Helvetica', 12, 'bold')
TITLE_FONT = ('Helvetica', 16)
SMALL_FONT = ('Helvetica', 8)

def setup_bloomberg_theme():
    """Set up and apply the Bloomberg-like theme for the application"""
    # Define Bloomberg-like theme colors
    sg.theme_background_color(BLACK)
    sg.theme_text_color(WHITE)
    sg.theme_input_background_color(DARK_GRAY)
    sg.theme_input_text_color(WHITE)
    sg.theme_button_color((WHITE, BLUE))  # Text color, button color
    sg.theme_element_background_color(BLACK)
    sg.theme_element_text_color(WHITE)
    sg.theme_progress_bar_color((AMBER, DARK_GRAY))  # Bar color, background

    # Bloomberg-like theme definition
    bloomberg_theme = {
        'BACKGROUND': BLACK,
        'TEXT': WHITE,
        'INPUT': DARK_GRAY,
        'TEXT_INPUT': WHITE,
        'SCROLL': BLUE,
        'BUTTON': (WHITE, BLUE),
        'PROGRESS': (AMBER, DARK_GRAY),
        'BORDER': 1,
        'SLIDER_DEPTH': 0,
        'PROGRESS_DEPTH': 0,
    }

    # Register the custom theme
    sg.theme_add_new('Bloomberg', bloomberg_theme)
    sg.theme('Bloomberg')

# UI element style functions
def title_text(text):
    """Create a styled title text element"""
    return sg.Text(text, font=TITLE_FONT, text_color=AMBER)

def label_text(text):
    """Create a styled label text element"""
    return sg.Text(text, text_color=AMBER)

def bold_label(text):
    """Create a styled bold label text element"""
    return sg.Text(text, font=BOLD_FONT, text_color=AMBER)

def input_field(default="", key=None, size=(15, 1)):
    """Create a styled input field"""
    return sg.Input(default, key=key, size=size, background_color=DARK_GRAY)

def primary_button(text, key=None, bind_return_key=False):
    """Create a styled primary button"""
    return sg.Button(text, key=key, button_color=(WHITE, BLUE), bind_return_key=bind_return_key)

def secondary_button(text, key=None):
    """Create a styled secondary button"""
    return sg.Button(text, key=key, button_color=(WHITE, RED))

def status_text(size=(50, 1), key=None):
    """Create a status text field"""
    return sg.Text("", size=size, key=key)

def create_progress_bar(max_value=100, size=(20, 10), key=None):
    """Create a styled progress bar"""
    return sg.ProgressBar(max_value, orientation='h', size=size, key=key)

def window_params():
    """Common window parameters"""
    return {
        'background_color': BLACK,
        'finalize': True
    }

def table_params(values, headings, num_rows=25):
    """Common parameters for tables"""
    return {
        'values': values,
        'headings': headings,
        'auto_size_columns': False,
        'background_color': BLACK,
        'text_color': WHITE,
        'header_text_color': AMBER,
        'header_background_color': BLUE,
        'justification': 'right',
        'num_rows': num_rows,
        'alternating_row_color': DARK_GRAY
    } 