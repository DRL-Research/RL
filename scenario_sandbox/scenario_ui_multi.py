"""Compatibility wrapper for `one_element_scenario_ui.py`."""

from one_element_scenario_ui import *  # noqa: F401,F403
from one_element_scenario_ui import ScenarioBuilderApp
import tkinter as tk


if __name__ == "__main__":
    root = tk.Tk()
    app = ScenarioBuilderApp(root)
    root.mainloop()
