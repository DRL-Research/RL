"""Compatibility wrapper for `composer_scenario_ui.py`."""

from composer_scenario_ui import *  # noqa: F401,F403
from composer_scenario_ui import ComposerScenarioBuilderApp
import tkinter as tk


if __name__ == "__main__":
    root = tk.Tk()
    app = ComposerScenarioBuilderApp(root)
    root.mainloop()
