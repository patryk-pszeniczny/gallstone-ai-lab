import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class BaseTab:
    def __init__(self, app, tab):
        self.app = app
        self.state = app.state
        self.root = app
        self.tab = tab

    def info(self, title, text):
        messagebox.showinfo(title, text, parent=self.root)

    def warn(self, title, text):
        messagebox.showwarning(title, text, parent=self.root)

    def error(self, title, text):
        messagebox.showerror(title, text, parent=self.root)

    def draw_figure(self, fig, plot_area, toolbar_area):
        # czyść
        for child in plot_area.winfo_children():
            child.destroy()
        for child in toolbar_area.winfo_children():
            child.destroy()
        canvas = FigureCanvasTkAgg(fig, master=plot_area)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_area)
        toolbar.update()
