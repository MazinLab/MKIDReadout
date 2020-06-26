from __future__ import division
import os
import signal
import logging
import argparse
import threading
import numpy as np
import multiprocessing as mp
import tkinter as tk
from tkinter import ttk, font, filedialog

from mkidcore.config import yaml
from mkidreadout.configuration.optimalfilters import filters, make_filters

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

MIN_HEIGHT = 100
MIN_WIDTH = 500
START_HEIGHT = 200
START_WIDTH = 865
MAX_SUBDIRECTORIES = 25
DEFAULT_CONFIG = os.path.join(os.path.abspath(os.path.dirname(__file__)), "filter.yml")
DEFAULT_FILTER = filters.__all__[1]


class ConfigurationTab(tk.Frame):
    def __init__(self, *args, **kwargs):
        # optional kwargs
        self.config_file = kwargs.pop("config_file", DEFAULT_CONFIG)

        tk.Frame.__init__(self, *args, **kwargs)
        self._setup_ui()
        self._layout()
        log.info("ConfigurationTab built.")

    def _setup_ui(self):
        self.hscrollbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL)
        self.vscrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL)
        self.text = tk.Text(self, yscrollcommand=self.vscrollbar.set, xscrollcommand=self.hscrollbar.set, wrap=tk.NONE)
        self.text.bind('<Control-v>', lambda *args: delete_selection(self.text))
        self.hscrollbar.config(command=self.text.xview)
        self.vscrollbar.config(command=self.text.yview)
        with open(self.config_file, "r") as f:
            text = f.read()
            self.text.insert(tk.END, text)

    def _layout(self):
        self.hscrollbar.grid(row=1, column=0, sticky=tk.EW)
        self.vscrollbar.grid(row=0, column=1, sticky=tk.NS)
        self.text.grid(row=0, column=0, sticky=tk.NSEW)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)


class CalculationRow(tk.Frame):
    def __init__(self, *args, **kwargs):
        # optional kwargs
        self.directory = kwargs.pop("directory", None)
        self.all = kwargs.pop("all", False)
        self.config_callback = kwargs.pop("config_callback", None)

        # attributes
        self.process = None  # calculation process
        self.process_cleanup = None  # thread for cleaning up gui state after process is done
        self.progress_thread = None  # thread for updating the progress bar
        self.progress_queue = None  # queue for passing progress information to the progress thread

        tk.Frame.__init__(self, *args, **kwargs)
        self._setup_ui()
        self._layout()
        log.info("CalculationRow built.")

    def _setup_ui(self):
        self.label = tk.Label(self, text="All Directories:" if self.all else os.path.basename(self.directory) + ":")
        current_font = font.Font(font=self.label['font'])
        current_font.configure(weight=font.BOLD)
        self.label['font'] = current_font
        self.force = tk.IntVar(self)
        self.force.set(0)
        self.force_check = tk.Checkbutton(self, text="Force", variable=self.force)
        self.filter_ = tk.StringVar(self)
        self.filter_options = ttk.OptionMenu(self, self.filter_, DEFAULT_FILTER, *filters.__all__)
        self.filter_options.config(width=max([len(name) for name in filters.__all__]))
        self.start = tk.Button(self, text="Start", command=self.master.run if self.all else self.run)
        self.stop = tk.Button(self, text="Stop", command=self.master.abort if self.all else self.abort)
        if not self.all:
            self.stop.config(state=tk.DISABLED)
        self.progress = ttk.Progressbar(self, orient=tk.HORIZONTAL, mode='determinate')
        self.plot = tk.Button(self, text="Plot", command=self.plot)

    def _layout(self):
        self.label.pack(side=tk.LEFT, fill=tk.X)
        self.force_check.pack(side=tk.LEFT, fill=tk.X)
        self.filter_options.pack(side=tk.LEFT, fill=tk.X)
        self.start.pack(side=tk.LEFT, fill=tk.X)
        self.stop.pack(side=tk.LEFT, fill=tk.X)
        if not self.all:
            self.plot.pack(side=tk.RIGHT, fill=tk.X)
            self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=tk.TRUE)

    def run(self):
        self.start.config(state=tk.DISABLED)
        # get the config file
        config = self.config_callback()
        config.paths.register('data', self.directory, update=True)
        config.paths.register('out', self.directory, update=True)
        config.filters.filter.register('filter_type', self.filter_.get(), update=True)

        # set up the process and threads
        force = self.force.get()
        self.progress_queue = mp.Queue()

        def setup(n):
            self.progress_queue.put({"maximum": n + 1, "value": 0})

        def callback(*args):
            self.progress_queue.put({"step": True})

        self.process = mp.Process(target=make_filters.run, args=[config],
                                  kwargs={"force": force, "progress_setup": setup, "progress_callback": callback})
        self.progress_thread = threading.Thread(target=self.update_progress)
        self.process_cleanup = threading.Thread(target=self.finish_process)

        # make all threads close on gui close
        self.progress_thread.daemon = True
        self.process_cleanup.daemon = True

        # start the process and threads
        self.process.start()
        self.progress_thread.start()  # update progress bar
        self.process_cleanup.start()  # return button states after progress is done
        self.stop.config(state=tk.NORMAL)
        log.info("{}: calculation process started".format(os.path.basename(self.directory)))

    def abort(self):
        log.info("{}: aborting calculation process".format(os.path.basename(self.directory)))
        self.stop.config(state=tk.DISABLED)  # can't stop twice
        os.kill(self.process.pid, signal.SIGINT)  # send keyboard interrupt to process

    def update_progress(self):
        while True:
            # wait for the next command
            kwargs = self.progress_queue.get()
            # change the maximum of the progress bar
            maximum = kwargs.get("maximum", False)
            if maximum:
                self.progress['maximum'] = int(maximum)
            # step the progress bar
            step = kwargs.get("step", False)
            if step:
                self.progress.step()
            # change the progress bar value
            value = kwargs.get("value", False)
            if value is not False:
                self.progress['value'] = value
            # stop the progress bar loop
            stop = kwargs.get("stop", False)
            if stop:
                break
        self.start.config(state=tk.NORMAL)  # return the start button to normal when thread is finished

    def finish_process(self):
        self.process.join()  # wait for process to finish
        if self.stop['state'] == tk.DISABLED:
            value = 0  # process was aborted
        else:
            value = self.progress['maximum']  # process finished
            self.stop.config(state=tk.DISABLED)
        self.progress_queue.put({"value": value, "stop": True})
        log.info("{}: calculation process finished".format(os.path.basename(self.directory)))

    def plot(self):
        file_name = os.path.join(self.directory, make_filters.DEFAULT_SAVE_NAME)
        name = os.path.basename(self.directory)
        if os.path.isfile(file_name):
            solution = make_filters.Solution.load(file_name)
            process = mp.Process(target=solution.plot, kwargs={"filter_type": self.filter_.get(), "report": True})
            process.daemon = True
            process.start()
            log.info("{}: plot process started".format(name))
        else:
            log.error("{}: could not start plot process because '{}' does not exist".format(name, file_name))


class DirectoryRow(tk.Frame):
    def __init__(self, *args, **kwargs):
        # optional kwargs
        self.directory_callback = kwargs.pop("directory_callback", None)

        tk.Frame.__init__(self, *args, **kwargs)
        self._setup_ui()
        self._layout()
        log.info("DirectoryRow built.")

    def _setup_ui(self):
        self.button = tk.Button(self, text="Choose Directory", command=self.get_directory)
        self.directory = tk.StringVar(self)
        self.entry = tk.Entry(self, textvariable=self.directory)
        self.entry.bind('<Return>', lambda *args: self.set_directory(self.directory.get()))
        self.entry.bind('<Control-v>', lambda *args: delete_selection(self.entry))

    def _layout(self):
        self.button.pack(side=tk.LEFT, fill=tk.X)
        self.entry.pack(side=tk.RIGHT, fill=tk.X, expand=tk.TRUE)

    def get_directory(self):
        directory = filedialog.askdirectory(initialdir=self.directory.get())
        if directory:
            self.set_directory(directory)

    def set_directory(self, directory):
        if os.path.isdir(directory):
            self.directory.set(directory)
            if self.directory_callback is not None:
                self.directory_callback(directory)
            log.info("Changed main directory to {}".format(directory))


class CalculationTab(tk.Frame):
    def __init__(self, *args, **kwargs):
        # optional kwargs
        self.resize_callback = kwargs.pop("resize_callback", None)
        self.config_callback = kwargs.pop("config_callback", None)

        # attributes
        self.calculation_rows = []

        tk.Frame.__init__(self, *args, **kwargs)
        self._setup_ui()
        self._layout()
        log.info("CalculationTab built.")

    def _setup_ui(self):
        self.directory_row = DirectoryRow(self, directory_callback=self.reset_rows)
        self.change_all = CalculationRow(self, all=True, config_callback=self.config_callback)
        self.change_all.force.trace("w", lambda *args: self.force_all())
        self.change_all.filter_.trace("w", lambda *args: self.filter_all())

    def _layout(self):
        self.directory_row.pack(side="top", fill=tk.X)
        self.change_all.pack(side="bottom", fill=tk.X)

    def reset_rows(self, directory):
        # quick check so that the GUI doesn't explode if the wrong directory is chosen
        subdirectories = [os.path.join(directory, d) for d in os.listdir(directory)
                          if os.path.isdir(os.path.join(directory, d))]
        if len(subdirectories) > MAX_SUBDIRECTORIES:
            raise ValueError("Too many subdirectories in the chosen folder")

        # delete the old rows
        for row in self.calculation_rows:
            row.destroy()
        self.calculation_rows = []

        # make the new rows
        for subdirectory in subdirectories:
            self.calculation_rows.append(CalculationRow(self, directory=subdirectory,
                                                        config_callback=self.config_callback))
            self.calculation_rows[-1].pack(side="top", fill=tk.X)

        if self.resize_callback is not None:
            self.resize_callback()

    def run(self):
        for calculation in self.calculation_rows:
            if not calculation.start['state'] == tk.DISABLED:
                calculation.run()

    def abort(self):
        for calculation in self.calculation_rows:
            if not calculation.stop['state'] == tk.DISABLED:
                calculation.abort()

    def force_all(self):
        for calculation in self.calculation_rows:
            calculation.force.set(self.change_all.force.get())

    def filter_all(self):
        for calculation in self.calculation_rows:
            calculation.filter_.set(self.change_all.filter_.get())


class MainWindow(ttk.Notebook):
    def __init__(self, *args, **kwargs):
        # optional kwargs
        self.resize_callback = kwargs.pop("resize_callback", None)
        self.configuration = kwargs.pop("configuration", None)

        ttk.Notebook.__init__(self, *args, **kwargs)
        self._setup_ui()
        self._layout()
        log.info("MainWindow built.")

    def _setup_ui(self):
        self.calculation_tab = CalculationTab(self, resize_callback=self.resize_callback,
                                              config_callback=self.get_config)
        self.configuration_tab = ConfigurationTab(self, config_file=self.configuration)

    def _layout(self):
        self.add(self.calculation_tab, text="calculation")
        self.add(self.configuration_tab, text="configuration")

    def get_config(self):
        return yaml.load(self.configuration_tab.text.get("1.0", tk.END))


def delete_selection(widget):
    try:
        widget.delete("sel.first", "sel.last")
    except tk.TclError:
        pass


def resize(root):
    root.update()  # ensure geometry info is up to date
    width = root.winfo_width()  # get current width
    height = root.winfo_height()  # get current width
    root.geometry("")  # resize gui to the minimum size
    root.minsize(height=height, width=width)  # set the geometry to be no smaller than it was
    root.update()  # ensure geometry info is up to date
    root.geometry(root.geometry())  # resize without changing the dimensions (so recalling minsize won't shrink the gui)
    root.minsize(height=MIN_HEIGHT, width=MIN_WIDTH)  # restore the original minsize
    log.info("Resized GUI.")


if __name__ == "__main__":
    # parse the command line arguments
    parser = argparse.ArgumentParser(description='Filter Computation GUI')
    parser.add_argument('configuration', type=str, default=DEFAULT_CONFIG, nargs="?",
                        help='The path to the configuration file to use for the computation. If not specified, the '
                             'default will be used. The configuration values can be changed at any time directly in '
                             'the GUI.')
    parser.add_argument('-l', '--log', type=str, dest='level', default="INFO",
                        help='The logging level to display.')
    a = parser.parse_args()

    # set up logging
    logging.basicConfig(level=a.level)

    # set up the GUI
    app = tk.Tk()
    app.title("Filter GUI")
    app.minsize(height=MIN_HEIGHT, width=MIN_WIDTH)
    screen_width = app.winfo_screenwidth()
    screen_height = app.winfo_screenheight()
    left = screen_width / 4 - START_WIDTH / 2
    top = screen_height / 2 - START_HEIGHT / 2
    app.geometry('%dx%d+%d+%d' % (START_WIDTH, START_HEIGHT, left, top))

    window = MainWindow(app, configuration=a.configuration, resize_callback=lambda: resize(app))
    window.pack(fill=tk.BOTH, expand=tk.TRUE)

    # set up a call to abort all processes on closing the window
    def on_closing():
        window.calculation_tab.abort()
        app.destroy()
    app.protocol("WM_DELETE_WINDOW", on_closing)

    app.mainloop()
