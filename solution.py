#!/usr/bin/env python3
"""Run the modular earthquake inspection assignment: python3 solution.py."""
import io
import multiprocessing as mp
import sys
from contextlib import redirect_stdout

from earthquake_lab.config import OUT
from earthquake_lab.pipeline import run


class Tee:
    """Copy the actual console output into the submitted run log."""
    def __init__(self, console, buffer):
        self.console = console
        self.buffer = buffer

    def write(self, text):
        self.console.write(text)
        return self.buffer.write(text)

    def flush(self):
        self.console.flush()


def main():
    buffer = io.StringIO()
    with redirect_stdout(Tee(sys.stdout, buffer)):
        run()
    (OUT / 'output.txt').write_text(buffer.getvalue())


if __name__ == '__main__':
    mp.freeze_support()
    main()
