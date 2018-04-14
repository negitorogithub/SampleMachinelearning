import os

import time
import os
import random

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

BASEDIR = "F:\Images\Vegetables\GreenPepper"


def getext(filename):
    return os.path.splitext(filename)[-1].lower()


class ChangeHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if getext(event.src_path) in ('.jpg', '.png', '.txt'):
            renamed = os.path.splitext(event.src_path)[0] + str(int(time.time())) + str(random.randint(1,10000)) + os.path.splitext(event.src_path)[-1]
            os.rename(event.src_path, renamed)
            print(renamed)

if __name__ in '__main__':
    while 1:
        event_handler = ChangeHandler()
        observer = Observer()
        observer.schedule(event_handler, BASEDIR)
        observer.start()
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            observer.stop()
            quit(1001)
        observer.join()
