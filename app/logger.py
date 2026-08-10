from datetime import datetime

from threading import Lock

logs = []
lock = Lock()

def log(message, level="info"):

    with lock:

        logs.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "message": message
        })


def get_logs():

    global logs

    with lock:

        current = logs.copy()

        logs.clear()

    return current