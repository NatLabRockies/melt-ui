import logging
import os
import signal
import sys

import meltui.backend  # noqa: F401
from flask import send_from_directory
from meltui.app import app


# Signal handler for SIGINT
def signal_handler(sig, frame):
    logging.info("Received SIGINT, exiting.")
    sys.exit(0)


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(app.static_folder + "/" + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    print("Starting server at http://127.0.0.1:8080")
    app.run(
        debug=True,
        port=8080,
        use_reloader=True,
        use_debugger=True,
        use_evalex=True,
    )
