"""Wait until the demo server answers, then open the demo page in the default browser.
Used by start_demo.bat; gives up after two minutes."""
import sys
import time
import urllib.request
import webbrowser

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

for _ in range(240):
    try:
        urllib.request.urlopen(URL + "/health", timeout=1)
        webbrowser.open(URL)
        break
    except Exception:
        time.sleep(0.5)
