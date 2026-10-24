#!/usr/bin/env python3
"""Open the self-contained WebGPU lab on localhost. Ctrl-C stops the server."""
import functools, http.server, pathlib, webbrowser
root = pathlib.Path(__file__).resolve().parent
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
for port in range(8778, 8788):
    try:
        server = http.server.ThreadingHTTPServer(('127.0.0.1', port), handler)
        break
    except OSError:
        continue
else:
    raise SystemExit('Ports 8778–8787 are occupied. Free a port and retry.')
url = f'http://127.0.0.1:{port}/webgpu.html'
print(f'WebGPU lab: {url}\nCtrl-C stops this local-only server.', flush=True)
webbrowser.open(url)
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    server.server_close()
