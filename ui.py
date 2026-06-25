import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from app.market_data import fetch_market_inputs
from app.orchestrator import SimpleOrchestrator

HOST = "127.0.0.1"
PORT = 8080

HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MACE Market Analyzer UI</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; background: #f8fafc; color: #0f172a; }
    main { max-width: 850px; margin: auto; background: white; padding: 2rem; border-radius: 16px; box-shadow: 0 10px 30px #0001; }
    button { margin-right: 1rem; padding: 0.8rem 1rem; border: 0; border-radius: 10px; cursor: pointer; font-weight: 700; }
    #generate { background: #2563eb; color: white; }
    #retrieve { background: #f59e0b; color: #111827; }
    pre { background: #0f172a; color: #e2e8f0; padding: 1rem; border-radius: 12px; overflow-x: auto; }
  </style>
</head>
<body>
  <main>
    <h1>MACE Gold/Silver Market Analyzer</h1>
    <p>Click <strong>Generate result</strong> to automatically fetch latest real-world gold/silver inputs, run the agents, and store context in <code>store.json</code>.</p>
    <button id="generate" onclick="generateResult()">Generate result</button>
    <button id="retrieve" onclick="retrieveNews()">Retrieve</button>
    <p><small>The Retrieve button shows the "news" string.</small></p>
    <h2>Output</h2>
    <pre id="output">Waiting for action...</pre>
  </main>
  <script>
    async function generateResult() {
      const response = await fetch('/api/generate', { method: 'POST' });
      const data = await response.json();
      document.getElementById('output').textContent = JSON.stringify(data, null, 2);
    }
    async function retrieveNews() {
      const response = await fetch('/api/retrieve');
      const data = await response.text();
      document.getElementById('output').textContent = data;
    }
  </script>
</body>
</html>
"""


class MarketAnalyzerUIHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/":
            self._send_html(HTML)
            return
        if self.path == "/api/retrieve":
            self._send_text("news")
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        if self.path == "/api/generate":
            try:
                market_input = fetch_market_inputs()
                context = SimpleOrchestrator().run(market_input)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=502)
                return
            self._send_json(context)
            return
        self.send_error(404, "Not found")

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_html(self, body: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def _send_text(self, body: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def _send_json(self, body: dict, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(body, indent=2).encode("utf-8"))


def run_ui(host: str = HOST, port: int = PORT) -> None:
    server = HTTPServer((host, port), MarketAnalyzerUIHandler)
    print(f"MACE UI running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_ui()
