"""Side-by-side demo of one decode step.

Each token is one attention call at Llama 3 8B width. The page times both
paths on this GPU, then plays the race slowly enough to watch. The microseconds
under the bars are the real measurement, not the slowed animation.

    python -m decode_attn.demo
"""

from __future__ import annotations

import json
import statistics
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
import torch.nn.functional as F

from decode_attn import decode_attention

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>One new word</title>
<style>
  :root { color-scheme: dark; }
  body {
    margin: 0; min-height: 100vh; background: #0c0d10; color: #f4f1ea;
    font-family: Georgia, "Iowan Old Style", serif;
    display: flex; flex-direction: column; align-items: center; padding: 48px 20px;
  }
  h1 { font-weight: 500; font-size: 42px; margin: 0 0 8px; letter-spacing: -0.03em; }
  p.lead { max-width: 640px; text-align: center; color: #b7b1a6; margin: 0 0 28px; line-height: 1.45; }
  .row { display: flex; gap: 10px; margin-bottom: 22px; }
  button {
    background: #1a1c22; color: #f4f1ea; border: 1px solid #343842;
    border-radius: 999px; padding: 10px 16px; font: inherit; cursor: pointer;
  }
  button.on { background: #f4f1ea; color: #0c0d10; border-color: #f4f1ea; }
  button.go { background: #e85d4c; border-color: #e85d4c; color: white; padding: 10px 22px; }
  .lanes { width: min(880px, 100%); display: grid; gap: 18px; }
  .lane { background: #14161c; border-radius: 18px; padding: 18px 18px 16px; }
  .top { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
  .name { font-size: 22px; }
  .us { font-variant-numeric: tabular-nums; font-family: ui-monospace, monospace; color: #d8d2c8; }
  .track { margin-top: 14px; height: 18px; background: #0c0d10; border-radius: 999px; overflow: hidden; }
  .fill { height: 100%; width: 0%; border-radius: 999px; }
  .ours .fill { background: #e85d4c; }
  .torch .fill { background: #8d93a3; }
  .sentence { margin-top: 12px; min-height: 1.4em; font-size: 20px; }
  .foot { margin-top: 22px; color: #8d877c; font-size: 14px; max-width: 640px; text-align: center; }
</style>
</head>
<body>
  <h1>One new word</h1>
  <p class="lead">Same graphics card. Same sentence. Each word is one real attention step. The clock under each bar was just measured on this PC.</p>
  <div class="row" id="lengths"></div>
  <div class="lanes">
    <div class="lane ours">
      <div class="top"><div class="name">This kernel</div><div class="us" id="oursUs">—</div></div>
      <div class="track"><div class="fill" id="oursFill"></div></div>
      <div class="sentence" id="oursText"></div>
    </div>
    <div class="lane torch">
      <div class="top"><div class="name">Normal PyTorch</div><div class="us" id="torchUs">—</div></div>
      <div class="track"><div class="fill" id="torchFill"></div></div>
      <div class="sentence" id="torchText"></div>
    </div>
  </div>
  <p class="foot" id="note">Pick a cache length and press start. The animation is slowed so the gap is visible. The microseconds are not.</p>
<script>
const sentence = "The quick brown fox jumps over the lazy dog and keeps going";
const words = sentence.split(" ");
const lengths = [1024, 4096, 16384];
let chosen = 4096;
let playing = false;
const box = document.getElementById("lengths");
function paintLengths() {
  box.innerHTML = "";
  for (const n of lengths) {
    const b = document.createElement("button");
    b.textContent = n.toLocaleString() + " tokens";
    b.className = n === chosen ? "on" : "";
    b.onclick = () => { chosen = n; paintLengths(); };
    box.appendChild(b);
  }
  const go = document.createElement("button");
  go.textContent = playing ? "Running" : "Start";
  go.className = "go";
  go.disabled = playing;
  go.onclick = run;
  box.appendChild(go);
}
paintLengths();
function setBar(id, frac) {
  document.getElementById(id).style.width = Math.max(0, Math.min(1, frac)) * 100 + "%";
}
function wordsAt(frac) {
  const n = Math.min(words.length, Math.floor(frac * words.length));
  return words.slice(0, n).join(" ");
}
async function run() {
  playing = true;
  paintLengths();
  document.getElementById("oursText").textContent = "";
  document.getElementById("torchText").textContent = "";
  setBar("oursFill", 0); setBar("torchFill", 0);
  document.getElementById("note").textContent = "Timing both paths on the GPU…";
  const res = await fetch("/measure?s=" + chosen);
  const data = await res.json();
  document.getElementById("oursUs").textContent = Math.round(data.ours_us) + " µs / token";
  document.getElementById("torchUs").textContent = Math.round(data.sdpa_us) + " µs / token";
  const slowMs = 7000;
  const oursMs = slowMs * (data.ours_us / data.sdpa_us);
  const t0 = performance.now();
  function frame(now) {
    const o = Math.min(1, (now - t0) / oursMs);
    const p = Math.min(1, (now - t0) / slowMs);
    setBar("oursFill", o); setBar("torchFill", p);
    document.getElementById("oursText").textContent = wordsAt(o);
    document.getElementById("torchText").textContent = wordsAt(p);
    if (o < 1 || p < 1) requestAnimationFrame(frame);
    else {
      playing = false;
      paintLengths();
      const x = (data.sdpa_us / data.ours_us).toFixed(1);
      document.getElementById("note").textContent =
        "Measured just now at " + data.s.toLocaleString() + " cached tokens. " +
        x + "× on this PC. Animation slowed so you can see it. " + data.note;
    }
  }
  requestAnimationFrame(frame);
}
</script>
</body>
</html>
"""


def _time_us(fn, iters: int) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    samples = []
    for _ in range(3):
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iters * 1000.0)
    return statistics.median(samples)


def measure(seq_len: int) -> dict:
    if seq_len not in (1024, 4096, 16384):
        raise ValueError("unsupported cache length")
    b, hq, hkv, d = 1, 32, 8, 128
    q = torch.randn(b, hq, 1, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, hkv, seq_len, d, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    iters = 20 if seq_len >= 8192 else 40
    ours = _time_us(lambda: decode_attention(q, k, v), iters)
    sdpa = _time_us(lambda: F.scaled_dot_product_attention(q, k, v, enable_gqa=True), iters)
    return {
        "s": seq_len,
        "ours_us": ours,
        "sdpa_us": sdpa,
        "note": "PyTorch is on its math kernel here because this Windows install has no FlashAttention.",
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/measure":
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            seq = 4096
            for part in query.split("&"):
                if part.startswith("s="):
                    seq = int(part[2:])
            try:
                payload = json.dumps(measure(seq)).encode()
                code = 200
            except Exception as exc:
                payload = json.dumps({"error": str(exc)}).encode()
                code = 500
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(404)
        self.end_headers()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    server = HTTPServer(("127.0.0.1", 8765), Handler)
    print("Open http://127.0.0.1:8765")
    server.serve_forever()


if __name__ == "__main__":
    main()
