"""Run against two sim-compute mock stages:
  swp-mockstage --listen 127.0.0.1:7821 --sim-base-ms 3 --sim-row-us 10 &
  swp-mockstage --listen 127.0.0.1:7822 --sim-base-ms 6 --sim-row-us 25 &
  python3 test_bench.py"""
import struct, sys, time
from swp import StageClient
from bench import Trace

HIDDEN, VOCAB, LAYERS = 16, 64, 8
addrs = sys.argv[1:] or ["127.0.0.1:7821", "127.0.0.1:7822"]
stages = [StageClient(a) for a in addrs]
bounds = [0, 4, LAYERS]
for i, s in enumerate(stages):
    s.configure("b", (bounds[i], bounds[i + 1]), i == 0, i == len(stages) - 1, {"kind": "mock", "hidden": HIDDEN, "vocab": VOCAB})
tr = Trace([(bounds[i], bounds[i + 1]) for i in range(len(stages))])
step = 0

def fwd(start, toks, want, phase):
    global step
    step += 1
    hidden = b""
    for i, s in enumerate(stages):
        last = i == len(stages) - 1
        h = {"session": "b", "step": step, "start_pos": start, "n_rows": len(toks), "token_ids": toks,
             "input": "tokens" if i == 0 else "hidden", "dtype": "f32", "shape": [len(toks), HIDDEN],
             "want": want if last else "hidden", "return_rows": "last", "rewrite": False}
        oh, hidden = tr.forward(i, s, h, hidden, phase)
    return list(struct.unpack(f"<{len(hidden)//4}i", bytes(hidden))) if hidden else []

prompt = [(i * 7 + 3) % VOCAB for i in range(256)]
t0 = time.perf_counter(); pos = 0
for ci in range(0, len(prompt), 64):
    c = prompt[ci:ci + 64]
    first = fwd(pos, c, "token" if ci + 64 >= len(prompt) else "none", "prefill"); pos += len(c)
pre = (time.perf_counter() - t0) * 1e3
tok, t1 = first[0], time.perf_counter()
for _ in range(32):
    tok = fwd(pos, [tok], "token", "decode")[0]; pos += 1
dec = (time.perf_counter() - t1) * 1e3
r = tr.report(256, pre, 32, dec)
assert 80 < r["decode_tok_s"] < 130 and r["stages"][1]["decode_compute_ms_per_tok"] > r["stages"][0]["decode_compute_ms_per_tok"]
print("OK: python accounting agrees with the simulated 3 ms / 6 ms stages")
