"""Interop test: drive Rust swp-mockstage processes from Python and compare with the Rust driver's tokens.
  swp-mockstage --listen 127.0.0.1:7801 & swp-mockstage --listen 127.0.0.1:7802 &
  python3 test_interop.py 127.0.0.1:7801 127.0.0.1:7802"""
import struct, sys
from swp import StageClient, SwpError

HIDDEN, VOCAB, LAYERS = 16, 64, 8
EXPECTED = [53, 47, 31, 1, 54, 47, 59, 5, 36, 51]          # from `swp-drive --selftest` reference
PROMPT = [5, 17, 3, 42, 9, 60, 21, 8, 33, 12, 1]

addrs = sys.argv[1:] or ["127.0.0.1:7801", "127.0.0.1:7802"]
stages = [StageClient(a) for a in addrs]
n, bounds = len(stages), [0, 4, LAYERS]
for i, s in enumerate(stages):
    s.configure("py", (bounds[i], bounds[i + 1]), i == 0, i == n - 1, {"kind": "mock", "hidden": HIDDEN, "vocab": VOCAB})

step = 0
def forward(start, toks, want="token"):
    global step
    step += 1
    hidden = b""
    for i, s in enumerate(stages):
        last = i == n - 1
        h = {"session": "py", "step": step, "mode": "prefill" if len(toks) > 1 else "decode", "start_pos": start,
             "n_rows": len(toks), "token_ids": toks, "input": "tokens" if i == 0 else "hidden", "dtype": "f32",
             "shape": [len(toks), HIDDEN], "want": want if last else "hidden", "return_rows": "last", "rewrite": False}
        oh, hidden = s.forward(h, hidden)
    return list(struct.unpack(f"<{len(hidden)//4}i", bytes(hidden)))

pos, first = 0, []
for ci in range(0, len(PROMPT), 4):
    c = PROMPT[ci:ci + 4]
    last_chunk = ci + 4 >= len(PROMPT)
    first = forward(pos, c, "token" if last_chunk else "none")
    pos += len(c)
out = [first[0]]
while len(out) < 10:
    out.append(forward(pos, [out[-1]])[0]); pos += 1
print("python tokens:", out)
assert out == EXPECTED, f"MISMATCH vs rust reference {EXPECTED}"
try:
    stages[0].forward({"session": "py", "start_pos": 0, "n_rows": 1, "token_ids": [1], "input": "tokens", "want": "hidden"})
    raise SystemExit("expected pos_mismatch")
except SwpError as e:
    assert e.code == "pos_mismatch"
print("OK: Python client interoperates with Rust stage (tokens identical, errors surfaced)")
