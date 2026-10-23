"""Per-stage timing + tokens/s accounting for a Python SWP coordinator (same definitions as `swp-drive --bench`).

  compute    = `stage_ms` reported in FORWARD_OUT (a real stage must sync its device before reading its clock)
  wire+queue = client round trip minus compute: serialization, network both ways, queueing, copies the stage does not count
  coordinator = wall time outside any hop (relay, sampling, bookkeeping)

Usage inside a coordinator:
    tr = Trace(layers=[(0, 4), (4, 8)])
    h, p = tr.forward(stage_idx, stage_client, header, payload, phase="decode")   # instead of stage_client.forward(...)
    tr.report(prefill_tokens=..., prefill_wall_ms=..., decode_tokens=..., decode_wall_ms=...)
"""
import time


class Trace:
    def __init__(self, layers):
        self.layers = layers
        self.hops = []  # (phase, stage, rtt_ms, stage_ms)

    def forward(self, stage, client, header, payload=b"", phase="decode"):
        t = time.perf_counter()
        h, p = client.forward(header, payload)
        rtt = (time.perf_counter() - t) * 1e3
        self.hops.append((phase, stage, rtt, float(h.get("stage_ms", 0.0))))
        return h, p

    def _sum(self, phase, stage):
        c = o = 0.0
        for ph, st, rtt, sm in self.hops:
            if ph == phase and st == stage:
                c += sm
                o += max(rtt - sm, 0.0)
        return c, o

    def report(self, prefill_tokens, prefill_wall_ms, decode_tokens, decode_wall_ms):
        n = len(self.layers)
        out = {"prefill_tok_s": prefill_tokens / (prefill_wall_ms / 1e3),
               "decode_tok_s": decode_tokens / (decode_wall_ms / 1e3), "stages": []}
        print(f"PREFILL {prefill_tokens} tok in {prefill_wall_ms:.1f} ms -> {out['prefill_tok_s']:.0f} tok/s")
        print(f"DECODE  {decode_tokens} tok in {decode_wall_ms:.1f} ms -> {out['decode_tok_s']:.1f} tok/s")
        print("  stage layers   prefill: compute  wire+q   decode/tok: compute  wire+q   busy")
        for i, (a, b) in enumerate(self.layers):
            pc, po = self._sum("prefill", i)
            dc, do = self._sum("decode", i)
            s = {"stage": i, "layers": [a, b], "prefill_compute_ms": pc, "prefill_wire_ms": po,
                 "decode_compute_ms_per_tok": dc / decode_tokens, "decode_wire_ms_per_tok": do / decode_tokens}
            out["stages"].append(s)
            print(f"  {i:>5} {a:>2}..{b:<2}   {pc:>16.1f} {po:>7.1f}   {dc/decode_tokens:>18.3f} {do/decode_tokens:>7.3f}   {dc/decode_wall_ms*100:>3.0f}%")
        return out
