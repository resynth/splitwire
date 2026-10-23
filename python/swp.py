"""SWP/1 minimal Python implementation: framing, handshake, stage client.
Pure stdlib (+ `blake3` only when authentication or payload hashing is used).
This is the transport layer the exllama3/kyojin adapters build on."""
import json, socket, struct

MAGIC = b"SWP1"
HELLO, HELLO_OK, ERROR, PING, PONG = 0x01, 0x02, 0x0F, 0x10, 0x11
STAGE_CONFIGURE, STAGE_READY, FORWARD, FORWARD_OUT = 0x20, 0x21, 0x22, 0x23
CACHE_OP, CACHE_OK, SESSION_CLOSE, SESSION_CLOSED = 0x24, 0x25, 0x26, 0x27
FLAG_HASH = 1


class SwpError(Exception):
    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code, self.message = code, message


def _recv_exact(sock, n):
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        r = sock.recv_into(view[got:], n - got)
        if r == 0:
            raise ConnectionError("peer closed")
        got += r
    return buf


def send_frame(sock, kind, seq, header, payload=b"", hash=False):
    hdr = json.dumps(header, separators=(",", ":")).encode()
    fixed = MAGIC + struct.pack("<BBHIIQ", kind, FLAG_HASH if hash else 0, 0, seq, len(hdr), len(payload))
    sock.sendall(fixed + hdr)
    if len(payload):
        sock.sendall(payload)  # bytes / memoryview / numpy buffer: no copy
    if hash:
        import blake3
        sock.sendall(blake3.blake3(bytes(payload)).digest()[:16])


def recv_frame(sock):
    fixed = _recv_exact(sock, 24)
    if bytes(fixed[:4]) != MAGIC:
        raise ConnectionError("bad magic")
    kind, flags, _rsv, seq, hl, pl = struct.unpack("<BBHIIQ", bytes(fixed[4:]))
    header = json.loads(bytes(_recv_exact(sock, hl)))
    payload = _recv_exact(sock, pl) if pl else bytearray()
    if flags & FLAG_HASH:
        import blake3
        t = bytes(_recv_exact(sock, 16))
        if t != blake3.blake3(bytes(payload)).digest()[:16]:
            raise ConnectionError("payload hash mismatch")
    return kind, seq, header, payload


def _expect(frame, want):
    kind, _seq, header, payload = frame
    if kind == ERROR:
        raise SwpError(header.get("code", "error"), header.get("message", ""))
    if kind != want:
        raise ConnectionError(f"unexpected frame kind {kind:#x}")
    return header, payload


class StageClient:
    def __init__(self, addr, node="py-client", secret=None, caps=("coordinator",)):
        host, port = addr.rsplit(":", 1)
        self.sock = socket.create_connection((host, int(port)))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.seq = 10
        kind, _s, sh, _p = recv_frame(self.sock)
        assert kind == HELLO and sh["proto"] == "swp/1", "not an swp/1 server"
        hello = {"proto": "swp/1", "node": node, "caps": list(caps)}
        if sh.get("auth"):
            if not secret:
                raise SwpError("auth", "server requires a secret")
            import blake3
            key = blake3.blake3(secret.encode()).digest()
            hello["proof"] = blake3.blake3(f"{sh['nonce']}|{node}".encode(), key=key).hexdigest()
        send_frame(self.sock, HELLO, 1, hello)
        _expect(recv_frame(self.sock), HELLO_OK)
        self.server = sh

    def _rpc(self, kind, header, payload=b"", want=None):
        self.seq += 1
        send_frame(self.sock, kind, self.seq, header, payload)
        return _expect(recv_frame(self.sock), want)

    def configure(self, session, layers, has_embed, has_head, model, wire_dtype="f32"):
        return self._rpc(STAGE_CONFIGURE, {"session": session, "layers": list(layers), "has_embed": has_embed,
                                           "has_head": has_head, "wire_dtype": wire_dtype, "model": model}, want=STAGE_READY)[0]

    def forward(self, header, payload=b""):
        return self._rpc(FORWARD, header, payload, want=FORWARD_OUT)

    def truncate(self, session, pos):
        return self._rpc(CACHE_OP, {"session": session, "op": "truncate", "pos": pos}, want=CACHE_OK)[0]

    def close(self, session):
        return self._rpc(SESSION_CLOSE, {"session": session}, want=SESSION_CLOSED)[0]
