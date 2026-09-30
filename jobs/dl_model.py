"""Download HF model weights, forcing IPv4 (IPv6 routes from the cluster hang)."""
import socket, sys
_gai = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: [r for r in _gai(*a, **k) if r[0] == socket.AF_INET]
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], allow_patterns=sys.argv[2:] or ["*.json", "*.txt", "model.safetensors"])
print("DL_DONE", sys.argv[1])
