from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DistributionConfig:
    state_dir: Path
    root_path: str = "/Neri_Data/Model"
    openlist_url: str = "http://127.0.0.1:5244"
    openlist_token: str = ""
    public_url: str = "https://myneri.top/api/models"
    capability_ttl_seconds: int = 600
    requests_per_minute: int = 120
    daily_proxy_ip_bytes: int = 2 * 1024**3
    daily_proxy_total_bytes: int = 20 * 1024**3

    @classmethod
    def from_env(cls):
        return cls(
            state_dir=Path(os.environ.get("NERI_MODEL_STATE", "/var/lib/neri-models")),
            openlist_url=os.environ.get("NERI_OPENLIST_URL", "http://127.0.0.1:5244"),
            openlist_token=os.environ["NERI_OPENLIST_TOKEN"],
            public_url=os.environ.get("NERI_MODEL_PUBLIC_URL", "https://myneri.top/api/models"),
            capability_ttl_seconds=int(os.environ.get("NERI_MODEL_CAPABILITY_TTL", "600")),
            requests_per_minute=int(os.environ.get("NERI_MODEL_REQUESTS_PER_MINUTE", "120")),
            daily_proxy_ip_bytes=int(os.environ.get("NERI_MODEL_PROXY_IP_BYTES", str(2 * 1024**3))),
            daily_proxy_total_bytes=int(os.environ.get("NERI_MODEL_PROXY_TOTAL_BYTES", str(20 * 1024**3))),
        )
