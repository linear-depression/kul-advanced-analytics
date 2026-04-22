import os
import json
import zipfile
import tempfile
import urllib.request
import urllib.parse
from pathlib import Path


class DataDownloader:
    """Downloads and saves project data files into proj2/data/."""

    # Resolve proj2/data relative to this script's location (proj2/scripts/)
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"

    SOURCES = {
        "zip": {
            "pcloud_code": "XZYQmGZwIUEyLDBKRmnOsRuq4zub5BpRs2X",
            "description": "zipped dataset",
        },
        "json": {
            "url": "https://seppe.net/aa/assignment2/minifigs.json",
            "output": "minifigs.json",
            "description": "minifigs JSON",
        },
    }

    # e.pcloud.link shares are hosted on the EU cluster → try eapi first, then api
    _PCLOUD_API_CANDIDATES = [
        "https://eapi.pcloud.com/getpublinkdownload?code={code}",
        "https://api.pcloud.com/getpublinkdownload?code={code}",
    ]

    _HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DataDownloader/1.0)"}

    def __init__(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def download_all(self) -> None:
        """Download and save both files."""
        print(f"Saving data to: {self.DATA_DIR}\n")
        self._download_zip()
        self._download_json()
        print("\nAll downloads complete.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_pcloud_url(self, code: str) -> str:
        """
        Try each pCloud API cluster in turn until one returns a valid
        direct download URL for the given share code.
        """
        last_error = None
        for api_template in self._PCLOUD_API_CANDIDATES:
            api_url = api_template.format(code=code)
            try:
                req = urllib.request.Request(api_url, headers=self._HEADERS)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    payload = json.loads(resp.read().decode())

                if payload.get("result") == 0:
                    host = payload["hosts"][0]
                    path = payload["path"]
                    direct = f"https://{host}{path}"
                    print(f"      Resolved via {api_url.split('/')[2]}: {direct}")
                    return direct

                last_error = f"API returned error {payload.get('result')}: {payload.get('error')}"
                print(f"      {api_url.split('/')[2]} -> {last_error}, trying next...")

            except Exception as exc:
                last_error = str(exc)
                print(f"      {api_url.split('/')[2]} unreachable ({exc}), trying next...")

        raise RuntimeError(
            f"Could not resolve pCloud share code '{code}'. Last error: {last_error}"
        )

    def _download_zip(self) -> None:
        """Resolve the pCloud share link, download the ZIP, and extract it."""
        src = self.SOURCES["zip"]
        print(f"[1/2] Downloading {src['description']}...")
        print("      Resolving pCloud share link...", flush=True)

        direct_url = self._resolve_pcloud_url(src["pcloud_code"])

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self._fetch(direct_url, tmp_path)

            with zipfile.ZipFile(tmp_path, "r") as zf:
                names = zf.namelist()
                print(f"      Extracting {len(names)} file(s): {names}")
                zf.extractall(self.DATA_DIR)

            print(f"      Extracted to {self.DATA_DIR}")
        finally:
            os.unlink(tmp_path)

    def _download_json(self) -> None:
        """Download the JSON file and save it to DATA_DIR."""
        src = self.SOURCES["json"]
        dest = self.DATA_DIR / src["output"]

        print(f"[2/2] Downloading {src['description']}...")
        self._fetch(src["url"], dest)

        with open(dest, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            print(f"      Saved {len(data)} records -> {dest}")
        elif isinstance(data, dict):
            print(f"      Saved JSON object with {len(data)} keys -> {dest}")
        else:
            print(f"      Saved -> {dest}")

    def _fetch(self, url: str, dest: "Path | str") -> None:
        """Download *url* to *dest* with a live progress indicator."""
        dest = Path(dest)
        req = urllib.request.Request(url, headers=self._HEADERS)

        with urllib.request.urlopen(req, timeout=60) as response, open(dest, "wb") as out:
            total = response.headers.get("Content-Length")
            total = int(total) if total else None
            downloaded = 0
            chunk_size = 65_536  # 64 KB

            while chunk := response.read(chunk_size):
                out.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(
                        f"\r      {pct:5.1f}%  ({downloaded:,} / {total:,} bytes)",
                        end="",
                        flush=True,
                    )

            if total:
                print()  # newline after progress bar


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    DataDownloader().download_all()