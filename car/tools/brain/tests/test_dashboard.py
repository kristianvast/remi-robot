"""Verify that a dashboard client can load the page and its local assets."""

import re
from html.parser import HTMLParser
from types import SimpleNamespace
from typing import Final, Literal
from urllib.parse import urldefrag, urljoin, urlsplit

import anyio
import httpx2
import pytest
from conftest import LOOPBACK, FakeCar
from pydantic import BaseModel
from websockets.asyncio.client import connect

from behaviors import scan
from car import Car
from dashboard import Dashboard
from pilot import Pilot
from telemetry import Telemetry

MODULE_PATH: Final = re.compile(
    r"""(?:\b(?:import|export)\s+[^;'"]*?\s+from\s*|\bimport\s*\(?\s*)"""
    r"""['"]((?:/|\./|\.\./)[^'"]+)['"]""",
)
CSS_PATH: Final = re.compile(
    r"""(?:url\(\s*['"]?([^'"\s)]+)['"]?\s*\)|@import\s+['"]([^'"]+)['"])""",
)


class PageAssets(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: set[str] = set()
        self._inline_kind = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.references.update(value for key, value in attrs if key in {"src", "href"} and value)
        if tag in {"script", "style"}:
            self._inline_kind = tag
        if tag == "script" and ("type", "importmap") in attrs:
            self._inline_kind = "importmap"

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self._inline_kind = ""

    def handle_data(self, data: str) -> None:
        if self._inline_kind == "script":
            self.references.update(MODULE_PATH.findall(data))
        if self._inline_kind == "style":
            self.references.update(url or imported for url, imported in CSS_PATH.findall(data))
        if self._inline_kind == "importmap":
            # Import-map aliases may point to a local module or to an external CDN.
            self.references.update(
                match.group(1) for match in re.finditer(r':\s*"([^"]+)"', data)
                if not match.group(1).endswith("/")
            )


class MapAssets(BaseModel, frozen=True):
    image: str


class Hello(BaseModel, frozen=True):
    t: Literal["hello"]
    car_model: str
    map: MapAssets


@pytest.mark.asyncio
async def test_dashboard_serves_page_local_assets_and_websocket_hello(
    fake_car: FakeCar, dashboard_port: int, local_http: httpx2.Client,
) -> None:
    stream_url = f"{fake_car.camera_url}/stream"
    telemetry = Telemetry(25)
    car = Car(LOOPBACK, fake_car.port, telemetry)
    pane = Dashboard(telemetry, Pilot(car, telemetry, 150, scan), SimpleNamespace(), stream_url, dashboard_port)
    origin = f"http://{LOOPBACK}:{dashboard_port}"

    try:
        await pane.start()
        page = await anyio.to_thread.run_sync(local_http.get, f"{origin}/")
        assert page.status_code == 200
        assert page.headers["content-type"].split(";", 1)[0] == "text/html"

        async with connect(f"ws://{LOOPBACK}:{dashboard_port}/ws", open_timeout=3, close_timeout=1,
                           proxy=None) as ws:
            with anyio.fail_after(3):
                hello = Hello.model_validate_json(await ws.recv(decode=False))

        assets = PageAssets()
        assets.feed(page.text)
        assets.close()
        assets.references.update((hello.map.image, hello.car_model))

        pending = {(f"{origin}/", ref) for ref in assets.references}
        seen = {f"{origin}/"}
        while pending:
            parent, ref = pending.pop()
            url, _ = urldefrag(urljoin(parent, ref))
            parts = urlsplit(url)
            if parts.scheme != "http" or parts.netloc != f"{LOOPBACK}:{dashboard_port}" or url in seen:
                continue
            seen.add(url)
            response = await anyio.to_thread.run_sync(local_http.get, url)
            assert response.status_code == 200, f"{url} referenced by {parent}"
            content_type = response.headers.get("content-type", "")
            if "javascript" in content_type or parts.path.endswith((".js", ".mjs")):
                pending.update((url, child) for child in MODULE_PATH.findall(response.text))
            if "text/css" in content_type or parts.path.endswith(".css"):
                pending.update((url, src or imported) for src, imported in CSS_PATH.findall(response.text))
    finally:
        await pane.close()
        await anyio.to_thread.run_sync(car.close)
