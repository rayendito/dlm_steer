# make_gif.py
from pathlib import Path
import asyncio
from bs4 import BeautifulSoup
import imageio.v2 as imageio
from playwright.async_api import async_playwright

INPUT_HTML = Path("highlighted.html")
OUT_DIR = Path("frames")
OUT_GIF = Path("evolution.gif")

FRAME_DURATION = 0.9  # seconds per phase


def build_single_phase_html(original_html: str, phase):
    soup = BeautifulSoup(original_html, "html.parser")

    # remove all phases
    body = soup.body
    for section in body.find_all("section", class_="phase"):
        section.decompose()

    # insert only this phase
    body.append(BeautifulSoup(str(phase), "html.parser"))

    return str(soup)


async def render_frames():
    OUT_DIR.mkdir(exist_ok=True)

    original_html = INPUT_HTML.read_text(encoding="utf-8")
    soup = BeautifulSoup(original_html, "html.parser")
    phases = soup.select("section.phase")

    frame_paths = []

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(
            viewport={"width": 1200, "height": 800},
            device_scale_factor=2,
        )

        for i, phase in enumerate(phases, start=1):
            html = build_single_phase_html(original_html, phase)
            await page.set_content(html, wait_until="networkidle")

            frame_path = OUT_DIR / f"frame_{i:02d}.png"
            await page.screenshot(path=str(frame_path), full_page=True)
            frame_paths.append(frame_path)

        await browser.close()

    return frame_paths


async def main():
    frame_paths = await render_frames()

    frames = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(
        OUT_GIF,
        frames,
        duration=FRAME_DURATION,
        loop=0,
    )

    print(f"Saved GIF to: {OUT_GIF}")


if __name__ == "__main__":
    asyncio.run(main())