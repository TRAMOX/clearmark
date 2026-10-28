"""Command-line version of ClearMark, handy for batches of videos.

Watermark / thumbnail / export defaults come from the settings you last used in the app,
so you can set things up once in the app and then batch-process from here.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

from vwstudio.ai_fill import MODEL_SIZE, download_model, model_available
from vwstudio.logo_removal import METHODS, LogoRegion, detect_logos, refine_region
from vwstudio.media import probe, sample_frames
from vwstudio.pipeline import Processor, export_video, render_thumbnail
from vwstudio.settings import QUALITY, load_settings
from vwstudio.thumbnail import best_frame_time
from vwstudio.watermark import POSITIONS

THUMB_SIZES = {"youtube": "YouTube 1280x720", "shorts": "Shorts / Reels 1080x1920",
               "square": "Square 1080x1080", "video": "Same as video"}

EXAMPLES = """examples:
  python cli.py clip.mp4 --auto --text "@mychannel"
  python cli.py clip.mp4 --region 560,1120,90,115 --logo mylogo.png --logo-pos top-left
  python cli.py "D:/videos/*.mp4" --saved -o D:/out --thumbnail auto --title "Watch this!" --cover
"""


def parse_region(text: str) -> LogoRegion:
    box, _, when = text.partition("@")
    try:
        x, y, w, h = (int(float(v)) for v in box.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected X,Y,W,H[@START-END], got {text!r}")
    start = end = None
    if when:
        a, _, b = when.partition("-")
        start = float(a) if a else None
        end = float(b) if b else None
    return LogoRegion(x, y, w, h, start, end)


def build_parser(settings) -> argparse.ArgumentParser:
    wm, th, ex = settings.watermark, settings.thumbnail, settings.export
    p = argparse.ArgumentParser(prog="clearmark", description="ClearMark: remove logos, add your watermark and make thumbnails.",
                                epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="*", help="video file(s); wildcards like *.mp4 are fine")
    p.add_argument("--download-model", action="store_true",
                   help=f"download the AI fill model ({MODEL_SIZE // 1_000_000} MB) used by --method smart")
    p.add_argument("-o", "--output", help="output file (single input) or output folder")

    g = p.add_argument_group("logo removal")
    g.add_argument("--auto", action="store_true", help="find logos that stay in one place automatically")
    g.add_argument("--region", type=parse_region, action="append", default=[], metavar="X,Y,W,H[@START-END]",
                   help="box to clean (pixels), optionally only between START and END seconds; repeatable")
    g.add_argument("--saved", action="store_true", help="use the logo boxes saved in the app for this video size")
    g.add_argument("--box-only", action="store_true", help="fill each --region box completely "
                                                         "instead of finding the exact logo shape")
    g.add_argument("--method", choices=list(METHODS), default=settings.removal_method,
                   help="smart = real background from other frames + AI fill (best, slower); "
                        "inpaint = fast fill; blur")
    g.add_argument("--grow", type=int, default=settings.removal_expand, help="grow logo areas by N pixels")

    w = p.add_argument_group("your watermark")
    w.add_argument("--text", default=wm.text, help="text watermark, e.g. @mychannel")
    w.add_argument("--font", default=wm.font, help="font file name (arialbd.ttf) or path")
    w.add_argument("--text-size", type=float, default=wm.text_size, help="%% of the video's shorter side")
    w.add_argument("--text-color", default=wm.text_color)
    w.add_argument("--text-opacity", type=float, default=wm.text_opacity)
    w.add_argument("--text-pos", choices=POSITIONS, default=wm.text_position)
    w.add_argument("--no-outline", action="store_true")
    w.add_argument("--logo", default=wm.logo_path, help="your logo image (PNG)")
    w.add_argument("--logo-size", type=float, default=wm.logo_size)
    w.add_argument("--logo-opacity", type=float, default=wm.logo_opacity)
    w.add_argument("--logo-pos", choices=POSITIONS, default=wm.logo_position)
    w.add_argument("--no-watermark", action="store_true", help="don't add any watermark")

    t = p.add_argument_group("thumbnail")
    t.add_argument("--thumbnail", nargs="?", const="auto", metavar="SECONDS",
                   help="save <name>_thumbnail.jpg from the frame at SECONDS (default: best frame)")
    t.add_argument("--title", default=th.title, help="title text on the thumbnail")
    t.add_argument("--thumb-size", choices=list(THUMB_SIZES), default=None)
    t.add_argument("--cover", action="store_true", help="embed the thumbnail as the video's cover")

    e = p.add_argument_group("export")
    e.add_argument("--crf", type=int, default=QUALITY.get(ex.quality, 19),
                   help="quality: lower = better/bigger (16-28)")
    e.add_argument("--preset", default=ex.preset, choices=["ultrafast", "veryfast", "fast", "medium", "slow"])
    e.add_argument("--no-audio", action="store_true")
    e.add_argument("--thumbnail-only", action="store_true", help="only make thumbnails, don't export video")
    return p


def expand_inputs(patterns: list[str]) -> list[str]:
    files = []
    for pat in patterns:
        matches = sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
        files.extend(m for m in matches if os.path.isfile(m))
    return files


def output_path(src: str, output: str | None, many: bool) -> str:
    name = os.path.splitext(os.path.basename(src))[0] + "_edited.mp4"
    if not output:
        return os.path.join(os.path.dirname(os.path.abspath(src)), name)
    if many or os.path.isdir(output) or not os.path.splitext(output)[1]:
        os.makedirs(output, exist_ok=True)
        return os.path.join(output, name)
    return output


def main(argv=None) -> int:
    settings = load_settings()
    parser = build_parser(settings)
    args = parser.parse_args(argv)
    if args.download_model:
        def show(done, total):
            print(f"\r  downloading AI model {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
        print(f"\n  saved -> {download_model(show)}")
        if not args.inputs:
            return 0
    if not args.inputs:
        parser.error("give at least one video file")
    if args.method == "smart" and not model_available():
        print("Note: AI fill model not installed; hidden areas get a simpler fill. "
              "Run with --download-model to install it.")
    files = expand_inputs(args.inputs)
    if not files:
        print("No input videos found.", file=sys.stderr)
        return 1

    wm = settings.watermark
    wm.text, wm.font, wm.text_size = args.text, args.font, args.text_size
    wm.text_color, wm.text_opacity, wm.text_position = args.text_color, args.text_opacity, args.text_pos
    wm.text_outline = wm.text_outline and not args.no_outline
    wm.logo_path, wm.logo_size, wm.logo_opacity, wm.logo_position = (
        args.logo, args.logo_size, args.logo_opacity, args.logo_pos)
    th = settings.thumbnail
    th.title = args.title
    if args.thumb_size:
        th.size = THUMB_SIZES[args.thumb_size]

    failures = 0
    for n, src in enumerate(files, 1):
        print(f"[{n}/{len(files)}] {src}")
        try:
            info = probe(src)
            regions = list(args.region)
            samples = None
            if args.auto or (regions and not args.box_only):
                samples = sample_frames(info, 32)
            if regions and not args.box_only:
                regions = [refine_region(samples[1], r.clamp(info.width, info.height)) for r in regions]
            if args.saved:
                saved = settings.saved_regions.get(f"{info.width}x{info.height}", [])
                regions += [LogoRegion.from_dict(d) for d in saved]
                print(f"  using {len(saved)} saved logo box(es)")
            if args.auto:
                found = detect_logos(samples[1])
                print(f"  auto-detect found {len(found)} logo(s): "
                      + ", ".join(f"{r.x},{r.y},{r.w},{r.h}" for r in found))
                regions += found

            processor = Processor(info, regions, args.method, args.grow, None if args.no_watermark else wm)
            out = output_path(src, args.output, len(files) > 1)

            cover = None
            if args.thumbnail or args.cover or args.thumbnail_only:
                t = best_frame_time(info) if args.thumbnail in (None, "auto") else float(args.thumbnail)
                cover = render_thumbnail(info, processor, t, th)
                thumb_path = os.path.splitext(out)[0] + "_thumbnail.jpg"
                cover.save(thumb_path, quality=92)
                print(f"  thumbnail ({t:.2f}s) -> {thumb_path}")
            if args.thumbnail_only:
                continue

            def progress(i, total):
                print(f"\r  exporting {100 * i / max(1, total):5.1f}%", end="", flush=True)
            export_video(info, out, processor, crf=args.crf, preset=args.preset, keep_audio=not args.no_audio,
                         cover=cover if args.cover else None, progress=progress)
            print(f"\r  saved -> {out}            ")
        except Exception as e:
            failures += 1
            print(f"\n  FAILED: {e}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
