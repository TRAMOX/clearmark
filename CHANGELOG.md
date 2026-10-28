# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-10-28

### Added
- Desktop app with a four-step workflow: remove logo, add watermark, make thumbnail, export.
- Automatic detection of static and semi-transparent logos, with exact shape masks.
- **Best** removal method: background recovered from neighbouring frames using motion tracking, seam-free blending, and LaMa AI fill for areas never uncovered, carried with the camera for flicker-free results.
- Fast fill and blur removal methods.
- Per-region time ranges for logos that move or appear only part of the time.
- Text and image watermarks with font, colour, outline, opacity and nine positions.
- Thumbnails with automatic best-frame selection, titles, and YouTube / Shorts / square sizes.
- H.264 export keeping the original audio, with optional embedded cover art.
- Batch command-line interface that shares settings with the app.
- Saved settings and reusable logo boxes per resolution.
- One-time in-app download of the AI model.
