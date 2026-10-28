# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 1.x | Yes |

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Report vulnerabilities privately through GitHub's [private vulnerability reporting](https://github.com/TRAMOX/clearmark/security/advisories/new). Include:

- a description of the issue and its impact,
- steps to reproduce (a sample file if relevant),
- the affected version and platform.

You'll get an acknowledgement within 7 days. We'll work with you on a fix and coordinate disclosure once a patched release is available.

## Scope notes

- ClearMark processes files locally and makes no network requests, except the optional one-time AI model download from `huggingface.co`. The download is size-checked before use.
- Media files are decoded by OpenCV and FFmpeg. Vulnerabilities in those libraries should also be reported upstream.
