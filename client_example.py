"""Minimal client for the CosyVoice HTTP API (stdlib only).

Usage:
    python client_example.py --text "你好，世界" --voice myvoice --output out.mp3
    python client_example.py --text "Hello" --voice myvoice --format wav --model cosyvoice3
    python client_example.py --upload-voice myvoice --voice-file ref.wav --prompt-text "参考音频的文字"
"""

import argparse
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
import uuid


def _request(url: str, payload: dict, token: str) -> bytes:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def upload_voice(base: str, token: str, voice_id: str, file_path: str, prompt_text: str) -> dict:
    boundary = uuid.uuid4().hex
    parts = []

    def field(name: str, value: str) -> None:
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        )

    field("name", voice_id)
    if prompt_text:
        field("prompt_text", prompt_text)
    mime = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    with open(file_path, "rb") as fp:
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{os.path.basename(file_path)}\"\r\nContent-Type: {mime}\r\n\r\n".encode()
        )
        parts.append(fp.read())
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(base + "/v1/voices", data=b"".join(parts), method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def main() -> int:
    parser = argparse.ArgumentParser(description="Call the CosyVoice /v1/audio/speech endpoint")
    parser.add_argument("--url", default="http://127.0.0.1:8001", help="Server base URL")
    parser.add_argument("--token", default=None, help="Bearer token (only if the server requires one)")
    parser.add_argument("--model", default="cosyvoice2", help="cosyvoice2 / cosyvoice3")
    parser.add_argument("--text", default=None, help="Text to synthesize")
    parser.add_argument("--voice", default=None, help="Voice id (file name in the voices directory)")
    parser.add_argument("--prompt-text", default=None, help="Override the transcript stored with the voice")
    parser.add_argument("--output", default="output.mp3", help="Output file path")
    parser.add_argument("--format", default="mp3", choices=["mp3", "wav", "ogg", "pcm"])
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None, help="Reproducible synthesis")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--upload-voice", default=None, help="Voice id to create from --voice-file")
    parser.add_argument("--voice-file", default=None, help="Reference audio for --upload-voice")
    args = parser.parse_args()

    token = args.token or ""
    base = args.url.rstrip("/")

    if args.upload_voice:
        if not args.voice_file:
            parser.error("--upload-voice requires --voice-file")
        info = upload_voice(base, token, args.upload_voice, args.voice_file, args.prompt_text or "")
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0

    if not args.text or not args.voice:
        parser.error("--text and --voice are required for synthesis")

    payload = {
        "model": args.model,
        "input": args.text,
        "voice": args.voice,
        "response_format": args.format,
        "speed": args.speed,
        "stream": args.stream,
    }
    if args.prompt_text is not None:
        payload["prompt_text"] = args.prompt_text
    if args.seed is not None:
        payload["seed"] = args.seed

    try:
        if args.stream:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(base + "/v1/audio/speech", data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(req) as resp, open(args.output, "wb") as out:
                while chunk := resp.read(64 * 1024):
                    out.write(chunk)
            print(f"saved {args.output} (streamed)")
            return 0

        audio = _request(base + "/v1/audio/speech", payload, token)
        with open(args.output, "wb") as fp:
            fp.write(audio)
        print(f"saved {args.output} ({len(audio)} bytes)")
        return 0
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
