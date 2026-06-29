#!/usr/bin/env python3
"""
Bulk image downloader for Pinterest reference collection.

Auto-discovers all 'scrapped*.txt' files in the current directory,
filters out only real image URLs (i.pinimg.com/originals/ JPG/PNG/WEBP),
and downloads them into organized subfolders named after each search term.

Usage:
    python bulk_download.py                          # auto-discover all scrapped*.txt
    python bulk_download.py --urls my_urls.txt       # single file mode
    python bulk_download.py --output ./my_downloads  # custom output root
    python bulk_download.py --threads 8 --delay 0.3  # tuning
    python bulk_download.py --dry-run                # preview without downloading
"""
import os
import re
import sys
import argparse
import urllib.request
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# ── Image URL filter ────────────────────────────────────────────────────────
# Only keep high-res Pinterest originals (i.pinimg.com/originals/...)
# Exclude thumbnails like 30x30_RS, 75x75_RS, 140x140_RS
IMAGE_PATTERN = re.compile(
    r'https://i\.pinimg\.com/originals/[a-f0-9/]+\.(jpg|jpeg|png|webp)',
    re.IGNORECASE
)


def extract_image_urls(filepath: Path) -> list[str]:
    """Read a txt file and return only valid Pinterest original image URLs (deduplicated)."""
    seen = set()
    urls = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            url = line.strip()
            if IMAGE_PATTERN.match(url) and url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


def folder_name_from_file(filepath: Path) -> str:
    """Derive a clean folder name from the scrapped txt filename.

    'scrapped attractive woman oval face high cheekbones natural makeup portrait urls.txt'
    → 'attractive woman oval face high cheekbones natural makeup portrait'
    """
    stem = filepath.stem  # filename without extension
    # Strip leading 'scrapped' and trailing 'urls'
    name = re.sub(r'^scrapped\s*', '', stem, flags=re.IGNORECASE).strip()
    name = re.sub(r'\s*urls\s*$', '', name, flags=re.IGNORECASE).strip()
    # Sanitize for filesystem
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    return name or "images"


def download_image(url: str, output_dir: Path, idx: int, filename_prefix: str = "img") -> dict:
    """Download a single image with retry + exponential backoff."""
    result = {"url": url, "status": "failed", "path": None, "error": None}

    parsed = urllib.parse.urlparse(url)
    ext = Path(parsed.path).suffix.lower() or '.jpg'

    filename = f"{filename_prefix}_{idx:04d}{ext}"
    filepath = output_dir / filename

    # Skip if already downloaded
    if filepath.exists():
        result["status"] = "skipped"
        result["path"] = str(filepath)
        return result

    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/125.0.0.0 Safari/537.36'
        ),
        'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
        'Referer': 'https://www.pinterest.com/',
    }

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()
            with open(filepath, 'wb') as f:
                f.write(data)
            result["status"] = "success"
            result["path"] = str(filepath)
            return result
        except Exception as e:
            result["error"] = str(e)
            if attempt < 2:
                time.sleep(2 ** attempt)

    return result


def process_file(txt_file: Path, output_root: Path, threads: int, delay: float,
                 dry_run: bool, prefix: str) -> dict:
    """Filter URLs from one txt file and download them into a named subfolder."""
    folder_label = folder_name_from_file(txt_file)
    output_dir = output_root / folder_label
    urls = extract_image_urls(txt_file)

    print(f"\n{'='*60}")
    print(f"[FILE]   {txt_file.name}")
    print(f"[DIR]    {output_dir}")
    print(f"[FOUND]  {len(urls)} unique originals")

    if not urls:
        print("   [WARN] No image URLs matched -- skipping.");
        return {"file": str(txt_file), "found": 0, "success": 0, "skipped": 0, "failed": 0}

    if dry_run:
        print("   [DRY RUN] Would download the above URLs.")
        for u in urls[:5]:
            print(f"   - {u}")
        if len(urls) > 5:
            print(f"   ... and {len(urls) - 5} more")
        return {"file": str(txt_file), "found": len(urls), "success": 0, "skipped": 0, "failed": 0}

    output_dir.mkdir(parents=True, exist_ok=True)
    success = skipped = failed = 0

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {}
        for idx, url in enumerate(urls, start=1):
            future = executor.submit(download_image, url, output_dir, idx, prefix)
            futures[future] = url
            time.sleep(delay)

        for future in as_completed(futures):
            r = future.result()
            if r["status"] == "success":
                success += 1
                print(f"   [OK]   [{success+skipped+failed}/{len(urls)}] {Path(r['path']).name}")
            elif r["status"] == "skipped":
                skipped += 1
                print(f"   [SKIP] {Path(r['path']).name}")
            else:
                failed += 1
                err = (r["error"] or "")[:80]
                print(f"   [FAIL] {r['url'][-50:]} | {err}")

    print(f"\n   Summary -> OK:{success}  Skipped:{skipped}  Failed:{failed}")
    return {
        "file": str(txt_file),
        "found": len(urls),
        "success": success,
        "skipped": skipped,
        "failed": failed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Bulk-download Pinterest images from scrapped*.txt files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--urls", default=None,
                        help="Path to a single URL txt file (skips auto-discovery)")
    parser.add_argument("--output", default="./downloaded_images",
                        help="Root output directory (default: ./downloaded_images)")
    parser.add_argument("--threads", type=int, default=6,
                        help="Parallel download threads per file (default: 6)")
    parser.add_argument("--delay", type=float, default=0.2,
                        help="Seconds between requests (default: 0.2)")
    parser.add_argument("--prefix", default="img",
                        help="Filename prefix for downloaded images (default: img)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be downloaded without actually downloading")
    args = parser.parse_args()

    output_root = Path(args.output)
    workspace = Path(__file__).parent

    # Collect txt files to process
    if args.urls:
        txt_files = [Path(args.urls)]
    else:
        # Match 'scrapped*.txt', 'new scrapped*.txt', etc.
        txt_files = sorted(list(workspace.glob("scrapped*.txt")) + list(workspace.glob("new*.txt")))
        # Deduplicate list
        txt_files = sorted(list(set(txt_files)))

    if not txt_files:
        print("[ERROR] No 'scrapped*.txt' or 'new*.txt' files found in the workspace.")
        print("   Run with --urls <file> to specify a file manually.")
        sys.exit(1)

    print(f"[INFO] Found {len(txt_files)} file(s) to process:")
    for f in txt_files:
        print(f"   - {f.name}")

    all_results = []
    for txt_file in txt_files:
        result = process_file(
            txt_file, output_root,
            threads=args.threads,
            delay=args.delay,
            dry_run=args.dry_run,
            prefix=args.prefix
        )
        all_results.append(result)

    # Grand totals
    print(f"\n{'='*60}")
    print("GRAND TOTAL")
    print(f"{'='*60}")
    total_found = sum(r["found"] for r in all_results)
    total_ok    = sum(r["success"] for r in all_results)
    total_skip  = sum(r["skipped"] for r in all_results)
    total_fail  = sum(r["failed"] for r in all_results)
    print(f"   Files processed : {len(all_results)}")
    print(f"   Image URLs found: {total_found}")
    print(f"   Downloaded      : {total_ok}")
    print(f"   Skipped         : {total_skip}  (already existed)")
    print(f"   Failed          : {total_fail}")
    print(f"\n   Images saved to : {output_root.absolute()}")


if __name__ == "__main__":
    main()
