#!/usr/bin/env python3
"""
Bulk image downloader for Pinterest reference collection.
Usage: python bulk_download.py --urls urls.txt --output ./nyra_refs/

urls.txt format: one image URL per line (from Pinterest RSS or browser console export)
"""
import os
import sys
import argparse
import urllib.request
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def download_image(url: str, output_dir: Path, idx: int) -> dict:
    """Download a single image with retry logic."""
    result = {"url": url, "status": "failed", "path": None, "error": None}

    # Pinterest CDN URLs often have query params; get clean extension
    parsed = urllib.parse.urlparse(url)
    path = parsed.path
    ext = Path(path).suffix
    if not ext or ext not in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
        ext = '.jpg'  # default

    filename = f"nyra_ref_{idx:03d}{ext}"
    filepath = output_dir / filename

    # Request headers to avoid 403 blocks
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
        'Referer': 'https://www.pinterest.com/',
    }

    max_retries = 3
    for attempt in range(max_retries):
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
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
            continue

    return result


def main():
    parser = argparse.ArgumentParser(description="Bulk download Pinterest reference images")
    parser.add_argument("--urls", required=True, help="Path to text file with one URL per line")
    parser.add_argument("--output", default="./nyra_refs", help="Output directory for images")
    parser.add_argument("--threads", type=int, default=5, help="Number of parallel download threads")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests (seconds)")
    args = parser.parse_args()

    urls_file = Path(args.urls)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not urls_file.exists():
        print(f"Error: URLs file not found: {urls_file}")
        sys.exit(1)

    with open(urls_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    print(f"Found {len(urls)} URLs to download")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Parallel threads: {args.threads}")
    print("-" * 50)

    success_count = 0
    fail_count = 0
    results = []

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {}
        for idx, url in enumerate(urls, start=1):
            future = executor.submit(download_image, url, output_dir, idx)
            futures[future] = url
            time.sleep(args.delay)  # rate limiting between submissions

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result["status"] == "success":
                success_count += 1
                print(f"[OK] {result['path']}")
            else:
                fail_count += 1
                print(f"[FAIL] {result['url'][:60]}... | Error: {result['error'][:60]}")

    print("-" * 50)
    print(f"Download complete: {success_count} success, {fail_count} failed out of {len(urls)} total")
    print(f"Images saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
