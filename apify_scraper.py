#!/usr/bin/env python3
"""
Interactive Apify scraper for Pinterest reference collection.
Parses search terms, runs the Apify scraper, upgrades image resolutions,
saves URLs, and handles bulk download and QC pipeline integration.
"""
import os
import sys
import json
import urllib.request
import urllib.parse
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Set

DEFAULT_ACTOR_ID = "fatihtahta/pinterest-scraper-search"
SEARCH_TERMS_FILE = "pinterest_search_terms.md"
OUTPUT_URLS_FILE = "pinterest_urls.txt"
RAW_DIR = "./pinterest_raw"
OUTPUT_DIR = "./output"
ARCHETYPE_DIR = "./archetype_anchors"


def parse_search_terms(file_path: Path) -> Dict[str, List[str]]:
    """Parse categories and search queries from pinterest_search_terms.md."""
    categories = {}
    current_category = None
    
    if not file_path.exists():
        print(f"Error: Search terms file not found at {file_path}")
        return {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Detect category header
            if line.startswith("## ") and ("References" in line or "Mood" in line or "Body" in line or "Outfit" in line):
                # Clean up header name, e.g., "1. Face References (Board: ...)" -> "Face References"
                header_part = line[3:].split("(")[0].strip()
                current_category = header_part
                categories[current_category] = []
            elif current_category and line.startswith("|") and "`" in line:
                # Extract query from line: e.g. | 1 | `woman face close up portrait natural beauty` | ...
                parts = line.split("|")
                for part in parts:
                    part = part.strip()
                    if part.startswith("`") and part.endswith("`") and len(part) > 2:
                        query = part[1:-1].strip()
                        categories[current_category].append(query)
                        break
                        
    # Filter out empty categories
    return {k: v for k, v in categories.items() if v}


def upgrade_url_resolution(url: str) -> str:
    """Upgrade Pinterest image URL to high-resolution 'originals' format."""
    parsed = urllib.parse.urlparse(url)
    if "pinimg.com" in parsed.netloc:
        # Pinterest CDN paths are usually like: /736x/ab/cd/ef/abcdef...jpg
        path_parts = parsed.path.split("/")
        if len(path_parts) > 1:
            # Check if first part after slash looks like a dimension (e.g. 736x, 564x, 236x)
            if "x" in path_parts[1] or path_parts[1].isdigit():
                path_parts[1] = "originals"
                new_path = "/".join(path_parts)
                # Reassemble URL without query params to avoid CDN caching issues
                return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, new_path, "", "", ""))
    return url


def run_apify_actor(token: str, actor_id: str, input_data: dict) -> dict:
    """Run an Apify actor and return the run details."""
    # Convert actor name (creator/actor) to Apify's path format: creator~actor
    formatted_actor_id = actor_id.replace("/", "~")
    url = f"https://api.apify.com/v2/acts/{formatted_actor_id}/runs?token={token}"
    
    req_body = json.dumps(input_data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_body,
        headers={"Content-Type": "application/json", "User-Agent": "AntigravityScraper/1.0"},
        method="POST"
    )
    
    with urllib.request.urlopen(req, timeout=30) as response:
        res_data = json.loads(response.read().decode("utf-8"))
        return res_data["data"]


def get_run_status(token: str, run_id: str) -> dict:
    """Fetch status of a specific actor run."""
    url = f"https://api.apify.com/v2/actor-runs/{run_id}?token={token}"
    req = urllib.request.Request(url, headers={"User-Agent": "AntigravityScraper/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        res_data = json.loads(response.read().decode("utf-8"))
        return res_data["data"]


def fetch_dataset_items(token: str, dataset_id: str) -> List[dict]:
    """Retrieve dataset items from a completed Apify run."""
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?token={token}"
    req = urllib.request.Request(url, headers={"User-Agent": "AntigravityScraper/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_images_from_items(items: List[dict]) -> Set[str]:
    """Extract and upgrade image URLs from dataset items."""
    urls = set()
    for item in items:
        # Check standard fields for Pinterest scraper actors
        img_url = None
        
        # 1. Check nested structures (e.g. images -> originals -> url)
        if "images" in item:
            images = item["images"]
            if isinstance(images, dict):
                if "originals" in images and isinstance(images["originals"], dict):
                    img_url = images["originals"].get("url")
                elif "736x" in images and isinstance(images["736x"], dict):
                    img_url = images["736x"].get("url")
                elif "url" in images:
                    img_url = images.get("url")
            elif isinstance(images, list) and len(images) > 0:
                img_url = images[0]
                
        # 2. Check flat fields
        if not img_url:
            img_url = item.get("image") or item.get("imageUrl") or item.get("image_url")
            
        # 3. Check pin URL if it directly points to an image
        if not img_url:
            pin_url = item.get("url") or item.get("pinterestUrl")
            if pin_url and any(pin_url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                img_url = pin_url
                
        if img_url and isinstance(img_url, str) and img_url.startswith("http"):
            # Upgrade and add
            urls.add(upgrade_url_resolution(img_url))
            
    return urls


def main():
    print("=" * 60)
    print("        APIFY PINTEREST REFERENCE SCRAPING UTILITY")
    print("=" * 60)
    
    # 1. Parse categories and search terms
    workspace_dir = Path.cwd()
    terms_file = workspace_dir / SEARCH_TERMS_FILE
    categories = parse_search_terms(terms_file)
    
    if not categories:
        print("No search terms could be parsed. Exiting.")
        sys.exit(1)
        
    print(f"Parsed {len(categories)} categories from {SEARCH_TERMS_FILE}:")
    cat_keys = list(categories.keys())
    for idx, cat in enumerate(cat_keys, 1):
        print(f"  [{idx}] {cat} ({len(categories[cat])} queries)")
        
    # 2. Category selection
    print("\nSelect categories to scrape:")
    print("  Enter numbers separated by commas (e.g., '1,3,4'), 'all', or press Enter for ALL.")
    selection = input("Selection: ").strip()
    
    selected_categories = []
    if not selection or selection.lower() == "all":
        selected_categories = cat_keys
    else:
        try:
            indices = [int(i.strip()) - 1 for i in selection.split(",") if i.strip().isdigit()]
            selected_categories = [cat_keys[i] for i in indices if 0 <= i < len(cat_keys)]
        except Exception:
            print("Invalid input. Defaulting to ALL categories.")
            selected_categories = cat_keys
            
    if not selected_categories:
        print("No valid categories selected. Exiting.")
        sys.exit(1)
        
    print(f"\nSelected categories: {', '.join(selected_categories)}")
    
    # Compile list of queries
    queries = []
    for cat in selected_categories:
        queries.extend(categories[cat])
        
    print(f"Total queries to run: {len(queries)}")
    for q in queries[:5]:
        print(f"  - {q}")
    if len(queries) > 5:
        print(f"  ... and {len(queries) - 5} more")
        
    # Let user select a subset of queries to avoid expensive run if wanted
    print(f"\nYou can run all {len(queries)} queries, or specify a limit.")
    run_limit_input = input("How many queries to run? (Press Enter for ALL): ").strip()
    if run_limit_input.isdigit():
        queries = queries[:int(run_limit_input)]
        print(f"Limited run to first {len(queries)} queries.")
        
    # 3. Choose number of pins per query
    pins_per_query = 10
    pins_input = input("\nHow many pins to scrape per query? (Default 10): ").strip()
    if pins_input.isdigit():
        pins_per_query = int(pins_input)
        
    # 4. Get Apify API Token
    token = os.environ.get("APIFY_API_TOKEN", "").strip()
    if not token:
        print("\nApify API Token not found in environment variable 'APIFY_API_TOKEN'.")
        token = input("Please enter your Apify API Token: ").strip()
        if not token:
            print("API Token is required to use Apify. Exiting.")
            sys.exit(1)
            
    # 5. Confirm Actor ID
    actor_id = input(f"\nEnter Apify Actor ID (Default: '{DEFAULT_ACTOR_ID}'): ").strip()
    if not actor_id:
        actor_id = DEFAULT_ACTOR_ID
        
    # Prepare actor inputs
    # Let's customize based on actor ID type
    input_data = {}
    if "pinterest-scraper-search" in actor_id or "fatihtahta" in actor_id:
        # fatihtahta/pinterest-scraper-search input schema
        input_data = {
            "queries": queries,
            "type": "all-pins",
            "limit": pins_per_query,
            "sentinent_analysis": False
        }
    elif "pinterest-crawler" in actor_id or "danielmilevski" in actor_id:
        # danielmilevski9/pinterest-crawler uses startUrls or search
        # We can construct search startUrls
        start_urls = []
        for q in queries:
            encoded_query = urllib.parse.quote(q)
            start_urls.append(f"https://www.pinterest.com/search/pins/?q={encoded_query}")
        input_data = {
            "startUrls": start_urls,
            "maxPinsCnt": pins_per_query,
            "proxyConfig": {
                "useApifyProxy": True
            }
        }
    else:
        # Fallback default schema assuming queries + limit
        input_data = {
            "queries": queries,
            "limit": pins_per_query,
            "maxPins": pins_per_query * len(queries)
        }
        
    print("\nTriggering Apify scraper actor...")
    print(f"Actor: {actor_id}")
    print(f"Queries count: {len(queries)}")
    print(f"Pins limit per query: {pins_per_query}")
    
    try:
        run_data = run_apify_actor(token, actor_id, input_data)
        run_id = run_data["id"]
        dataset_id = run_data["defaultDatasetId"]
        print(f"✓ Scraper run successfully started!")
        print(f"  Run ID: {run_id}")
        print(f"  Dataset ID: {dataset_id}")
        print(f"  View Progress: https://console.apify.com/actors/runs/{run_id}")
    except Exception as e:
        print(f"✗ Failed to start Apify actor run: {e}")
        sys.exit(1)
        
    # 6. Polling loop
    print("\nWaiting for scrape to complete (polling every 10 seconds)...")
    print("Press Ctrl+C to abort polling (the run will continue in background).")
    try:
        while True:
            status_data = get_run_status(token, run_id)
            status = status_data["status"]
            print(f"  [{time.strftime('%H:%M:%S')}] Status: {status}")
            
            if status in ["SUCCEEDED", "FINISHED"]:
                print("✓ Scrape completed successfully!")
                break
            elif status in ["FAILED", "ABORTED", "TIMED-OUT"]:
                print(f"✗ Scraper run ended with status: {status}")
                sys.exit(1)
                
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nPolling stopped. You can fetch results later using the run ID.")
        print(f"Run ID: {run_id}")
        sys.exit(0)
        
    # 7. Fetch results
    print("\nFetching dataset items...")
    try:
        items = fetch_dataset_items(token, dataset_id)
        print(f"Retrieved {len(items)} items from dataset.")
    except Exception as e:
        print(f"✗ Failed to fetch dataset items: {e}")
        sys.exit(1)
        
    # 8. Extract & upgrade URLs
    print("Extracting high-resolution image URLs...")
    image_urls = extract_images_from_items(items)
    print(f"Found {len(image_urls)} unique high-resolution image URLs.")
    
    if not image_urls:
        print("No image URLs could be extracted. Exiting.")
        sys.exit(0)
        
    # Write to urls file
    urls_file = workspace_dir / OUTPUT_URLS_FILE
    with open(urls_file, "w", encoding="utf-8") as f:
        for url in sorted(image_urls):
            f.write(f"{url}\n")
            
    print(f"✓ Saved image URLs to {urls_file.absolute()}")
    
    # 9. Ask user to download
    print("\n" + "=" * 60)
    do_download = input("Would you like to start the bulk download now? (y/n, default: y): ").strip().lower()
    if do_download != "n":
        print("\nStarting bulk downloader...")
        threads = input("Number of parallel threads (default: 8): ").strip()
        threads = int(threads) if threads.isdigit() else 8
        
        cmd = [
            sys.executable,
            "bulk_download.py",
            "--urls", str(urls_file),
            "--output", RAW_DIR,
            "--threads", str(threads)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print("✓ Bulk download finished successfully!")
        except Exception as e:
            print(f"✗ Error running bulk downloader: {e}")
            
    # 10. Ask user to run QC
    print("\n" + "=" * 60)
    do_qc = input("Would you like to run the AI-enhanced QC pipeline now? (y/n, default: y): ").strip().lower()
    if do_qc != "n":
        qc_script = "pinterest_qc_pro.py" if (workspace_dir / "pinterest_qc_pro.py").exists() else "pinterest_qc.py"
        print(f"\nStarting QC pipeline using {qc_script}...")
        
        cmd = [
            sys.executable,
            qc_script,
            "--archetype", ARCHETYPE_DIR,
            "--input", RAW_DIR,
            "--output", OUTPUT_DIR
        ]
        
        # Add config file if using pro version
        if qc_script == "pinterest_qc_pro.py" and (workspace_dir / "qc_config.yaml").exists():
            cmd.extend(["--config", "qc_config.yaml"])
            
        try:
            subprocess.run(cmd, check=True)
            print("✓ QC pipeline completed successfully!")
            print(f"Results are available in: {Path(OUTPUT_DIR).absolute()}")
        except Exception as e:
            print(f"✗ Error running QC pipeline: {e}")
            
    print("\nAll tasks completed!")


if __name__ == "__main__":
    main()
