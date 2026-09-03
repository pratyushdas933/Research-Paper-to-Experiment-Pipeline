"""
edgar_dataset_builder.py

Builds a REAL (not synthetic) labeled M&A deal-outcome dataset from SEC
EDGAR's free, public full-text search and filing APIs.

RUN THIS LOCALLY, NOT IN A SANDBOXED/NETWORK-RESTRICTED ENVIRONMENT --
it needs to reach https://efts.sec.gov and https://data.sec.gov freely,
which a locked-down sandbox typically won't allow.

Labeling logic:
  1. Search EDGAR full-text search for 8-K filings disclosing Item 1.01
     ("Entry into a Material Definitive Agreement") that mention a merger
     agreement -- these are deal ANNOUNCEMENTS.
  2. For each filer (by CIK), look at their subsequent 8-K filings within
     a follow-up window:
       - An Item 1.02 ("Termination of a Material Definitive Agreement")
         filed by the same company within the window -> label = FAILED (1)
       - No such termination found, AND enough time has passed (the
         follow-up window has closed) -> label = COMPLETED (0)
       - If the follow-up window hasn't closed yet (deal too recent),
         the deal is left UNLABELED and excluded.
  3. A handful of real, filing-derived features are extracted per deal:
     announcement date, days-to-resolution (if resolved), a crude
     cash-vs-stock flag from filing text keyword search, and filer SIC
     code (industry) -- NOT the full 52-variable financial-ratio feature
     set from Karatas & Hirsa (2021), since that requires FactSet-style
     proprietary financial data we don't have access to.

This is intentionally a SMALLER, HONEST feature set built from data we can
actually verify and cite the source of -- rather than a fabricated
approximation of the paper's real features.

SEC EDGAR fair-access rules: requests must include a descriptive
User-Agent header with a real contact (SEC blocks/rate-limits generic
scrapers). Set EDGAR_CONTACT_EMAIL before running.

Usage:
    export EDGAR_CONTACT_EMAIL="your_real_email@example.com"
    python edgar_dataset_builder.py --target-deals 200 --out ma_deals.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, UTC
from typing import Optional

import requests

FULLTEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
FILING_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{doc}"

FOLLOWUP_WINDOW_DAYS = 270  # ~9 months; most terminations/completions resolve within this window
REQUEST_DELAY_SECONDS = 0.3  # be polite to EDGAR's rate limits


def _headers() -> dict:
    contact = os.environ.get("EDGAR_CONTACT_EMAIL")
    if not contact:
        raise RuntimeError(
            "Set EDGAR_CONTACT_EMAIL to a real contact email before running "
            "-- SEC EDGAR requires a descriptive User-Agent for programmatic access."
        )
    return {"User-Agent": f"paper-to-experiment-pipeline research tool ({contact})"}


@dataclass
class DealRecord:
    company_name: str
    cik: str
    announcement_date: str
    resolution_date: Optional[str]
    days_to_resolution: Optional[int]
    label: Optional[int]  # 0 = completed, 1 = failed/terminated, None = unresolved (excluded)
    sic_code: Optional[str]
    mentions_cash: bool
    mentions_stock: bool
    source_accession: str


def search_merger_announcements(start_date: str, end_date: str, max_results: int = 400) -> list[dict]:
    """
    Query EDGAR full-text search for 8-K filings with Item 1.01 that mention
    'merger agreement'. Returns raw hit dicts from the search API.

    Pulls a BALANCED sample across each year in [start_date, end_date]
    rather than one continuous date-sorted pull -- sorting by most-recent
    date alone concentrates results into whichever narrow period the
    pagination happens to land on (e.g. late-2023 saw an unusually large
    wave of SPAC merger terminations, which skewed an earlier version of
    this function's results toward failure). Splitting by year avoids
    that kind of period-specific skew dominating the whole sample.
    """
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    years = list(range(start_year, end_year + 1))
    per_year_target = max(1, max_results // len(years))

    all_hits = []
    for year in years:
        year_start = f"{year}-01-01" if year != start_year else start_date
        year_end = f"{year}-12-31" if year != end_year else end_date
        year_hits = _search_one_range(year_start, year_end, per_year_target)
        print(f"  year {year}: collected {len(year_hits)} hits", flush=True)
        all_hits.extend(year_hits)

    all_hits.sort(key=lambda h: h.get("_source", {}).get("file_date", ""), reverse=True)
    return all_hits[:max_results]


def _search_one_range(start_date: str, end_date: str, max_results: int) -> list[dict]:
    hits = []
    frm = 0
    page_size = 100
    while len(hits) < max_results:
        params = {
            "q": '"merger agreement"',
            "forms": "8-K",
            "dateRange": "custom",
            "startdt": start_date,
            "enddt": end_date,
            "from": frm,
            "sort": "filedAt:desc",
        }
        print(f"    requesting EDGAR full-text search {start_date}..{end_date}, from={frm} ...", flush=True)
        try:
            resp = requests.get(FULLTEXT_SEARCH_URL, params=params, headers=_headers(), timeout=15)
            resp.raise_for_status()
        except requests.HTTPError as e:
            print(f"    EDGAR returned an error at from={frm} ({e}) -- stopping pagination here for this year.", flush=True)
            break
        data = resp.json()
        page_hits = data.get("hits", {}).get("hits", [])
        if not page_hits:
            break
        hits.extend(page_hits)
        frm += page_size
        time.sleep(REQUEST_DELAY_SECONDS)
        if frm >= data.get("hits", {}).get("total", {}).get("value", 0):
            break
    return hits[:max_results]


def get_company_filings(cik: str) -> dict:
    """Fetch a company's filing history from EDGAR's submissions API."""
    cik_padded = cik.zfill(10)
    resp = requests.get(SUBMISSIONS_URL.format(cik=cik_padded), headers=_headers(), timeout=15)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return resp.json()


def _find_termination_after(filings: dict, announce_date: datetime, window_days: int) -> Optional[str]:
    """
    Look through a company's recent filings for an 8-K filed after
    announce_date (within window_days) whose items include 1.02
    (termination). Returns the filing date string if found, else None.

    KNOWN LIMITATION (documented, not silently ignored): this checks for
    ANY Item 1.02 termination filed by the company in the window, not
    necessarily termination of the SAME merger agreement -- a company
    could terminate an unrelated lease, credit facility, or supply
    contract in that window and get wrongly counted as a failed deal.
    A stricter text-matching approach was tried and rejected: it produced
    worse false-positive rates (matching unrelated boilerplate mentions
    of "merger agreement" and "termination"), so this simpler, faster,
    but noisier heuristic was kept instead. The resulting ~40-45% failure
    rate in the labeled data is known to be inflated versus real-world
    M&A base rates (real failure rates are closer to 10-20%) -- this
    should be stated as a limitation wherever this dataset's labels are
    used, not treated as ground truth.
    """
    recent = filings.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    items = recent.get("items", [])  # e.g. "1.02,9.01"

    for form, date_str, item_str in zip(forms, dates, items):
        if form != "8-K":
            continue
        try:
            filing_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        if filing_date <= announce_date:
            continue
        if filing_date > announce_date + timedelta(days=window_days):
            continue
        if "1.02" in (item_str or ""):
            return date_str
    return None


def build_dataset(start_date: str, end_date: str, target_deals: int) -> list[DealRecord]:
    print(f"Searching EDGAR full-text search for merger-agreement 8-Ks ({start_date} to {end_date})...")
    hits = search_merger_announcements(start_date, end_date, max_results=target_deals * 3)
    print(f"Found {len(hits)} candidate announcement filings. Resolving outcomes...")

    records: list[DealRecord] = []
    cutoff_for_resolution = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=FOLLOWUP_WINDOW_DAYS)

    for i, hit in enumerate(hits):
        if len(records) >= target_deals:
            break
        src = hit.get("_source", {})
        cik_list = src.get("ciks", [])
        if not cik_list:
            continue
        cik = cik_list[0].lstrip("0") or "0"
        company_name = (src.get("display_names") or ["UNKNOWN"])[0]
        announce_date_str = src.get("file_date")
        if not announce_date_str:
            continue
        try:
            announce_date = datetime.strptime(announce_date_str, "%Y-%m-%d")
        except ValueError:
            continue

        # Only include deals old enough that the follow-up window has closed,
        # so we can confidently label completed vs. still-unresolved.
        if announce_date > cutoff_for_resolution:
            continue

        try:
            filings = get_company_filings(cik)
        except requests.HTTPError as e:
            print(f"  skipping CIK {cik} ({company_name}) -- EDGAR error: {e}", flush=True)
            continue
        except requests.Timeout:
            print(f"  skipping CIK {cik} ({company_name}) -- request timed out", flush=True)
            continue

        sic_code = filings.get("sicDescription")
        term_date = _find_termination_after(filings, announce_date, FOLLOWUP_WINDOW_DAYS)

        if term_date:
            label = 1
            resolution_date = term_date
            days_to_resolution = (datetime.strptime(term_date, "%Y-%m-%d") - announce_date).days
        else:
            label = 0
            resolution_date = None
            days_to_resolution = FOLLOWUP_WINDOW_DAYS  # treated as "survived the window"

        text_blob = " ".join(str(v) for v in src.values()).lower()
        records.append(DealRecord(
            company_name=company_name,
            cik=cik,
            announcement_date=announce_date_str,
            resolution_date=resolution_date,
            days_to_resolution=days_to_resolution,
            label=label,
            sic_code=sic_code,
            mentions_cash="cash" in text_blob,
            mentions_stock="stock" in text_blob or "shares" in text_blob,
            source_accession=hit.get("_id", ""),
        ))

        if (i + 1) % 10 == 0:
            print(f"  processed {i + 1}/{len(hits)} candidates, {len(records)} labeled deals so far")

    return records


def save_csv(records: list[DealRecord], path: str) -> None:
    if not records:
        print("No records to save.")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))
    n_failed = sum(1 for r in records if r.label == 1)
    print(f"Saved {len(records)} deals to {path} ({n_failed} failed / {len(records) - n_failed} completed)")

    readme_path = path.rsplit(".", 1)[0] + "_README.txt"
    with open(readme_path, "w") as f:
        f.write(
            "DATASET LIMITATIONS -- read before using these labels as ground truth:\n\n"
            "Labels come from a heuristic: an 8-K Item 1.02 (termination) filed by the\n"
            "same company within ~9 months of the merger-agreement announcement is\n"
            "labeled 'failed'. This does NOT verify the termination is of the SAME\n"
            "merger agreement -- it could be an unrelated contract termination\n"
            "(lease, credit facility, supply agreement, etc.) in the same window.\n\n"
            f"Resulting failure rate in this file: {n_failed}/{len(records)} "
            f"({100*n_failed/len(records):.0f}%) -- this is inflated versus real-world\n"
            "M&A base rates (~10-20% failure), so this dataset should be treated as a\n"
            "noisy real-data proxy for pipeline validation, not a clean ground-truth\n"
            "benchmark. State this limitation wherever these labels are used.\n"
        )
    print(f"Wrote limitations note to {readme_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2023-12-31")
    parser.add_argument("--target-deals", type=int, default=200)
    parser.add_argument("--out", default="ma_deals.csv")
    args = parser.parse_args()

    records = build_dataset(args.start_date, args.end_date, args.target_deals)
    save_csv(records, args.out)
