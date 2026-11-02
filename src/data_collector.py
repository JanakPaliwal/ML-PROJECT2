"""
data_collector.py
Data Ingestion and Harvesting Module for Agmarknet / OGD India Platform.
Collects daily mandi modal prices, arrival volumes, and grade metrics for Onion and Tomato.
Supports:
1. data.gov.in OGD API (Resource: Current Daily Price of Various Commodities from Various Markets)
2. agmarknet.gov.in date-range CSV / Tabular report importer
"""

import os
import sys
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import urllib.request
import urllib.parse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Agmarknet OGD India Resource Configuration
OGD_BASE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
ALT_OGD_URL = "https://api.data.gov.in/resource/current-daily-price-various-commodities-various-markets-mandi"

DEFAULT_MANDIS = {
    "Onion": ["Lasalgaon", "Pimpalgaon", "Azadpur", "Solapur", "Nashik"],
    "Tomato": ["Kolar", "Pimpalgaon", "Azadpur", "Madanapalle", "Nashik"]
}

class AgmarknetCollector:
    """
    Client for fetching and storing Agmarknet wholesale mandi price and arrival records.
    Licence: GODL-India (Government Open Data License - India).
    """

    def __init__(self, api_key: Optional[str] = None, data_dir: str = "data/raw"):
        self.api_key = api_key or os.environ.get("OGD_API_KEY", "")
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_daily_api_records(
        self,
        commodity: str = "Onion",
        market: Optional[str] = None,
        state: Optional[str] = None,
        limit: int = 500,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Pull real-time mandi records from data.gov.in API.
        Requires free API key from https://data.gov.in.
        """
        if not self.api_key:
            logger.warning(
                "No OGD_API_KEY provided. Set OGD_API_KEY environment variable or pass api_key "
                "to AgmarknetCollector to connect to live data.gov.in endpoint."
            )
            return {"status": "error", "message": "API key required for live data.gov.in access", "records": []}

        params = {
            "api-key": self.api_key,
            "format": "json",
            "offset": offset,
            "limit": limit,
            "filters[commodity]": commodity
        }
        if market:
            params["filters[market]"] = market
        if state:
            params["filters[state]"] = state

        query_str = urllib.parse.urlencode(params)
        req_url = f"{OGD_BASE_URL}?{query_str}"

        try:
            logger.info(f"Querying OGD API for commodity: {commodity}, market: {market or 'All'}")
            req = urllib.request.Request(req_url, headers={"User-Agent": "MandiForecastingResearch/1.0"})
            with urllib.request.urlopen(req, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
                records = payload.get("records", [])
                logger.info(f"Successfully retrieved {len(records)} records from OGD API.")
                return {"status": "success", "count": len(records), "records": records}
        except Exception as e:
            logger.error(f"Failed to fetch records from OGD API: {e}")
            return {"status": "error", "message": str(e), "records": []}

    def parse_agmarknet_csv(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Parse CSV downloaded directly from https://agmarknet.gov.in date-range reports.
        Normalizes varying column headers to canonical schema:
        ['date', 'mandi', 'state', 'commodity', 'variety', 'arrivals_tonnes', 'min_price', 'max_price', 'modal_price']
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Agmarknet file not found at: {filepath}")

        records = []
        import csv
        with open(filepath, mode="r", encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.DictReader(f)
            # Normalize fieldnames by stripping and lowercasing
            clean_fields = {k: k.strip().lower() for k in (reader.fieldnames or [])}
            
            for row in reader:
                norm_row = {clean_fields.get(k, k): v.strip() for k, v in row.items()}
                
                # Extract date
                raw_date = norm_row.get("arrival_date") or norm_row.get("date") or norm_row.get("reported date", "")
                parsed_date = self._normalize_date(raw_date)
                if not parsed_date:
                    continue

                # Parse prices (Rs per Quintal)
                modal = self._safe_float(norm_row.get("modal_price") or norm_row.get("modal_x0020_price") or norm_row.get("modal price"))
                min_p = self._safe_float(norm_row.get("min_price") or norm_row.get("min_x0020_price") or norm_row.get("min price"))
                max_p = self._safe_float(norm_row.get("max_price") or norm_row.get("max_x0020_price") or norm_row.get("max price"))
                arrivals = self._safe_float(norm_row.get("arrivals") or norm_row.get("arrivals (tonnes)") or norm_row.get("arrival"))

                mandi = norm_row.get("market") or norm_row.get("mandi") or norm_row.get("market center", "Unknown")
                state = norm_row.get("state") or norm_row.get("state name", "Unknown")
                comm = norm_row.get("commodity", "Unknown")

                records.append({
                    "date": parsed_date,
                    "mandi": mandi,
                    "state": state,
                    "commodity": comm,
                    "variety": norm_row.get("variety", "Common"),
                    "arrivals_tonnes": arrivals if arrivals is not None else 100.0,
                    "min_price": min_p if min_p is not None else modal,
                    "max_price": max_p if max_p is not None else modal,
                    "modal_price": modal
                })

        logger.info(f"Parsed {len(records)} records from {filepath}")
        return records

    @staticmethod
    def _normalize_date(date_str: str) -> Optional[str]:
        if not date_str:
            return None
        formats = [
            "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%b/%Y", "%d-%b-%Y",
            "%d-%B-%Y", "%m/%d/%Y", "%Y/%m/%d"
        ]
        date_str = date_str.strip().split(" ")[0]  # Remove any trailing time
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None

    @staticmethod
    def _safe_float(val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            s = str(val).replace(",", "").replace("Rs.", "").replace("₹", "").strip()
            if not s or s.lower() in ("nan", "null", "nil", "-"):
                return None
            return float(s)
        except ValueError:
            return None

if __name__ == "__main__":
    collector = AgmarknetCollector()
    print("AgmarknetCollector initialized.")
