"""
Ticker universe: a 281-name large-cap core (11 GICS sectors) plus the S&P
MidCap 400 / SmallCap 600 members (load_universe).

- The core list was cross-checked against Wikipedia's S&P 500 list
  (2026-09-18); the rest is general knowledge. A "no price history" flag in
  data.data_quality_report usually means a completed M&A or a ticker change
  (BK -> BNY, FI -> FISV), so check that before assuming a bug.
- Energy, Materials, Communication Services and Real Estate have fewer names:
  there are fewer long-listed large companies, and padding with recent IPOs
  adds almost no Train-period data. Partial histories are simply NaN early on.
- A few awkward names (meme stocks GME/AMC/KOSS/BB, "value traps" T/VZ/MO...)
  are kept on purpose as real cases for screening's meme / value-trap flags.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd

UNIVERSE: dict[str, dict[str, str]] = {
    # ---- Technology (34) ----
    "AAPL": {"name": "Apple", "sector": "Technology"},
    "MSFT": {"name": "Microsoft", "sector": "Technology"},
    "ORCL": {"name": "Oracle", "sector": "Technology"},
    "IBM": {"name": "IBM", "sector": "Technology"},
    "INTC": {"name": "Intel", "sector": "Technology"},
    "CSCO": {"name": "Cisco Systems", "sector": "Technology"},
    "NVDA": {"name": "Nvidia", "sector": "Technology"},
    "ADBE": {"name": "Adobe", "sector": "Technology"},
    "INTU": {"name": "Intuit", "sector": "Technology"},
    "SNPS": {"name": "Synopsys", "sector": "Technology"},
    "CDNS": {"name": "Cadence Design Systems", "sector": "Technology"},
    "QCOM": {"name": "Qualcomm", "sector": "Technology"},
    "AMD": {"name": "Advanced Micro Devices", "sector": "Technology"},
    "TXN": {"name": "Texas Instruments", "sector": "Technology"},
    "MCHP": {"name": "Microchip Technology", "sector": "Technology"},
    "MU": {"name": "Micron Technology", "sector": "Technology"},
    "LRCX": {"name": "Lam Research", "sector": "Technology"},
    "KLAC": {"name": "KLA Corporation", "sector": "Technology"},
    "AMAT": {"name": "Applied Materials", "sector": "Technology"},
    "HPQ": {"name": "HP Inc.", "sector": "Technology"},
    "XRX": {"name": "Xerox", "sector": "Technology"},
    "NTAP": {"name": "NetApp", "sector": "Technology"},
    "WDC": {"name": "Western Digital", "sector": "Technology"},
    "STX": {"name": "Seagate Technology", "sector": "Technology"},
    "ADSK": {"name": "Autodesk", "sector": "Technology"},
    "CTSH": {"name": "Cognizant", "sector": "Technology"},
    "ACN": {"name": "Accenture", "sector": "Technology"},
    "FISV": {"name": "Fiserv", "sector": "Technology"},  # FISV -> FI (2023) -> back to FISV
    "TER": {"name": "Teradyne", "sector": "Technology"},
    "ADI": {"name": "Analog Devices", "sector": "Technology"},
    "TYL": {"name": "Tyler Technologies", "sector": "Technology"},
    "BB": {"name": "BlackBerry", "sector": "Technology"},  # meme-stock watchlist, see bottom
    "MSI": {"name": "Motorola Solutions", "sector": "Technology"},
    "GLW": {"name": "Corning", "sector": "Technology"},

    # ---- Healthcare (29) ----
    "JNJ": {"name": "Johnson & Johnson", "sector": "Healthcare"},
    "PFE": {"name": "Pfizer", "sector": "Healthcare"},
    "ABBV": {"name": "AbbVie", "sector": "Healthcare"},
    "MRK": {"name": "Merck", "sector": "Healthcare"},
    "UNH": {"name": "UnitedHealth Group", "sector": "Healthcare"},
    "ABT": {"name": "Abbott Laboratories", "sector": "Healthcare"},
    "LLY": {"name": "Eli Lilly", "sector": "Healthcare"},
    "TMO": {"name": "Thermo Fisher Scientific", "sector": "Healthcare"},
    "DHR": {"name": "Danaher", "sector": "Healthcare"},
    "CVS": {"name": "CVS Health", "sector": "Healthcare"},
    "GILD": {"name": "Gilead Sciences", "sector": "Healthcare"},
    "BIIB": {"name": "Biogen", "sector": "Healthcare"},
    "AMGN": {"name": "Amgen", "sector": "Healthcare"},
    "REGN": {"name": "Regeneron Pharmaceuticals", "sector": "Healthcare"},
    "VRTX": {"name": "Vertex Pharmaceuticals", "sector": "Healthcare"},
    "ISRG": {"name": "Intuitive Surgical", "sector": "Healthcare"},
    "MCK": {"name": "McKesson", "sector": "Healthcare"},
    "CAH": {"name": "Cardinal Health", "sector": "Healthcare"},
    "BAX": {"name": "Baxter International", "sector": "Healthcare"},
    "BDX": {"name": "Becton Dickinson", "sector": "Healthcare"},
    "MDT": {"name": "Medtronic", "sector": "Healthcare"},
    "SYK": {"name": "Stryker", "sector": "Healthcare"},
    "ZBH": {"name": "Zimmer Biomet", "sector": "Healthcare"},
    "BSX": {"name": "Boston Scientific", "sector": "Healthcare"},
    "HUM": {"name": "Humana", "sector": "Healthcare"},
    "CI": {"name": "Cigna", "sector": "Healthcare"},
    "ELV": {"name": "Elevance Health", "sector": "Healthcare"},  # formerly Anthem
    "COR": {"name": "Cencora", "sector": "Healthcare"},  # renamed from ABC (AmerisourceBergen) in 2023
    "COO": {"name": "Cooper Companies", "sector": "Healthcare"},

    # ---- Financials (29) ----
    "JPM": {"name": "JPMorgan Chase", "sector": "Financials"},
    "BAC": {"name": "Bank of America", "sector": "Financials"},
    "WFC": {"name": "Wells Fargo", "sector": "Financials"},
    "C": {"name": "Citigroup", "sector": "Financials"},
    "USB": {"name": "U.S. Bancorp", "sector": "Financials"},
    "GS": {"name": "Goldman Sachs", "sector": "Financials"},
    "MS": {"name": "Morgan Stanley", "sector": "Financials"},
    "BLK": {"name": "BlackRock", "sector": "Financials"},
    "AXP": {"name": "American Express", "sector": "Financials"},
    "COF": {"name": "Capital One", "sector": "Financials"},
    "SCHW": {"name": "Charles Schwab", "sector": "Financials"},
    "CME": {"name": "CME Group", "sector": "Financials"},
    "NDAQ": {"name": "Nasdaq, Inc.", "sector": "Financials"},
    "TRV": {"name": "Travelers", "sector": "Financials"},
    "AIG": {"name": "American International Group", "sector": "Financials"},
    "MET": {"name": "MetLife", "sector": "Financials"},
    "PRU": {"name": "Prudential Financial", "sector": "Financials"},
    "ALL": {"name": "Allstate", "sector": "Financials"},
    "PNC": {"name": "PNC Financial Services", "sector": "Financials"},
    "BNY": {"name": "Bank of New York Mellon", "sector": "Financials"},  # renamed from BK in 2025; Finnhub PER/PBR under BNY look broken (~0.04/0.006) — excluded by the fair-value bounds, see data_quality_report
    "STT": {"name": "State Street", "sector": "Financials"},
    "FITB": {"name": "Fifth Third Bancorp", "sector": "Financials"},
    "RF": {"name": "Regions Financial", "sector": "Financials"},
    "KEY": {"name": "KeyCorp", "sector": "Financials"},
    "HBAN": {"name": "Huntington Bancshares", "sector": "Financials"},
    "ZION": {"name": "Zions Bancorporation", "sector": "Financials"},
    "CINF": {"name": "Cincinnati Financial", "sector": "Financials"},
    "AFL": {"name": "Aflac", "sector": "Financials"},
    "L": {"name": "Loews Corporation", "sector": "Financials"},

    # ---- Consumer Staples (27) ----
    "PG": {"name": "Procter & Gamble", "sector": "Consumer Staples"},
    "KO": {"name": "Coca-Cola", "sector": "Consumer Staples"},
    "PEP": {"name": "PepsiCo", "sector": "Consumer Staples"},
    "CL": {"name": "Colgate-Palmolive", "sector": "Consumer Staples"},
    "MO": {"name": "Altria Group", "sector": "Consumer Staples"},  # value-trap watchlist, see bottom
    "KMB": {"name": "Kimberly-Clark", "sector": "Consumer Staples"},
    "WMT": {"name": "Walmart", "sector": "Consumer Staples"},
    "KR": {"name": "Kroger", "sector": "Consumer Staples"},
    "COST": {"name": "Costco", "sector": "Consumer Staples"},
    "ADM": {"name": "Archer-Daniels-Midland", "sector": "Consumer Staples"},
    "GIS": {"name": "General Mills", "sector": "Consumer Staples"},
    "HSY": {"name": "Hershey", "sector": "Consumer Staples"},
    "SJM": {"name": "J.M. Smucker", "sector": "Consumer Staples"},
    "TSN": {"name": "Tyson Foods", "sector": "Consumer Staples"},
    "STZ": {"name": "Constellation Brands", "sector": "Consumer Staples"},
    "TAP": {"name": "Molson Coors", "sector": "Consumer Staples"},
    "DLTR": {"name": "Dollar Tree", "sector": "Consumer Staples"},
    "SYY": {"name": "Sysco", "sector": "Consumer Staples"},
    "CLX": {"name": "Clorox", "sector": "Consumer Staples"},
    "CHD": {"name": "Church & Dwight", "sector": "Consumer Staples"},
    "CAG": {"name": "Conagra Brands", "sector": "Consumer Staples"},
    "HRL": {"name": "Hormel Foods", "sector": "Consumer Staples"},
    "MKC": {"name": "McCormick & Company", "sector": "Consumer Staples"},
    "PM": {"name": "Philip Morris International", "sector": "Consumer Staples"},  # value-trap watchlist
    "DG": {"name": "Dollar General", "sector": "Consumer Staples"},
    "MDLZ": {"name": "Mondelez International", "sector": "Consumer Staples"},
    "CPB": {"name": "Campbell's Company", "sector": "Consumer Staples"},

    # ---- Industrials (32) ----
    "HON": {"name": "Honeywell", "sector": "Industrials"},
    "CAT": {"name": "Caterpillar", "sector": "Industrials"},
    "UPS": {"name": "United Parcel Service", "sector": "Industrials"},
    "GE": {"name": "General Electric", "sector": "Industrials"},
    "MMM": {"name": "3M", "sector": "Industrials"},
    "BA": {"name": "Boeing", "sector": "Industrials"},
    "DE": {"name": "Deere & Company", "sector": "Industrials"},
    "RTX": {"name": "RTX Corporation", "sector": "Industrials"},  # Raytheon/United Technologies merger, 2020
    "LMT": {"name": "Lockheed Martin", "sector": "Industrials"},
    "NOC": {"name": "Northrop Grumman", "sector": "Industrials"},
    "GD": {"name": "General Dynamics", "sector": "Industrials"},
    "ETN": {"name": "Eaton Corporation", "sector": "Industrials"},
    "EMR": {"name": "Emerson Electric", "sector": "Industrials"},
    "ITW": {"name": "Illinois Tool Works", "sector": "Industrials"},
    "RSG": {"name": "Republic Services", "sector": "Industrials"},
    "WM": {"name": "Waste Management", "sector": "Industrials"},
    "FDX": {"name": "FedEx", "sector": "Industrials"},
    "DAL": {"name": "Delta Air Lines", "sector": "Industrials"},  # 2007 bankruptcy reorg — history discontinuity
    "LUV": {"name": "Southwest Airlines", "sector": "Industrials"},
    "CSX": {"name": "CSX Corporation", "sector": "Industrials"},
    "NSC": {"name": "Norfolk Southern", "sector": "Industrials"},
    "PCAR": {"name": "PACCAR", "sector": "Industrials"},
    "ROK": {"name": "Rockwell Automation", "sector": "Industrials"},
    "CMI": {"name": "Cummins", "sector": "Industrials"},
    "PH": {"name": "Parker Hannifin", "sector": "Industrials"},
    "DOV": {"name": "Dover Corporation", "sector": "Industrials"},
    "AME": {"name": "AMETEK", "sector": "Industrials"},
    "FAST": {"name": "Fastenal", "sector": "Industrials"},
    "PWR": {"name": "Quanta Services", "sector": "Industrials"},
    "J": {"name": "Jacobs Solutions", "sector": "Industrials"},
    "EFX": {"name": "Equifax", "sector": "Industrials"},
    "NDSN": {"name": "Nordson Corporation", "sector": "Industrials"},

    # ---- Energy (15) ----
    "XOM": {"name": "ExxonMobil", "sector": "Energy"},
    "CVX": {"name": "Chevron", "sector": "Energy"},
    "COP": {"name": "ConocoPhillips", "sector": "Energy"},
    "SLB": {"name": "Schlumberger", "sector": "Energy"},
    "OXY": {"name": "Occidental Petroleum", "sector": "Energy"},
    "VLO": {"name": "Valero Energy", "sector": "Energy"},
    "HAL": {"name": "Halliburton", "sector": "Energy"},
    "EOG": {"name": "EOG Resources", "sector": "Energy"},
    "OKE": {"name": "ONEOK", "sector": "Energy"},
    "DVN": {"name": "Devon Energy", "sector": "Energy"},
    "EQT": {"name": "EQT Corporation", "sector": "Energy"},
    "APA": {"name": "APA Corporation", "sector": "Energy"},  # formerly Apache
    "WMB": {"name": "Williams Companies", "sector": "Energy"},
    "NOV": {"name": "NOV Inc.", "sector": "Energy"},
    "BKR": {"name": "Baker Hughes", "sector": "Energy"},

    # ---- Materials (18) ----
    "LIN": {"name": "Linde", "sector": "Materials"},
    "SHW": {"name": "Sherwin-Williams", "sector": "Materials"},
    "APD": {"name": "Air Products and Chemicals", "sector": "Materials"},
    "ECL": {"name": "Ecolab", "sector": "Materials"},
    "NUE": {"name": "Nucor", "sector": "Materials"},
    "PPG": {"name": "PPG Industries", "sector": "Materials"},
    "FCX": {"name": "Freeport-McMoRan", "sector": "Materials"},
    "VMC": {"name": "Vulcan Materials Company", "sector": "Materials"},
    "MLM": {"name": "Martin Marietta Materials", "sector": "Materials"},
    "IP": {"name": "International Paper", "sector": "Materials"},
    "PKG": {"name": "Packaging Corporation of America", "sector": "Materials"},
    "IFF": {"name": "International Flavors & Fragrances", "sector": "Materials"},
    "NEM": {"name": "Newmont Corporation", "sector": "Materials"},
    "ALB": {"name": "Albemarle Corporation", "sector": "Materials"},
    "EMN": {"name": "Eastman Chemical Company", "sector": "Materials"},
    "AVY": {"name": "Avery Dennison", "sector": "Materials"},
    "BALL": {"name": "Ball Corporation", "sector": "Materials"},
    "RPM": {"name": "RPM International", "sector": "Materials"},

    # ---- Consumer Discretionary (37) ----
    "AMZN": {"name": "Amazon", "sector": "Consumer Discretionary"},
    "HD": {"name": "Home Depot", "sector": "Consumer Discretionary"},
    "MCD": {"name": "McDonald's", "sector": "Consumer Discretionary"},
    "NKE": {"name": "Nike", "sector": "Consumer Discretionary"},
    "SBUX": {"name": "Starbucks", "sector": "Consumer Discretionary"},
    "LOW": {"name": "Lowe's", "sector": "Consumer Discretionary"},
    "TSLA": {"name": "Tesla", "sector": "Consumer Discretionary"},  # 2010 IPO — short Train history, kept for weight
    "F": {"name": "Ford Motor Company", "sector": "Consumer Discretionary"},  # value-trap watchlist
    "GM": {"name": "General Motors", "sector": "Consumer Discretionary"},  # 2010 post-bankruptcy re-IPO — discontinuity
    "LEN": {"name": "Lennar Corporation", "sector": "Consumer Discretionary"},
    "DHI": {"name": "D.R. Horton", "sector": "Consumer Discretionary"},
    "RCL": {"name": "Royal Caribbean Group", "sector": "Consumer Discretionary"},
    "CCL": {"name": "Carnival Corporation", "sector": "Consumer Discretionary"},
    "MAR": {"name": "Marriott International", "sector": "Consumer Discretionary"},
    "EXPE": {"name": "Expedia Group", "sector": "Consumer Discretionary"},
    "TJX": {"name": "TJX Companies", "sector": "Consumer Discretionary"},
    "DRI": {"name": "Darden Restaurants", "sector": "Consumer Discretionary"},
    "EBAY": {"name": "eBay", "sector": "Consumer Discretionary"},
    "ROST": {"name": "Ross Stores", "sector": "Consumer Discretionary"},
    "YUM": {"name": "Yum! Brands", "sector": "Consumer Discretionary"},
    "BKNG": {"name": "Booking Holdings", "sector": "Consumer Discretionary"},  # formerly Priceline (PCLN)
    "ORLY": {"name": "O'Reilly Automotive", "sector": "Consumer Discretionary"},
    "AZO": {"name": "AutoZone", "sector": "Consumer Discretionary"},
    "BBY": {"name": "Best Buy", "sector": "Consumer Discretionary"},
    "GPC": {"name": "Genuine Parts Company", "sector": "Consumer Discretionary"},
    "TGT": {"name": "Target Corporation", "sector": "Consumer Discretionary"},
    "KMX": {"name": "CarMax", "sector": "Consumer Discretionary"},
    "WHR": {"name": "Whirlpool Corporation", "sector": "Consumer Discretionary"},
    "NVR": {"name": "NVR, Inc.", "sector": "Consumer Discretionary"},
    "PHM": {"name": "PulteGroup", "sector": "Consumer Discretionary"},
    "VFC": {"name": "VF Corporation", "sector": "Consumer Discretionary"},
    "HAS": {"name": "Hasbro", "sector": "Consumer Discretionary"},
    "MAT": {"name": "Mattel", "sector": "Consumer Discretionary"},
    "MHK": {"name": "Mohawk Industries", "sector": "Consumer Discretionary"},
    "GME": {"name": "GameStop", "sector": "Consumer Discretionary"},  # meme-stock watchlist, see bottom
    "KOSS": {"name": "Koss Corporation", "sector": "Consumer Discretionary"},  # meme-stock watchlist — NOT in a major index
    "M": {"name": "Macy's", "sector": "Consumer Discretionary"},  # value-trap watchlist

    # ---- Communication Services (13) ----
    "GOOGL": {"name": "Alphabet", "sector": "Communication Services"},
    "DIS": {"name": "Disney", "sector": "Communication Services"},
    "VZ": {"name": "Verizon", "sector": "Communication Services"},  # value-trap watchlist
    "T": {"name": "AT&T", "sector": "Communication Services"},  # value-trap watchlist
    "CMCSA": {"name": "Comcast", "sector": "Communication Services"},
    "META": {"name": "Meta Platforms", "sector": "Communication Services"},  # 2012 IPO — short Train history, kept for weight
    "NFLX": {"name": "Netflix", "sector": "Communication Services"},
    "EA": {"name": "Electronic Arts", "sector": "Communication Services"},
    "TTWO": {"name": "Take-Two Interactive", "sector": "Communication Services"},
    "OMC": {"name": "Omnicom Group", "sector": "Communication Services"},
    "NYT": {"name": "New York Times Company", "sector": "Communication Services"},  # not in a major index
    "LUMN": {"name": "Lumen Technologies", "sector": "Communication Services"},  # value-trap watchlist, not in a major index
    "AMC": {"name": "AMC Entertainment", "sector": "Communication Services"},  # meme-stock watchlist, not in a major index

    # ---- Utilities (25) ----
    "NEE": {"name": "NextEra Energy", "sector": "Utilities"},
    "DUK": {"name": "Duke Energy", "sector": "Utilities"},
    "SO": {"name": "Southern Company", "sector": "Utilities"},
    "D": {"name": "Dominion Energy", "sector": "Utilities"},
    "AEP": {"name": "American Electric Power", "sector": "Utilities"},
    "EXC": {"name": "Exelon", "sector": "Utilities"},
    "ETR": {"name": "Entergy", "sector": "Utilities"},
    "ED": {"name": "Consolidated Edison", "sector": "Utilities"},
    "DTE": {"name": "DTE Energy", "sector": "Utilities"},
    "EIX": {"name": "Edison International", "sector": "Utilities"},
    "ES": {"name": "Eversource Energy", "sector": "Utilities"},
    "AES": {"name": "AES Corporation", "sector": "Utilities"},
    "WEC": {"name": "WEC Energy Group", "sector": "Utilities"},
    "XEL": {"name": "Xcel Energy", "sector": "Utilities"},
    "PNW": {"name": "Pinnacle West Capital", "sector": "Utilities"},
    "LNT": {"name": "Alliant Energy", "sector": "Utilities"},
    "AEE": {"name": "Ameren", "sector": "Utilities"},
    "CNP": {"name": "CenterPoint Energy", "sector": "Utilities"},
    "PPL": {"name": "PPL Corporation", "sector": "Utilities"},
    "FE": {"name": "FirstEnergy", "sector": "Utilities"},
    "PEG": {"name": "Public Service Enterprise Group", "sector": "Utilities"},
    "NI": {"name": "NiSource", "sector": "Utilities"},
    "ATO": {"name": "Atmos Energy", "sector": "Utilities"},
    "CMS": {"name": "CMS Energy", "sector": "Utilities"},
    "SRE": {"name": "Sempra", "sector": "Utilities"},

    # ---- Real Estate (22) ----
    "PLD": {"name": "Prologis", "sector": "Real Estate"},
    "AMT": {"name": "American Tower", "sector": "Real Estate"},
    "SPG": {"name": "Simon Property Group", "sector": "Real Estate"},
    "O": {"name": "Realty Income", "sector": "Real Estate"},
    "PSA": {"name": "Public Storage", "sector": "Real Estate"},
    "VTR": {"name": "Ventas", "sector": "Real Estate"},
    "EQR": {"name": "Equity Residential", "sector": "Real Estate"},
    "AVB": {"name": "AvalonBay Communities", "sector": "Real Estate"},
    "CBRE": {"name": "CBRE Group", "sector": "Real Estate"},
    "CCI": {"name": "Crown Castle", "sector": "Real Estate"},
    "EQIX": {"name": "Equinix", "sector": "Real Estate"},
    "DLR": {"name": "Digital Realty Trust", "sector": "Real Estate"},
    "WELL": {"name": "Welltower", "sector": "Real Estate"},
    "BXP": {"name": "BXP, Inc.", "sector": "Real Estate"},  # formerly Boston Properties
    "ARE": {"name": "Alexandria Real Estate Equities", "sector": "Real Estate"},
    "ESS": {"name": "Essex Property Trust", "sector": "Real Estate"},
    "MAA": {"name": "Mid-America Apartment Communities", "sector": "Real Estate"},
    "UDR": {"name": "UDR, Inc.", "sector": "Real Estate"},
    "HST": {"name": "Host Hotels & Resorts", "sector": "Real Estate"},
    "REG": {"name": "Regency Centers", "sector": "Real Estate"},
    "FRT": {"name": "Federal Realty Investment Trust", "sector": "Real Estate"},
    "KIM": {"name": "Kimco Realty", "sector": "Real Estate"},
}

TICKERS: list[str] = list(UNIVERSE.keys())

SECTORS: list[str] = sorted({meta["sector"] for meta in UNIVERSE.values()})


# ---- Mid/small-cap extension (2026-09-29) ---------------------------------
# Wikipedia constituent tables, fetched on the first build and frozen as CSV
# under <cache_dir>/universe/ (delete to refresh). Today's members only, so a
# return test on small caps is survivorship-biased toward "cheap did well".
# size_group comes from index membership (market cap, i.e. price): reports only,
# never a model input.
INDEX_PAGES = {
    "sp400": ("mid", "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"),
    "sp600": ("small", "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"),
}
# Wikipedia uses GICS names; the core list uses these two shorter ones.
_GICS_TO_SECTOR = {"Information Technology": "Technology", "Health Care": "Healthcare"}
# Mortgage REITs Wikipedia still lists under Real Estate (GICS moved them to
# Financials in 2023); their "sales", EBITDA and FCF are interest flows.
_SECTOR_OVERRIDES = {"ADAM": "Financials", "ARR": "Financials", "FBRT": "Financials", "PMT": "Financials"}


def parse_constituents(html: str) -> pd.DataFrame:
    """ticker / name / sector from the first table on a Wikipedia index page
    that has a symbol and a GICS sector column. Tickers keep Wikipedia's
    class-share dot (MOG.A; data.py maps it for yfinance)."""
    for table in pd.read_html(StringIO(html)):
        cols = {str(c).strip().lower(): c for c in table.columns}
        symbol = next((cols[c] for c in cols if c in ("symbol", "ticker", "ticker symbol")), None)
        sector = next((cols[c] for c in cols if "sector" in c), None)
        name = next((cols[c] for c in cols if c in ("security", "company", "name")), None)
        if symbol is None or sector is None:
            continue
        out = pd.DataFrame({
            "ticker": table[symbol].astype(str).str.strip().str.upper(),
            "name": table[name].astype(str).str.strip() if name is not None else "",
            "sector": table[sector].astype(str).str.strip().replace(_GICS_TO_SECTOR),
        })
        return out[out["ticker"].str.match(r"^[A-Z][A-Z0-9.\-]*$")].drop_duplicates("ticker").reset_index(drop=True)
    raise ValueError("no constituents table (symbol + sector columns) found")


def _index_members(index: str, cache_dir: Path) -> pd.DataFrame:
    path = Path(cache_dir) / "universe" / f"{index}.csv"
    if path.exists():
        return pd.read_csv(path)
    import requests

    _, url = INDEX_PAGES[index]
    response = requests.get(url, headers={"User-Agent": "stock-valuation-model/1.0 (research)"}, timeout=30)
    response.raise_for_status()
    members = parse_constituents(response.text).assign(fetched=pd.Timestamp.today().date().isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    members.to_csv(path, index=False)
    print(f"  {index}: {len(members)} constituents fetched from Wikipedia -> {path}")
    return members


def load_universe(cache_dir: str | Path, indexes: tuple[str, ...] = tuple(INDEX_PAGES)) -> dict[str, dict[str, str]]:
    """The core UNIVERSE (size_group "large") plus each index's members.
    A ticker in both keeps its core sector label but takes the index's
    size_group. _SECTOR_OVERRIDES corrects sector labels last."""
    universe = {t: {**meta, "size": "large"} for t, meta in UNIVERSE.items()}
    for index in indexes:
        size, _ = INDEX_PAGES[index]
        for row in _index_members(index, Path(cache_dir)).itertuples():
            base = universe.get(row.ticker, {"name": row.name, "sector": row.sector})
            universe[row.ticker] = {**base, "size": size}
    for ticker, sector in _SECTOR_OVERRIDES.items():
        if ticker in universe:
            universe[ticker] = {**universe[ticker], "sector": sector}
    return universe


# GICS sub-industry per ticker (2026-09-30): sectors are broad (hardware and
# software, airlines and defense, mortgage REITs and banks share one), so
# sub-industries that are cheap by nature land in 저평가 together. Fetched
# once from the three Wikipedia index pages — the S&P 500 page covers most of
# the core list — and frozen like the constituents. Membership is NOT taken
# from these pages; only the ticker -> sub-industry map is.
SUB_INDUSTRY_PAGES = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    **{k: url for k, (_, url) in INDEX_PAGES.items()},
}


def parse_sub_industries(html: str) -> dict[str, str]:
    """ticker -> GICS sub-industry from the first table with a symbol and a
    sub-industry column."""
    for table in pd.read_html(StringIO(html)):
        cols = {str(c).strip().lower(): c for c in table.columns}
        symbol = next((cols[c] for c in cols if c in ("symbol", "ticker", "ticker symbol")), None)
        sub = next((cols[c] for c in cols if "sub-industry" in c or "sub industry" in c), None)
        if symbol is None or sub is None:
            continue
        tickers = table[symbol].astype(str).str.strip().str.upper()
        return dict(zip(tickers, table[sub].astype(str).str.strip()))
    return {}


def load_sub_industries(cache_dir: str | Path) -> dict[str, str]:
    """ticker -> GICS sub-industry, from <cache_dir>/universe/sub_industry.csv
    (fetched once from Wikipedia; delete the file to refresh)."""
    path = Path(cache_dir) / "universe" / "sub_industry.csv"
    if path.exists():
        df = pd.read_csv(path)
        return dict(zip(df["ticker"], df["sub_industry"]))
    import requests

    mapping: dict[str, str] = {}
    for url in SUB_INDUSTRY_PAGES.values():
        response = requests.get(url, headers={"User-Agent": "stock-valuation-model/1.0 (research)"}, timeout=30)
        response.raise_for_status()
        for ticker, sub in parse_sub_industries(response.text).items():
            mapping.setdefault(ticker, sub)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": list(mapping), "sub_industry": list(mapping.values())}).to_csv(path, index=False)
    print(f"  sub-industries: {len(mapping)} tickers fetched from Wikipedia -> {path}")
    return mapping


def industry_groups(universe: dict[str, dict[str, str]], sub_industries: dict[str, str],
                    min_tickers: int) -> dict[str, str]:
    """ticker -> industry group: its GICS sub-industry when at least
    `min_tickers` universe members share it, else "<sector> 기타" (a
    sub-industry of 2-3 stocks would give every one of them its own
    intercept). Tickers without a sub-industry also fall back to the sector."""
    subs = {t: sub_industries.get(t) for t in universe}
    counts = pd.Series([s for s in subs.values() if s]).value_counts()
    return {t: s if s and counts.get(s, 0) >= min_tickers else f"{universe[t]['sector']} 기타"
            for t, s in subs.items()}


def tickers_by_sector(sector: str) -> list[str]:
    """All tickers assigned to one sector — mainly useful for sanity-checking
    the universe (e.g. confirming each sector's size)."""
    return [ticker for ticker, meta in UNIVERSE.items() if meta["sector"] == sector]


if __name__ == "__main__":
    # Quick sanity check when run directly: confirms sector sizes and total
    # count before this feeds into data collection.
    print(f"{len(TICKERS)} tickers across {len(SECTORS)} sectors\n")
    for sector in SECTORS:
        members = tickers_by_sector(sector)
        print(f"{sector:26s} ({len(members)}): {', '.join(members)}")
    assert len(TICKERS) == len(set(TICKERS)), "duplicate ticker in UNIVERSE"
