"""
Ticker universe (281 tickers, 11 GICS sectors).

Sector is the only industry information the fair-value model gets, and GICS
sectors are coarse: hardware vs. software (HPQ vs. ADBE) or airlines vs.
defense (DAL vs. LMT) share a sector but not a normal multiple. Stocks from
structurally cheaper sub-industries therefore tend to show up as "저평가".
Finer industry labels would be the first thing to add if that matters.

v3 (2026-09-18) — expanded from 66 (11 sectors x 6) toward ~30/sector, per
user request after the first two real-data runs (66 tickers) came back
0/52 and 0/60 statistically significant factor tests. Same rationale as the
original 33->66 expansion (2026-09-14, see git history / prior chat):
well-known value/quality factors are normally validated on hundreds to
thousands of stocks over decades, not a few dozen — this is the other lever
(besides the 16y Train window already in config.py) for real statistical
power. Also folds in two other 2026-09-18 decisions:

1. Per-sector model splitting is explicitly NOT done yet, on purpose. A
   separate model per sector only makes sense once each sector has enough
   tickers to support its own regression — 6/sector clearly wasn't enough,
   and even the sectors below that only reached the high teens/20s (Energy,
   Materials, Communication Services, Real Estate — see note at the bottom)
   are still thin for a fully separate per-sector model. Sector enters the
   fair-value model as a one-hot level shift instead (fair_value.py), and
   every _pct column is sector-relative (features.add_percentile_scores).

2. A handful of tickers below are deliberately NOT S&P 500-only picks, and
   a handful are deliberately chosen for being awkward cases (meme-stock
   volatility, "value trap" reputations) rather than clean blue chips — see
   the MEME_STOCK_WATCHLIST / VALUE_TRAP_WATCHLIST lists at the bottom. The
   point is to have real examples of the two failure modes the user flagged
   ("밈주식이라 갑자기 고평가", "저평가인데 평생 못 오르는 주식") actually
   sitting in the training data, so any future anomaly-detection/filtering
   logic has real cases to be tested against instead of being designed in
   the abstract. They are NOT excluded from anything — screening.py's
   meme_flag / value_trap_flag are what's meant to surface them.

HOW THIS LIST WAS BUILT — please read this before trusting it blindly:
Sector/ticker membership for the "core" names was cross-checked against
Wikipedia's "List of S&P 500 companies" (fetched 2026-09-18) grouped by
GICS sector. Everything beyond that core list (older delistings, spinoffs,
which tickers have decades of trading history vs. a handful of years) is
from general knowledge, NOT re-verified against a live data feed per
ticker — that verification is exactly what the pipeline's own
data.data_quality_report() does at collection time
("no price history" flags). Treat any ticker flagged there as suspect —
see the 2026-09-18 correction note below for what that turned up the first
time this ran on 283 tickers.

CORRECTION (2026-09-18, after the first 283-ticker real run flagged 10
tickers as "no price history"): checked each one via web search rather than
guessing. 7 were real — all completed M&A/going-private deals that
post-date this assistant's knowledge cutoff (Jan 2026), so they weren't
caught when this list was first built:
  - IPG merged into Omnicom (OMC, already in this universe), delisted
    2025-11-28
  - ANSS acquired by Synopsys (SNPS, already in this universe), completed
    2025-07
  - JNPR acquired by HPE (not in this universe), completed 2025-07
  - MRO acquired by ConocoPhillips (COP, already in this universe), 2024
  - HES acquired by Chevron (CVX, already in this universe), completed
    2025-07
  - WBA taken private by Sycamore Partners, completed 2025-08
  - SEE (Sealed Air) taken private by CD&R, 2025
All 7 are removed below rather than replaced 1:1 with something equally
acquisition-prone — see the per-sector replacement notes. 1 was a ticker
rename, not a delisting: BK -> BNY (Bank of New York Mellon renamed its
ticker; the company itself is unaffected and is kept, just relabeled). The
remaining 2 (FI, K) had no delisting evidence — K (Kellanova) WAS
subsequently confirmed acquired by Mars (completed 2025-12) and is removed;
FI (Fiserv) is still actively trading under that exact ticker per every
source checked, so its "no price history" is presumed a transient
yfinance/network hiccup, not a real problem — left as-is, but if it keeps
showing up in data_quality_report, that's worth a second look (possibly
try the pre-2023 ticker FISV as a fallback).

FOLLOW-UP (2026-09-23): FI flagged "no price history" again on the next
real run, so it was checked directly — yfinance now returns 404 / "possibly
delisted" for FI, while FISV returns current prices (through 2026-09-22).
Fiserv's ticker is FISV again, not a delisting — relabeled FI -> FISV
below, same treatment as BK -> BNY. (Checked on yfinance only; if Finnhub
still expects the old symbol, data_quality_report will show it.)

This will keep happening — completed M&A is exactly the kind of "special
case" the user's meme-stock/value-trap discussion was about, just from the
opposite direction (the company stops existing rather than staying
mispriced). Treat every data_quality_report "no price history" flag as
"go check if this ticker still trades" before assuming it's a bug.

WHY SECTORS AREN'T ALL ~30: Energy, Materials, Communication Services and
Real Estate are GICS sectors with fewer genuine large, long-listed (pre-
2004) public companies to begin with — padding them to exactly 30 would
mean including recent IPOs/spinoffs that contribute almost no Train-period
data (defeats the point) or thin unrelated small-caps just to hit a round
number. Better to have fewer, real, long-history tickers per sector than a
round number full of near-empty rows. Same "handle missing history as NaN,
don't crash" rule from before still applies (see build_raw_panel) — a
partial-history ticker (TSLA, META, GME, etc.) just contributes less to the
earliest Train years rather than breaking anything.
"""
from __future__ import annotations

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
    "FISV": {"name": "Fiserv", "sector": "Technology"},  # FISV -> FI (2023) -> back to FISV; see 2026-09-23 follow-up in docstring
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

    # ---- Energy (15 — see docstring on why this sector is smaller) ----
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

    # ---- Materials (18 — see docstring on why this sector is smaller) ----
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

    # ---- Communication Services (13 — see docstring on why this sector is smaller) ----
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

    # ---- Real Estate (22 — see docstring on why this sector is smaller) ----
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

# Deliberately included as stress-test cases for the still-undesigned
# anomaly-detection/filtering step (2026-09-18 discussion) — NOT excluded
# from anything by default. When that filtering logic exists, these are the
# tickers to check it against first, since we already know which failure
# mode each one represents:
#   - meme-stock: sudden, largely fundamentals-detached overvaluation
#     episodes (GME/AMC 2021, KOSS 2021, BB periodically)
#   - value-trap: persistently "cheap" on valuation percentiles without
#     re-rating (legacy telecom/tobacco names with high yield, declining
#     structural outlook)
# These are reputational/anecdotal labels from general knowledge, not a
# statistical finding from this pipeline — don't treat them as ground truth
# for evaluating a filter, just as known interesting cases to look at.
MEME_STOCK_WATCHLIST: list[str] = ["GME", "AMC", "KOSS", "BB"]
VALUE_TRAP_WATCHLIST: list[str] = ["T", "VZ", "MO", "PM", "F", "M", "LUMN"]


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
