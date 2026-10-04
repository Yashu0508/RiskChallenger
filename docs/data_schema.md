# Supplied price dataset schema

Inspected Phase 1 source: `data/raw/dataset_prices.csv`.

The file is a UTF-8 CSV in wide format. Its first column is `date`, followed by
50 equity ticker columns (for example, `HDFCBANK.NS` and `M&M.NS`). Each row
represents one observed date, and each ticker field contains that asset's price
as a decimal numeric string. Dates use ISO `YYYY-MM-DD` format.

The supplied source contains 4,185 rows covering 2008-01-02 through 2024-12-30.
The inspection found no missing cells, duplicate rows, duplicate dates, invalid
dates, non-numeric or non-finite prices, or zero/negative prices. The rows are
chronologically ordered and every asset has 4,185 observations. Calendar gaps
are reported descriptively; because no exchange trading calendar was supplied,
they are not treated as proof of missing trading sessions.

The canonical prices retain each valid source price token and sort dates
chronologically. Simple returns are calculated as `P_t / P_(t-1) - 1`; the first
return per asset is blank because no prior observed price exists. Returns are
decimal values, so `0.01` represents one percent. The source file is read-only
from the pipeline's perspective.
