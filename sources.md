# Sources and indicator definitions

All historical observations are requested from the World Bank Indicators API. The app builds a source link for the selected country and indicator and records the retrieval date in the report.

| Indicator | World Bank code | Units shown by the app |
|---|---|---|
| Life expectancy at birth | `SP.DYN.LE00.IN` | years |
| Unemployment, national estimate | `SL.UEM.TOTL.NE.ZS` | % of labor force |
| GDP per capita | `NY.GDP.PCAP.CD` | current US dollars per person |
| Crude birth rate | `SP.DYN.CBRT.IN` | births per 1,000 people per year |
| Net migration | `SM.POP.NETM` | people |
| Intentional homicide rate | `VC.IHR.PSRC.P5` | per 100,000 people |
| Consumer price inflation | `FP.CPI.TOTL.ZG` | % annual change |
| Central government debt | `GC.DOD.TOTL.GD.ZS` | % of GDP |
| Total reserves including gold | `FI.RES.TOTL.CD` | current US dollars |
| External balance on goods and services | `NE.RSB.GNFS.ZS` | % of GDP |

API endpoint pattern: `https://api.worldbank.org/v2/country/{ISO3}/indicator/{INDICATOR_CODE}?date={START}:{END}&format=json&per_page=1000`

Reference catalog: https://data.worldbank.org/indicator/

The country choices are Mexico, Brazil, Argentina, Colombia, Chile, Peru, Venezuela, Ecuador, and Uruguay. Brazil is presented as a Latin American comparison country, not a Spanish-speaking country.
