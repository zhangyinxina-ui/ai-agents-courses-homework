# Part A: BABILong qa1, accuracy vs context length

Single-message delivery, compaction disabled, exact-match scoring.

| bucket | n | mean | 95% CI | mean real input tokens |
| --- | --- | --- | --- | --- |
| 32k | 20 | 85.0% | [69.4%, 100.0%] | 30,549 |
| 128k | 20 | 75.0% | [56.0%, 94.0%] | 122,552 |
| 256k | 20 | 65.0% | [44.1%, 85.9%] | 241,380 |
| 512k | 20 | 70.0% | [49.9%, 90.1%] | 483,375 |
| 768k | 20 | 45.0% | [23.2%, 66.8%] | 766,805 |

The 768K bucket has no native BABILong equivalent: it is built by truncating 1M-bucket items to 768,000 tokens and keeping only those whose qa1 supporting fact survives truncation. Surviving needles never sit past about 80 percent of the original document, so that bucket is structurally somewhat easier than an unbiased sample of the same length; any decline it shows is, if anything, an understatement.
