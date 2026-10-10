# Part B: Isolate and Targeted summary -- summary

Model `openai/gpt-5.6-luna`, thinking `low`. Chunking targets ~96,000 tokens per chunk; the actual chunk count per item follows from its real token size.

## Data check: how many times is the target person mentioned?

32/60 items have the target person moving MORE THAN ONCE (regex hits for 'PERSON moved/went/travelled/... (back) to the ROOM'); 28 have exactly one mention, 0 had no literal regex hit. Across the 60 items with at least one hit, the LAST mention's room matches the gold target in 60/60 cases -- when a person moves more than once, the most recent move is usually the right answer. Every demonstration below is therefore told explicitly to take the LAST reported location, never the first.

## Bucket x condition table

| bucket | condition | correct/n | mean | 95% CI | mean total tokens | mean peak single-call context | total cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256k | Part A (single call) | 13/20 | 65.0% | [44.1%, 85.9%] | 241,380 | 241,380 | $1.2088 |
| 256k | Isolate | 12/20 | 60.0% | [38.5%, 81.5%] | 243,066 | 81,264 | $1.2179 |
| 256k | Targeted summary | 6/20 | 30.0% | [9.9%, 50.1%] | 244,370 | 81,278 | $1.2389 |
| 512k | Part A (single call) | 14/20 | 70.0% | [49.9%, 90.1%] | 483,375 | 483,375 | $4.8376 |
| 512k | Isolate | 15/20 | 75.0% | [56.0%, 94.0%] | 486,719 | 81,964 | $2.4377 |
| 512k | Targeted summary | 5/20 | 25.0% | [6.0%, 44.0%] | 489,202 | 81,978 | $2.4762 |
| 768k | Part A (single call) | 9/20 | 45.0% | [23.2%, 66.8%] | 766,805 | 766,805 | $7.6718 |
| 768k | Isolate | 17/20 | 85.0% | [69.4%, 100.0%] | 771,355 | 98,026 | $3.8637 |
| 768k | Targeted summary | 7/20 | 35.0% | [14.1%, 55.9%] | 774,651 | 98,040 | $3.9142 |

## The lesson: peak context, not total tokens

Part A's single call for an item has to hold the ENTIRE haystack in context at once: peak context = total input tokens. Isolate's and Targeted summary's peak single-call context is bounded near one chunk's size instead, while their TOTAL tokens barely move relative to Part A -- the same haystack still gets read in full, just split across more, smaller calls. Isolation and targeted summarization do not save tokens; they cap what any single call has to hold in context at once.
