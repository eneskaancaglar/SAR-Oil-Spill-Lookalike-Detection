# SAR Oil-Spill Detection — Final Research Prototype Report

## Final status

**Frozen version:** `v0.8-research-prototype`

**Project status:** Completed as a research prototype.

**Operational deployment:** Not approved.

**Required interpretation:** Any positive result is an **oil candidate requiring expert review**. It must not be presented as an automatic operational oil-spill decision.

## Frozen pipeline

1. **v0.5 input-domain gate**  
   Rejects unsupported non-SAR or out-of-domain inputs.

2. **v0.6 selective water gate**  
   Allows oil analysis only on high-confidence water pixels. Land and uncertain pixels are excluded from the final oil mask.

3. **Oil candidate segmenter**  
   Produces candidate dark regions only inside the accepted water mask.

4. **v0.8 look-alike-aware verifier**  
   Produces `LOOK_ALIKE`, `UNCERTAIN`, or `CONFIRMED_OIL`. Only `CONFIRMED_OIL` may enter the research mask.

## Final evidence

### Input-domain safety

- Locked negative samples: **1,582**
- False accepts: **0**
- Safe blocking rate: **100%**

### Water gate

- Locked test accepted scenes: **5/8**
- Water precision: **100.000%**
- Mean accepted recall: **90.54%**
- Land leakage: **0.0300%**

### v0.8 verifier calibration

- Confirmed-oil precision: **100.00%**
- Confirmed-oil recall: **80.88%**
- Negative safe-block rate: **100.00%**
- Calibration operational gate: **PASS**

This is calibration evidence, not independent final proof.

### Independent negative holdout

- Fresh negative scenes: **40**
- Water-gate accepted scenes: **32/40**
- Confirmed-oil false-alarm scenes: **2/40**
- Overall false-alarm rate: **5.00%**
- False-alarm rate among water-accepted scenes: **6.25%**
- Incorrect final oil pixels: **2,646**
- Independent negative gate: **FAIL**

### Improvement from v0.7 to v0.8

- False-alarm scenes: **6/40 → 2/40**
- Incorrect oil pixels: **27,750 → 2,646**

The model improved substantially, but the zero-false-alarm safety target was not reached.

## Scientific conclusion

The project successfully demonstrates a complete safeguarded SAR oil-spill research pipeline with:

- input-domain rejection,
- selective land/water isolation,
- water-only oil candidate production,
- look-alike and uncertainty handling,
- independent negative testing,
- reproducible model and evaluation artifacts.

The independent negative holdout still produced **2 false-alarm scenes out of 40**. In addition, a fresh independent positive holdout was not available. Therefore:

- no operational deployment claim is made,
- no independent positive-recall claim is made,
- automatic final oil-spill decisions are not allowed,
- outputs must remain research candidates for expert review.

## Stopped v0.9 branch

The attempted dual-view extension was intentionally stopped. Reliable positive source-mask pairs were found for only **756/1,365** scenes:

- `oc`: **375/375**
- `ow`: **381/990**

This coverage was insufficient for a scientifically balanced dual-view retraining data set. Continuing would have increased complexity without trustworthy evidence.

## Final deliverable decision

The technically and scientifically correct endpoint is the frozen `v0.8-research-prototype`, together with its passed component gates, failed independent negative gate, limitations, reproducible summaries, and expert-review-only usage policy.
