# v0.6 Manuel Kara Maskesi Düzeltmeleri

Bu paket, sohbette gönderilen yedi problemli DARTIS `nc` görüntüsü için
hazırlanmış hizalı kara ve güvenli-su maskelerini içerir.

## Panel anlamı

`outputs/v06_manual_land_corrections/review` klasöründeki görüntüler:

1. Orijinal SAR
2. Kırmızı: kara, sarı: kıyı sınırı
3. Yalnız analiz edilecek güvenli su

## Kurulum

ZIP içeriğini proje köküne açın:

```powershell
Expand-Archive `
  -LiteralPath "$env:USERPROFILE\Downloads\v06_manual_land_corrections.zip" `
  -DestinationPath "." `
  -Force
```

Ardından:

```powershell
python -m py_compile `
  .\scripts\59_apply_manual_land_corrections.py

python `
  .\scripts\59_apply_manual_land_corrections.py
```

Final manifest:

```text
data/metadata/v06_dartis_water_masks_final.csv
```

Bu manifestte:

- `ow` ve `nw` açık-su örnekleri eğitim için uygundur.
- Bu paketteki yedi kıyı örneği manuel düzeltme olarak uygundur.
- Elle doğrulanmamış diğer kıyı örnekleri otomatik olarak eğitim dışı kalır.

Bu yedi maske petrol/look-alike etiketi değildir; yalnız kara-su ayrımıdır.

## Final Detection Limitation

The final system is a safeguarded research and decision-support prototype, not an operational oil-spill detector.

Manual exploratory checks on known positive DARTIS scenes revealed false negatives. Some `oc` and `ow` scenes produced no high-confidence candidate, while some negative scenes produced false alarms in the independent holdout. Therefore:

- a positive output requires expert review,
- an empty candidate mask does not prove that oil is absent,
- the system must not be used for autonomous operational decisions,
- the frozen v0.8 pipeline is the final internship handoff version.

## Final Sensitive Screening Application

Run the final application locally with:

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run ".\app_final.py"
```

The application provides two separate outputs:

1. Sensitive screening mask: highlights suspicious regions to reduce missed detections. It may contain false positives.
2. Confirmed candidate mask: contains only candidates accepted by the verifier.

A sensitive screening result is not proof of an oil spill. An empty or weak mask does not prove that oil is absent. Expert interpretation is mandatory.
