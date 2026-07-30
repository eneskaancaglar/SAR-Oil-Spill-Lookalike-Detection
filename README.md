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
