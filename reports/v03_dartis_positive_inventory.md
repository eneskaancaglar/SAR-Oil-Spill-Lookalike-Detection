# DARTIS Pozitif Petrol Verisi Envanteri

## Amaç

DARTIS `ow` ve `oc` gruplarındaki petrollü görüntüleri Robust Binary Oil Detector v0.3 eğitimi için hazırlamak.

## Genel sonuç

- Pozitif katalog satırı: 3225
- Benzersiz petrollü görüntü: 1365
- Benzersiz kaynak sahne: 626
- Yerelde bulunan görüntü: 0
- Yerelde bulunan anotasyon: 0
- Anotasyon sütunu: `Binary`

## Grup sonuçları

| Grup | Açıklama | Görüntü | Sahne | Katalog nesne satırı | Yerel görüntü | Yerel anotasyon |
|---|---|---:|---:|---:|---:|---:|
| ow | Petrollü açık deniz | 990 | 452 | 2284 | 0 | 0 |
| oc | Petrollü kıyı | 375 | 281 | 941 | 0 | 0 |

## Anotasyon türleri

- `xml`: 1365

## Sonraki aşama

1. Petrollü `ow` ve `oc` görüntülerini indirmek.
2. Anotasyon dosyalarını indirmek.
3. Anotasyon biçimini incelemek.
4. Anotasyonları 640x640 binary petrol maskesine çevirmek.
5. Görüntü-mask eşleşmelerini doğrulamak.
