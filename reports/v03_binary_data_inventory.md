# Robust Binary Oil Detector v0.3 — Veri Envanteri

## Hedef etiket şeması

- `1`: Petrol
- `0`: Petrol değil

Kara, kıyı, rüzgâr izi, alg, biyolojik film, sığ su, gemi izi ve diğer tüm yapılar `petrol değil` sınıfındadır.

## SOS

- Toplam örnek: 8070
- Petrol örneği: 7745
- Petrolsüz örnek: 325
- Yerel görüntü: 8070
- Yerel maske: 8070

### SOS split dağılımı

- train: 6455
- val: 1615

### SOS sensör dağılımı

- Sentinel-1: 4193
- PALSAR: 3877

## DARTIS

- Katalog satırı: 5515
- Benzersiz görüntü: 3655
- Petrol görüntüsü: 1365
- Petrolsüz görüntü: 2290
- Yerelde bulunan görüntü: 2290

### DARTIS grupları

| Grup | Açıklama | Etiket | Görüntü | Sahne | Yerelde |
|---|---|---:|---:|---:|---:|
| ow | Petrollü açık deniz | 1 | 990 | 452 | 0 |
| oc | Petrollü kıyı | 1 | 375 | 281 | 0 |
| nw | Petrolsüz açık deniz | 0 | 1939 | 729 | 1939 |
| nc | Petrolsüz kıyı/kara | 0 | 351 | 289 | 351 |

## Anotasyonla ilişkili katalog sütunları

- `Binary`
- `annotation_reference_value`

## Sonraki işlem

1. DARTIS `ow` ve `oc` petrollü görüntülerinin yerel durumunu doğrulamak.
2. Pozitif anotasyonların XML, binary görüntü veya başka formatta olup olmadığını kesinleştirmek.
3. Pozitif DARTIS görüntü ve anotasyonlarını indirmek.
4. Bütün pozitif anotasyonları piksel maskesine çevirmek.
5. Yeni sahne bazlı train/validation/calibration/test ayrımı oluşturmak.
