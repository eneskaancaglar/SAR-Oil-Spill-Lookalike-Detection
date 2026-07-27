# DARTIS Negatif Verifier Crop Hazırlığı

## Amaç

Petrolsüz DARTIS `nc` ve `nw` görüntülerinde mevcut petrol segmentasyon modelinin en fazla petrol olasılığı verdiği bölgeleri zor negatif örnek olarak toplamak.

Bütün örneklerin verifier etiketi:

- `0`: Petrol değil

## Bilimsel koruma

`external_test` görüntüleri eğitim verisi üretiminde kullanılmamıştır.

## Sonuç

- İşlenen görüntü: 1797
- Başarılı görüntü: 1797
- Hatalı görüntü: 0
- Toplam negatif crop: 5391
- Zor negatif crop: 3594
- Kolay negatif crop: 1797
- Model threshold: 0.6

## Split sonuçları

| Kaynak split | Çıktı split | Görüntü | Crop |
|---|---|---:|---:|
| hard_negative_train | train | 1304 | 3912 |
| hard_negative_val | validation | 493 | 1479 |

## Grup sonuçları

| Grup | Görüntü | Hard | Easy | Toplam |
|---|---:|---:|---:|---:|
| nc | 280 | 560 | 280 | 840 |
| nw | 1517 | 3034 | 1517 | 4551 |

## Sonraki aşama

Pozitif ve negatif crop manifestleri sahne bazında birleştirilecek ve verifier modeli için dengeli train, validation ve calibration ayrımı oluşturulacaktır.
