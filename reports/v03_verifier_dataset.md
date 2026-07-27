# Robust Binary Oil Detector v0.3 — Verifier Veri Seti

## Amaç

DARTIS petrol bounding box crop'ları ile petrolsüz zor ve kolay negatif crop'ları birleştirilmiştir.

Verifier modelinin çıktısı:

- `1`: Petrol
- `0`: Petrol değil

## Bilimsel veri ayrımı

- Ayrım kaynak sahne bazında yapılmıştır.
- Aynı sahne birden fazla split içinde bulunmaz.
- DARTIS `external_test` eğitimde kullanılmamıştır.
- `calibration`, güven yüzdesini kalibre etmek için ayrılmıştır.
- `test_locked`, model seçimi bitene kadar kullanılmayacaktır.

## Genel sonuç

- Toplam örnek: 8566
- Petrol örneği: 3175
- Petrol değil örneği: 5391
- Benzersiz sahne: 1101
- Eksik crop dosyası: 0
- Scene leakage: 0

## Split dağılımı

| Split | Örnek | Sahne | Petrol | Petrol değil | Hard negatif | Easy negatif |
|---|---:|---:|---:|---:|---:|---:|
| train | 6475 | 872 | 2563 | 3912 | 2608 | 1304 |
| validation | 1203 | 97 | 345 | 858 | 572 | 286 |
| calibration | 769 | 97 | 148 | 621 | 414 | 207 |
| test_locked | 119 | 35 | 119 | 0 | 0 | 0 |

## Train sınıf ağırlıkları

- Petrol değil: 0.827582
- Petrol: 1.263168
- Hard-negative çarpanı: 1.50

## Test notu

`test_locked` bölümünde yalnızca kilitli pozitif petrol crop'ları bulunabilir. Nihai negatif test, daha önce saklanan DARTIS `external_test` görüntülerinden model seçimi tamamlandıktan sonra oluşturulacaktır.

## Sonraki aşama

Verifier sınıflandırma modeli eğitilecek ve validation sonuçlarına göre en iyi checkpoint seçilecektir.
