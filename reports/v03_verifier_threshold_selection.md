# Verifier Threshold Seçimi

## Amaç

İkinci aşama verifier modelinin petrol ve petrol değil kararını vereceği eşik validation verisi üzerinde seçilmiştir.

Calibration ve kilitli test verileri kullanılmamıştır.

## Aday eşikler

| Seçim | Threshold | Precision | Recall | Specificity | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| En iyi balanced accuracy | 0.15 | 1.0000 | 0.9942 | 1.0000 | 0.9971 | 0.9971 |
| En iyi F1 | 0.15 | 1.0000 | 0.9942 | 1.0000 | 0.9971 | 0.9971 |
| Safety — recall ≥ 0.95 | 0.15 | 1.0000 | 0.9942 | 1.0000 | 0.9971 | 0.9971 |

## Önerilen eşik

- Seçim yöntemi: `safety_recall_constraint`
- Verifier threshold: `0.15`
- Precision: 1.0000
- Recall: 0.9942
- Specificity: 1.0000
- F1: 0.9971
- Balanced accuracy: 0.9971
- False-positive rate: 0.0000
- False-negative rate: 0.0058

## Confusion matrix

- True positive: 343.0
- True negative: 858.0
- False positive: 0.0
- False negative: 2.0

## Grup bazlı değerlendirme

| Grup türü | Grup | Örnek | Petrol | Petrol değil | Recall | Specificity | False-positive rate |
|---|---|---:|---:|---:|---:|---:|---:|
| source_group | nc | 99 | 0 | 99 | 0.0000 | 1.0000 | 0.0000 |
| source_group | nw | 759 | 0 | 759 | 0.0000 | 1.0000 | 0.0000 |
| source_group | oc | 98 | 98 | 0 | 1.0000 | 0.0000 | 0.0000 |
| source_group | ow | 247 | 247 | 0 | 0.9919 | 0.0000 | 0.0000 |
| crop_type | easy | 286 | 0 | 286 | 0.0000 | 1.0000 | 0.0000 |
| crop_type | hard | 572 | 0 | 572 | 0.0000 | 1.0000 | 0.0000 |
| crop_type | positive_bbox_context | 345 | 345 | 0 | 0.9942 | 0.0000 | 0.0000 |

## Bilimsel not

Bu eşik yalnız validation verisiyle seçilmiştir. Sonraki aşamada calibration split kullanılarak verifier olasılıkları kalibre edilecektir. Kilitli test verisi henüz kullanılmayacaktır.
