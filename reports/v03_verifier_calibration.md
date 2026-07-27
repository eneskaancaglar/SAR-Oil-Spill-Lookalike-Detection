# Verifier Olasılık Kalibrasyonu

## Amaç

Verifier modelinin ürettiği skorları daha anlamlı petrol güven yüzdelerine dönüştürmek.

Kalibrasyonda yalnız `calibration` split kullanılmıştır. `test_locked` kullanılmamıştır.

## Temperature scaling

- Temperature: `1.111086`
- Ham verifier threshold: `0.150000`
- Kalibre edilmiş eşdeğer threshold: `0.173478`

## Kalibrasyon metrikleri

| Metrik | Önce | Sonra |
|---|---:|---:|
| NLL | 0.014736 | 0.014601 |
| Brier score | 0.004475 | 0.004414 |
| ECE | 0.006795 | 0.007971 |

Düşük NLL, Brier score ve ECE daha iyi kalibrasyon anlamına gelir.

## Korunan karar performansı

- Precision: 1.0000
- Recall: 0.9865
- Specificity: 1.0000
- F1: 0.9932
- False positive: 0
- False negative: 2

Temperature scaling monotonik olduğu için ham threshold kalibre edilmiş eşdeğer threshold'a dönüştürülmüş ve petrol/petrol değil kararları korunmuştur.

## Kullanım

Yeni bir aday crop için:

```text
raw_logit = verifier(crop)
calibrated_probability = sigmoid(raw_logit / temperature)
petrol = calibrated_probability >= calibrated_threshold
```

Bu güven değeri aday crop seviyesindedir. Bütün görüntü için nihai güven değeri, end-to-end pipeline aşamasında adayların sonuçları birleştirilerek hesaplanacaktır.
