# Robust Binary Oil Detector v0.3
# Final Model Raporu

## Sistem ne yapıyor?

Sistem bir gri seviye SAR görüntüsünü alır, petrol olabilecek bölgeleri U-Net ile maskeler ve bu bölgeleri ResNet18 verifier ile `petrol / petrol değil` şeklinde doğrular.

Nihai çıktı:

- Petrol var / yok kararı
- Nihai petrol maskesi
- Görüntü üzerindeki petrol kaplama yüzdesi
- Aday bölge güven skoru
- Görsel overlay

## Kilitli test sonuçları

- Kilitli petrollü görüntü: 51
- Pozitif görüntü recall: 0.6275
- XML kutu recall: 0.4118
- Petrolsüz external-test görüntüsü: 493
- Herhangi yanlış alarm: %21.70
- Ciddi yanlış alarm (≥%1): %9.53
- Yanlış pozitif piksel: %0.5091
- Kıyı `nc` ciddi alarm: %46.48
- Açık deniz `nw` ciddi alarm: %3.32

## Verifier kalibrasyonu

- Temperature: 1.111086
- Ham threshold: 0.150000
- Kalibre threshold: 0.173478
- NLL: 0.014736 → 0.014601
- Brier: 0.004475 → 0.004414
- ECE: 0.006795 → 0.007971

ECE küçük ölçüde yükseldiği için verifier çıktısı kesin başarı olasılığı değil, kalibre edilmiş aday-petrol güven skoru olarak sunulmalıdır.

## Final paket dosyaları

| Dosya | Boyut | SHA-256 |
|---|---:|---|
| `checkpoints\final_model_v03\oil_segmentation_unet.pth` | 23415571 bayt | `d4d26cf4b8eec484c71e8f86c39ee9a55b8c72c65f1f831d919e12e55b1f553b` |
| `checkpoints\final_model_v03\oil_candidate_verifier_resnet18.pth` | 134181783 bayt | `3eda133f2247e4bc8a0260c3d81d6d5a1b8d64d74d470211059064daf31ceb11` |
| `checkpoints\final_model_v03\calibration_config.json` | 573 bayt | `323ca85204d753334f40c5de6d5a3cf52b2d2ae6ad73ccaad51fb3f395c47160` |
| `checkpoints\final_model_v03\detector_config.json` | 1484 bayt | `d59ff6c1cf3690a6660ce70848fed6bce8f9bd4fcf601eba6896d1adac0b934d` |
| `checkpoints\final_model_v03\model_card.json` | 6319 bayt | `40dbfc263acf840c7ddbe950a57e928fefb00dc6a459d948d1214559c90ad016` |

## Kullanım sınırları

- Model araştırma prototipidir.
- Kara-deniz maskesi bulunmamaktadır.
- Kaplama yüzdesi bütün görüntüye göredir.
- Güven skoru aday bölge seviyesindedir.
- Kıyı görüntüleri açık denizden daha zordur.
- Kilitli test sonuçlarına bakılarak mevcut model yeniden ayarlanmamalıdır.

## Bilimsel durum

Bu model ve eşikler dondurulmuştur. Yeni bir iyileştirme yapılacaksa yeni geliştirme verisi, yeni validation/calibration bölümü ve yeni bir kilitli test seti oluşturulmalıdır.
