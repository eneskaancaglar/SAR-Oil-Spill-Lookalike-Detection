# v0.5 Input-Gate Veri Seti

## Amaç

Yalnız DARTIS benzeri deniz/kıyı SAR görüntülerinin petrol pipeline'ına gönderilmesini sağlamak.

## Veri kaynakları

- DARTIS: desteklenen deniz/kıyı SAR
- UC Merced: tüketilmiş optik hard-negative
- EuroSAT RGB: optik kara, nehir ve su
- OpenSARUrban: gerçek fakat deniz olmayan SAR

## Split dağılımı

| Split | Desteklenen | Desteklenmeyen | Toplam |
|---|---:|---:|---:|
| train | 2570 | 12009 | 14579 |
| validation | 361 | 2536 | 2897 |
| calibration | 344 | 2540 | 2884 |
| test_negative_locked | 0 | 1582 | 1582 |

## Yeni negatif kilitli test

- EuroSAT `SeaLake` sınıfı
- OpenSARUrban'dan tamamen ayrılmış gruplar:

  - `OpenSARUrban18`
  - `OpenSARUrban4`
  - `OpenSARUrban5`

Bu örnekler model eğitiminde, validation'da ve calibration'da kullanılmayacaktır.

## Önemli sınırlama

v0.5 için yeni bağımsız desteklenen deniz-SAR test kaynağı henüz bulunmamaktadır. Bu nedenle kilitli test yalnız negatif güvenlik performansını ölçecektir.

- Toplam örnek: 21942
- Split leakage: 0
- Eksik görüntü: 0
