# v0.5 Input-Gate Yeni Negatif Kilitli Test

## Protokol

- Model ağırlıkları donduruldu.
- Temperature değeri donduruldu.
- KABUL, BELİRSİZ ve RED eşikleri donduruldu.
- Kilitli test sonuçlarına göre eşik değiştirilmedi.
- Kilitli görüntüler eğitim ve calibration sırasında kullanılmadı.

## Genel sonuç

- Toplam negatif görüntü: 1582
- Yanlış kabul: 0
- Belirsiz: 2
- Red: 1580
- Yanlış kabul oranı: 0.000000
- Güvenli engelleme oranı: 1.000000
- Güvenlik testi: BAŞARILI

## Kaynak ve grup sonuçları

| Veri | Grup | Örnek | Kabul | Belirsiz | Red | Yanlış kabul oranı |
|---|---|---:|---:|---:|---:|---:|
| eurosat_rgb | SeaLake | 1200 | 0 | 2 | 1198 | 0.000000 |
| opensarurban | OpenSARUrban18 | 72 | 0 | 0 | 72 | 0.000000 |
| opensarurban | OpenSARUrban5 | 189 | 0 | 0 | 189 | 0.000000 |
| opensarurban | OpenSARUrban4 | 121 | 0 | 0 | 121 | 0.000000 |

## Sınırlama

Bu sürüm için yeni ve bağımsız pozitif deniz-SAR test kaynağı bulunmamaktadır. Bu nedenle bu test yalnız desteklenmeyen girişlerin güvenli biçimde engellenmesini ölçmektedir.

Calibration skorlarının 0 veya 1'e çok yakın olması skorların gerçek dünya olasılığı olduğu anlamına gelmez.
