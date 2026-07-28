# v0.5 EuroSAT RGB Envanteri

## Amaç

Input-gate modeline optik uydu, optik su, nehir, göl ve kara görüntülerini reddetmeyi öğretmek.

Bu veri setindeki bütün görüntüler:

- `binary_label = 0`
- `unsupported_input`
- Petrol analizinden önce reddedilecek

## Kritik hard-negative sınıflar

- River
- SeaLake

Bu iki sınıf mevcut modelin nehir ve su yüzeyi yanlış kabullerini azaltmak için daha yüksek eğitim ağırlığı alacaktır.

## Sınıf dağılımı

| Sınıf | Görüntü | Ağırlık | Tür |
|---|---:|---:|---|
| AnnualCrop | 3000 | 1.0 | optical_land_negative |
| Forest | 3000 | 1.0 | optical_land_negative |
| HerbaceousVegetation | 3000 | 1.0 | optical_land_negative |
| Highway | 2500 | 2.0 | optical_structural_negative |
| Industrial | 2500 | 2.0 | optical_structural_negative |
| Pasture | 2000 | 1.0 | optical_land_negative |
| PermanentCrop | 2500 | 1.0 | optical_land_negative |
| Residential | 3000 | 2.0 | optical_structural_negative |
| River | 2500 | 5.0 | optical_water_hard_negative |
| SeaLake | 3000 | 5.0 | optical_water_hard_negative |

## Doğrulama

- Toplam görüntü: 27000
- Beklenen görüntü: 27000
- Sınıf sayısı: 10
- ZIP MD5: `f46e308c4d50d4bf32fedad2d3d62f3b`
- Checksum doğrulandı: True

## Bilimsel kullanım

EuroSAT görüntüleri v0.5 geliştirme verisidir. Yeni v0.5 kilitli test sonucu olarak kullanılmayacaktır.

Optik görüntülerin yanında gerçek deniz olmayan SAR görüntüleri de ayrıca negatif sınıfa eklenecektir.
