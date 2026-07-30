# v0.6 Water / Oil / Look-Alike Etiket Denetimi

## Amaç

Mevcut verilerde aşağıdaki bilgilerin bulunup bulunmadığını doğrulamak:

- Piksel seviyesinde petrol maskesi
- Piksel seviyesinde kara–su maskesi
- Görüntü veya aday seviyesinde look-alike etiketi

## Genel sonuç

- Metadata CSV: 14
- Semantik klasör: 3
- Maske klasörü: 2
- Binary maske adayı klasör: 2
- Eşleşen görüntü–maske çifti: 6455
- DARTIS grup sayısı: 4
- Açık kara–su etiketi kanıtı: True
- Açık look-alike etiketi kanıtı: False

## Bilimsel uyarı

Klasör adlarının `ow`, `oc`, `nw`, `nc` olması tek başına etiket anlamlarını doğrulamaz. Dokümantasyon veya manifest kanıtı olmadan bu gruplar model sınıfı olarak kabul edilmeyecektir.

Bu denetim sırasında hiçbir veri değiştirilmemiş, model eğitilmemiş ve kilitli test kullanılmamıştır.