# Robust Binary Oil Detector v0.3
# Kilitli Uçtan Uca Değerlendirme

## Bilimsel protokol

- Verifier threshold validation verisiyle seçildi.
- Temperature calibration yalnız calibration split ile yapıldı.
- Bu aşamada test_locked pozitif sahneler ilk kez kullanıldı.
- DARTIS external_test negatif sahneleri eğitimde kullanılmadı.
- Bu testten sonra model veya eşikler bu sonuçlara göre değiştirilmemelidir.

## Kilitli pozitif sonuçları

- Pozitif görüntü: 51
- Petrol tespit edilen görüntü: 32
- Kaçırılan görüntü: 19
- Görüntü seviyesi recall: 0.6275
- XML petrol kutusu: 119
- Maskeyle kesişen kutu: 49
- Kutu seviyesi recall: 0.4118

XML anotasyonları bounding box olduğundan bu bölümde Dice veya IoU hesaplanmamıştır.

## Petrolsüz external-test sonuçları

| Grup | Görüntü | Herhangi alarm | ≥%0.01 | ≥%0.1 | ≥%1 | FP piksel |
|---|---:|---:|---:|---:|---:|---:|
| overall | 493 | %21.70 | %21.70 | %20.28 | %9.53 | %0.5091 |
| nc | 71 | %73.24 | %73.24 | %70.42 | %46.48 | %2.8046 |
| nw | 422 | %13.03 | %13.03 | %11.85 | %3.32 | %0.1229 |

## Yorumlama

Pozitif recall, sistemin gerçek petrol içeren kilitli görüntüleri yakalama oranını gösterir.

Negatif sonuçlardaki `≥%1`, petrol bulunmayan bir görüntünün en az yüzde birinin yanlışlıkla petrol olarak maskelendiği ciddi yanlış alarm oranıdır.

Bu sürümde bağımsız kara-deniz maskesi bulunmadığından kıyı grubu `nc` hâlâ en zor gruptur.
