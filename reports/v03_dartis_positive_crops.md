# DARTIS Pozitif Verifier Crop Hazırlığı

## Amaç

DARTIS petrollü `ow` ve `oc` görüntülerindeki XML bounding box anotasyonlarını, ikinci aşama petrol doğrulama modeli için pozitif crop örneklerine çevirmek.

## Önemli bilimsel karar

XML kutuları piksel seviyesinde petrol maskesi değildir. Bu nedenle kutunun tamamı petrol maskesi yapılmamıştır. Kutular yalnızca pozitif aday bölgeleri crop etmek için kullanılmıştır.

## Sonuç

- Manifest görüntüsü: 1365
- İşlenen görüntü: 1365
- Başarılı görüntü: 1365
- Hatalı görüntü: 0
- XML nesnesi: 3225
- Geçerli bounding box: 3175
- Geçersiz bounding box: 50
- Kaydedilen pozitif crop: 3175
- XML boyut uyuşmazlığı: 0

## Grup sonuçları

| Grup | Görüntü | Nesne | Crop |
|---|---:|---:|---:|
| ow | 990 | 2238 | 2238 |
| oc | 375 | 937 | 937 |

## Sonraki aşama

DARTIS `nc` ve `nw` petrolsüz görüntülerinden ve mevcut modelin yanlış alarm bölgelerinden negatif verifier crop örnekleri üretilecektir.
