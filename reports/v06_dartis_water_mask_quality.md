# v0.6 DARTIS Kara–Su Maskesi Kalite Denetimi

## Amaç

Homography ile üretilmiş kıyı maskelerinin SAR görüntüsündeki gerçek
kıyı sınırına ne kadar uyduğunu ölçmek.

## Sonuç

- Kıyılı örnek: 726
- AUTO_ACCEPT: 103
- REVIEW_REQUIRED: 614
- REJECT: 9
- Otomatik kabul oranı: %14.19

## Kullanılan işaretler

- Mevcut sınırın görüntü gradientiyle uyumu
- Yakındaki daha iyi global kayma
- Farklı kıyı bölümlerinin kayma uyuşmazlığı
- Aşırı kara veya güvenli-su oranı

## Bilimsel sınırlama

Bu puanlar insan anotasyonu değildir ve maskenin doğru olduğunu
kanıtlamaz. AUTO_ACCEPT klasörü de örnekleme yöntemiyle görsel olarak
denetlenmelidir.

REVIEW_REQUIRED ve REJECT örnekleri water-mask eğitimine otomatik olarak
girmeyecektir.

Bu aşamada maskeler değiştirilmemiş, model eğitilmemiş ve kilitli test
kullanılmamıştır.
