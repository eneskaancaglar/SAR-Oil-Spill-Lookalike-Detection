# v0.6 DARTIS Kara ve Güvenli Su Maskeleri

## İşlem

- `oc` ve `nc`: WGS84 köşe koordinatları ve OpenStreetMap kara
  poligonları kullanılarak kara maskesi üretildi.
- `ow` ve `nw`: DARTIS açık-su grup anlamına göre bütün görüntü
  su kabul edildi.
- Kara maskesi `8` piksel genişletilerek
  kıyı tamponu oluşturuldu.
- Petrol/look-alike analizi yalnız `safe_water_mask=1` alanında
  yapılacaktır.

## Sonuç

- Toplam kayıt: 3655
- Başarılı: 3655
- Hatalı: 0
- Kıyılı kayıt: 726
- Açık-su kayıt: 2929
- Kara pikseli bulunmayan kıyılı kayıt: 5
- Tamamen engellenen kayıt: 0

## Maske anlamı

- Kara maskesi: `255=kara`, `0=su`
- Güvenli su maskesi: `255=analiz edilebilir su`,
  `0=kara veya kıyı tamponu`

## Bilimsel durum

Bu maskeler görüntüden öğrenilmiş gerçek insan anotasyonları değil,
DARTIS köşe koordinatları ile OpenStreetMap kara poligonlarından üretilmiş
coğrafi denetim maskeleridir.

Model eğitiminden önce inceleme görüntülerinin elle kontrol edilmesi
zorunludur.

OpenStreetMap verisi ODbL kapsamında kullanılmıştır.

Kilitli test kullanılmamış ve model eğitilmemiştir.
