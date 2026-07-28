# v0.5 Input-Gate Kalibrasyonu

## Amaç

Yüklenen görüntüyü petrol analizine göndermeden önce muhafazakâr biçimde KABUL, BELİRSİZ veya RED olarak sınıflandırmak.

## Karar sistemi

### KABUL

Hem deniz-SAR olasılığı hem de DARTIS prototip benzerliği kabul eşiklerini geçmelidir.

### RED

Deniz-SAR olasılığı veya prototip benzerliği red eşiğinin altında kalırsa görüntü reddedilir.

### BELİRSİZ

Kabul edilecek kadar güçlü olmayan ancak doğrudan reddedilecek kadar da uzak olmayan görüntülerdir. Petrol analizi çalıştırılmaz.

## Threshold değerleri

- Temperature: 0.156739
- Accept probability: 0.999000
- Accept similarity: 0.500000
- Reject probability: 0.001000
- Reject similarity: -1.000000

## Calibration sonucu

- Desteklenen SAR kabul oranı: 1.0000
- Desteklenen SAR belirsiz oranı: 0.0000
- Desteklenen SAR yanlış red oranı: 0.0000
- Desteklenmeyen giriş red oranı: 1.0000
- Desteklenmeyen giriş belirsiz oranı: 0.0000
- Desteklenmeyen giriş yanlış kabul oranı: 0.0000

## Bilimsel durum

Kilitli test bu aşamada kullanılmamıştır.

Airplane, harbor, river ve runway kategorileri sonraki aşamada ilk kez kullanılacaktır.

Validation başarısının yüzde yüz olması sistemin evrensel biçimde güvenilir olduğunu göstermez. Asıl karar kilitli testten sonra verilecektir.
