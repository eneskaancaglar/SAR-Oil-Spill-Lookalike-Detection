# v0.4 Input-Gate Kalibrasyonu

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

- Temperature: 0.715186
- Accept probability: 0.398661
- Accept similarity: 0.606643
- Reject probability: 0.012930
- Reject similarity: -1.000000

## Calibration sonucu

- Desteklenen SAR kabul oranı: 0.9971
- Desteklenen SAR belirsiz oranı: 0.0029
- Desteklenen SAR yanlış red oranı: 0.0000
- Desteklenmeyen giriş red oranı: 1.0000
- Desteklenmeyen giriş belirsiz oranı: 0.0000
- Desteklenmeyen giriş yanlış kabul oranı: 0.0000

## Bilimsel durum

Kilitli test bu aşamada kullanılmamıştır.

Airplane, harbor, river ve runway kategorileri sonraki aşamada ilk kez kullanılacaktır.

Validation başarısının yüzde yüz olması sistemin evrensel biçimde güvenilir olduğunu göstermez. Asıl karar kilitli testten sonra verilecektir.
