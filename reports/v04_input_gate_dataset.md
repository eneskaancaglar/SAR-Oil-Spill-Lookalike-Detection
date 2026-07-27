# v0.4 Input Gate Veri Seti

## Amaç

Petrol analizinden önce yüklenen görüntünün desteklenen veri alanında olup olmadığını belirlemek.

Desteklenen giriş:

- DARTIS benzeri deniz veya kıyı içeren SAR görüntüsü

Desteklenmeyen giriş:

- Optik uydu görüntüsü
- Hava fotoğrafı
- Uçak, bina, tarım, yol ve benzeri optik sahneler

## Etiketler

- `1`: supported_sea_sar
- `0`: unsupported_input

## Split protokolü

- DARTIS görüntüleri sahne kimliğine göre bölündü.
- Aynı DARTIS sahnesi birden fazla split'e girmedi.
- UC Merced kategorileri sınıf bazında bölündü.
- `airplane`, `harbor`, `river` ve `runway` yalnız kilitli testte tutuldu.

## Dağılım

| Split | Desteklenen | Desteklenmeyen | Toplam |
|---|---:|---:|---:|
| train | 2570 | 1200 | 3770 |
| validation | 361 | 300 | 661 |
| calibration | 344 | 200 | 544 |
| test_locked | 380 | 400 | 780 |

- Toplam örnek: 5755
- Desteklenen deniz SAR: 3655
- Desteklenmeyen optik: 2100
- Split leakage: 0

## Önemli sınır

Bu veri seti evrensel bir `SAR / optik` sınıflandırıcısı oluşturmaz. Amaç, uygulamanın eğitim alanına uymayan girdileri muhafazakâr biçimde reddetmesidir.

Model kararsız kaldığında görüntü petrol analizine gönderilmeyecektir.
