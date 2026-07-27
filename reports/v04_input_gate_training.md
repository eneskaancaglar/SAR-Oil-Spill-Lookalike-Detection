# v0.4 Input-Gate Model Eğitimi

## Amaç

Petrol analizinden önce yüklenen görüntünün DARTIS benzeri deniz SAR veri alanında olup olmadığını belirlemek.

## Kullanılan split'ler

- Train: 3770
- Validation: 661
- Calibration: kullanılmadı
- Test locked: kullanılmadı

## Validation sonucu — geçici threshold 0.50

- Accuracy: 1.0000
- Balanced accuracy: 1.0000
- Desteklenen deniz SAR recall: 1.0000
- Desteklenmeyen giriş reddetme oranı: 1.0000
- False acceptance rate: 0.0000
- False rejection rate: 0.0000
- F1: 1.0000

## Bilimsel durum

Bu aşamada kullanılan 0.50 eşiği nihai kabul eşiği değildir.

Sonraki aşamada calibration split üzerinde muhafazakâr kabul, red ve belirsiz bölgeleri belirlenecektir.

Kilitli testteki airplane, harbor, river ve runway kategorileri henüz kullanılmamıştır.

Bu model tek başına denizin fiziksel varlığını garanti etmez. Sonraki aşamada sınıflandırıcı skoruna ek olarak desteklenen SAR dağılımına uzaklık kontrolü eklenecektir.
