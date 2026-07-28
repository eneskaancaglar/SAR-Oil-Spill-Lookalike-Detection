# v0.5 Input-Gate Fine-Tuning

## Amaç

Yalnız DARTIS benzeri deniz ve kıyı SAR görüntülerinin petrol pipeline'ına gönderilmesini sağlayan muhafazakâr giriş modeli geliştirmek.

## Başlangıç modeli

- v0.4 input-gate ResNet18
- Bütün katmanlar düşük öğrenme oranıyla fine-tune edildi

## Yeni negatif veri

- Tüketilmiş UC Merced hard-negative görüntüleri
- EuroSAT optik kara, River ve diğer optik sınıflar
- OpenSARUrban gerçek fakat deniz olmayan VV SAR

## Kullanılan split'ler

- Train: 14579
- Validation: 2897
- Calibration: kullanılmadı
- test_negative_locked: kullanılmadı

## Geçici validation sonucu — threshold 0.50

- Accuracy: 1.0000
- Balanced accuracy: 1.0000
- Desteklenen SAR recall: 1.0000
- Desteklenmeyen reddetme: 1.0000
- False acceptance rate: 0.0000
- False rejection rate: 0.0000
- F1: 1.0000

## Bilimsel durum

Bu aşamadaki 0.50 eşiği nihai eşik değildir.

Calibration ve test_negative_locked bölümleri eğitim sırasında kullanılmamıştır.

Nihai KABUL, BELİRSİZ ve RED eşikleri sonraki kalibrasyon aşamasında dondurulacaktır.
