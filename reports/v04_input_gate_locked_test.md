# v0.4 Input-Gate Kilitli Test

## Amaç

Petrol analizinden önce yalnız desteklenen DARTIS benzeri deniz SAR görüntülerinin kabul edilip edilmediğini bağımsız kilitli veri üzerinde ölçmek.

## Protokol

- Model ağırlıkları donduruldu.
- Temperature ve karar eşikleri donduruldu.
- Kilitli test sonuçlarına göre eşik ayarlanmadı.
- Airplane, harbor, river ve runway sınıfları eğitimde kullanılmadı.

## Genel sonuçlar

- Kilitli desteklenen SAR: 380
- Desteklenen SAR kabul oranı: 0.9974
- Desteklenen SAR belirsiz oranı: 0.0026
- Desteklenen SAR yanlış red oranı: 0.0000

- Kilitli desteklenmeyen görüntü: 400
- Desteklenmeyen red oranı: 0.8750
- Desteklenmeyen belirsiz oranı: 0.1075
- Desteklenmeyen yanlış kabul oranı: 0.0175
- Güvenli engelleme oranı: 0.9825

## Grup sonuçları

| Veri | Grup | Örnek | Kabul | Belirsiz | Red | Kabul oranı |
|---|---|---:|---:|---:|---:|---:|
| dartis | nc | 42 | 42 | 0 | 0 | 1.0000 |
| dartis | nw | 225 | 224 | 1 | 0 | 0.9956 |
| dartis | oc | 32 | 32 | 0 | 0 | 1.0000 |
| dartis | ow | 81 | 81 | 0 | 0 | 1.0000 |
| uc_merced | airplane | 100 | 0 | 0 | 100 | 0.0000 |
| uc_merced | harbor | 100 | 1 | 13 | 86 | 0.0100 |
| uc_merced | river | 100 | 6 | 24 | 70 | 0.0600 |
| uc_merced | runway | 100 | 0 | 6 | 94 | 0.0000 |

## Başarı koşulu

Airplane, harbor, river ve runway sınıflarındaki görüntülerin hiçbiri petrol analizine gönderilmemelidir.

`REJECT` ve `UNCERTAIN` kararlarının ikisi de petrol pipeline'ını engeller.

## Bilimsel sınır

Bu input-gate yalnız eğitimde tanımlanan veri alanı için güvenlik katmanıdır. Dünyadaki bütün optik, SAR ve radar görüntülerini kapsayan evrensel bir görüntü tipi doğrulayıcısı değildir.
