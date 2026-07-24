# SAR Oil Spill Segmentation — Final Deney Sonuçları

## 1. Problem

SAR görüntülerinde petrol tabakalarının segmentasyonu gerçekleştirilmiştir.
Temel sorun, petrol tabakalarının koyu deniz bölgeleri, düşük geri saçılım
alanları, kıyılar ve kara dokuları ile karıştırılmasıdır.

## 2. Veri kümeleri

### Refined Deep-SAR Oil Spill — SOS

- 8.070 görüntü-mask çifti
- 256x256 gri seviye görüntüler
- Sentinel-1 ve PALSAR örnekleri
- Train: 6.455
- Validation: 1.615

### DARTIS no-oil

- Toplam 2.290 petrolsüz görüntü
- 869 farklı kaynak SAR sahnesi
- NW: açık deniz negatifleri
- NC: kıyı ve kara içeren negatifler

DARTIS verisi kaynak sahne ID bilgisine göre ayrılmıştır:

- Hard-negative train: 1.304 görüntü, 623 sahne
- Hard-negative validation: 493 görüntü, 122 sahne
- External test: 493 görüntü, 124 sahne
- Sahne overlap: 0

## 3. Baseline model

Model: küçük U-Net

- Giriş kanalı: 1
- Çıkış kanalı: 1
- Yaklaşık parametre sayısı: 1,94 milyon
- Loss: BCEWithLogits + Dice
- Seçilen çalışma threshold'u: 0.60

Baseline model SOS validation üzerinde yüksek başarı göstermesine rağmen
DARTIS no-oil görüntülerinde yüksek yanlış alarm üretmiştir.

## 4. Hard-negative fine-tuning

SOS eğitim görüntüleri, DARTIS no-oil görüntülerinden çıkarılan 256x256
negatif crop'larla birlikte kullanılmıştır.

Bu işlem özellikle açık deniz lookalike yanlış alarmlarını azaltmıştır.

## 5. Coastal hard-negative fine-tuning

DARTIS NC görüntüleri ağırlıklı örnekleme ile daha sık gösterilmiştir.

- NC örnek ağırlığı: 8
- NW örnek ağırlığı: 1
- SOS örnek ağırlığı: 1
- Öğrenme oranı: 5e-5
- Epoch: 3
- Başlangıç modeli: hard-negative fine-tuned model

## 6. Final sonuçlar — Threshold 0.60

| Model | SOS Dice | SOS Recall | DARTIS >=%1 alarm | FP pixel | NC >=%1 | NW >=%1 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.8334 | 0.8745 | %32.25 | %3.9116 | %87.32 | %22.99 |
| Hard-negative | 0.8263 | 0.8201 | %13.59 | %1.1854 | %70.42 | %4.03 |
| Coastal hard-negative | 0.8232 | 0.8482 | %12.58 | %0.9130 | %60.56 | %4.50 |

## 7. Baseline ile final model arasındaki değişim

- Genel ciddi yanlış alarm yaklaşık %61 azaltılmıştır.
- Yanlış pozitif piksel oranı yaklaşık %77 azaltılmıştır.
- Açık deniz ciddi yanlış alarmı yaklaşık %80 azaltılmıştır.
- Kıyı ciddi yanlış alarmı yaklaşık %31 azaltılmıştır.
- SOS Dice yalnızca 0.0102 azalmıştır.
- SOS recall 0.8482 seviyesinde korunmuştur.

## 8. Post-processing deneyi

Küçük bağlı bileşenlerin silinmesi güvenli fakat sınırlı bir iyileşme
sağlamıştır.

Görüntü sınırına bağlı büyük bileşenlerin silinmesi başarısız olmuştur.
Gerçek petrol maskelerinin de görüntü kenarına temas edebilmesi nedeniyle
SOS Dice ve recall ciddi biçimde düşmüştür.

## 9. Sonuç

Sahne bazlı hard-negative fine-tuning, modelin veri kümesine özgü koyuluk
ipuçlarına bağımlılığını azaltmış ve bağımsız no-oil görüntülerindeki yanlış
alarmları önemli ölçüde düşürmüştür.

Coastal hard-negative fine-tuning, kıyı içeren görüntülerde ek iyileşme
sağlarken segmentasyon başarısını büyük ölçüde korumuştur.

Final model:

checkpoints/final_model/oil_spill_unet_t060.pth

Final threshold:

0.60

## 10. Sınırlamalar

- DARTIS no-oil görüntülerinde gerçek segmentasyon maskesi bulunmamaktadır.
- NC görüntülerindeki kara bölgeleri için ayrı bir land-sea maskesi yoktur.
- SOS görüntüleri ile DARTIS görüntülerinin boyut ve görüntü formatları farklıdır.
- Kıyı yanlış alarmları azaltılmış olsa da tamamen çözülmemiştir.
- External test sonuçları görüldükten sonra model üzerinde yeni ayar yapılmamalıdır.

## 11. Gelecek çalışma

- Ayrı bir kara-deniz segmentasyon modeli
- Sentinel-1 land mask ürünlerinin kullanılması
- Lookalike sınıflarının ayrıca etiketlenmesi
- Focal veya Tversky tabanlı loss deneyleri
- Kaynak sahne bazlı yeni ve daha büyük bağımsız test seti
- Petrol türü ve alan büyüklüğü analizi
