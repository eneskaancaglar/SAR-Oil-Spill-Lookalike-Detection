# Hard-Negative Fine-Tuning Karşılaştırması

Seçilen threshold: **0.60**

| Grup | N | Baseline FP piksel | Fine-tuned FP piksel | Baseline >=%1 alarm | Fine-tuned >=%1 alarm | İyileşen görüntü | Kötüleşen görüntü |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 493 | %3.9116 | %1.1854 | %32.25 | %13.59 | %59.84 | %0.20 |
| nc | 71 | %18.6445 | %7.4136 | %87.32 | %70.42 | %91.55 | %0.00 |
| nw | 422 | %1.4328 | %0.1375 | %22.99 | %4.03 | %54.50 | %0.24 |

## Sonuç

- Fine-tuning açık deniz `nw` örneklerinde yanlış alarmları güçlü biçimde azaltmıştır.
- Kıyı içeren `nc` örnekleri temel hata kaynağı olarak kalmıştır.
- Bir sonraki deney kara-deniz ayrımı veya kıyı bölgesi baskılama yöntemlerini incelemelidir.
