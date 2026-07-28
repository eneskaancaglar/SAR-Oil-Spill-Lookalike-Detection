# v0.5 Input-Gate Hard-Negative Postmortem

## v0.4 sonucu

- Desteklenmeyen kilitli görüntü: 400
- Yanlış kabul: 7
- Belirsiz: 43
- Doğru red: 350
- Güvenli engelleme oranı: 0.9825

v0.4 güvenlik testi başarısız olduğu için model uygulamaya eklenmemiştir.

## v0.5 kullanımı

Bu görüntüler artık bağımsız test verisi değildir. Tüketilmiş tanı seti olarak v0.5 model geliştirmesinde hard-negative örnekler şeklinde kullanılacaktır.

| Karar | Eğitim ağırlığı | Öncelik |
|---|---:|---:|
| ACCEPT | 5.0 | 3 |
| UNCERTAIN | 3.0 | 2 |
| REJECT | 1.0 | 1 |

## Grup dağılımı

| Grup | Örnek | Yanlış kabul | Belirsiz | Doğru red |
|---|---:|---:|---:|---:|
| river | 100 | 6 | 24 | 70 |
| harbor | 100 | 1 | 13 | 86 |
| runway | 100 | 0 | 6 | 94 |
| airplane | 100 | 0 | 0 | 100 |

## Bilimsel kural

v0.5 değerlendirmesinde bu görüntüler tekrar kilitli test olarak raporlanmayacaktır.

v0.5 için farklı kaynaklardan yeni ve dokunulmamış bir external-test seti oluşturulmalıdır.
