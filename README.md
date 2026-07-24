# ThyroVision MVP

İlk uçtan uca tiroid ultrason analiz demonstrasyonu.

## Akış
1. Görüntü yükleme
2. Kaliper tespiti ve kaldırma
3. Manuel dikdörtgen nodül ROI
4. Shape ve texture analizi
5. Kullanıcı girdili TI-RADS simülasyonu
6. Kural tabanlı demo benign/malignant risk skoru
7. Özet

## Kurulum

Projeyi şu dizine çıkarın:

```text
C:\TRAICK\caliper-removal-lab-ui
```
## Test

```powershell
.\.venv\Scripts\python.exe -m pytest -q


PowerShell:

```powershell
cd C:\TRAICK\caliper-removal-lab-ui
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Önemli
Risk skoru eğitilmiş klinik model değildir. Yalnızca MVP kullanıcı akışını tamamlamak için kullanılan demonstrasyon skorudur.
