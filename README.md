# PBRCELAIN

PyQt6 masaüstü uygulaması: tek bir materyal üzerinde çalışan, **PBR (Physically
Based Rendering) materyal haritaları** üreten sade bir araç. Substance 3D
Designer / Sampler, Materialize ve Quixel Mixer'dan ilham alır, ama çok daha
sade ve tek pencerelidir.

İlk sürümde desteklenen map türleri:

- **Albedo** — yüklenen görsel doğrudan kullanılır.
- **Height** — bir kaynak fotoğraftan, monoküler derinlik tahmini (Depth
  Anything V2 / Marigold) ile üretilir.
- **Normal** — Height map'ten otomatik türetilir (Sobel gradyan tabanlı).
- **Roughness** — arayüzde yer alır, ancak üretim mantığı henüz uygulanmadı
  (placeholder).

Mimari, ileride Metallic/AO gibi ek PBR haritalarının eklenmesine uygun
şekilde tasarlanmıştır (`core/maps.py`, `ui/panels/`).

## Kurulum

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

GPU (CUDA) hızlandırma istiyorsanız `torch`u ayrı kurun (requirements.txt'ten
önce):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Çalıştırma

```bash
venv\Scripts\python main.py
```

Bir proje dosyası doğrudan açılabilir: `venv\Scripts\python main.py proje.pcln`

### `.pcln` dosyalarını çift tıklayarak açma (Windows)

```bash
venv\Scripts\python register_file_association.py
```

Yalnızca mevcut kullanıcı için (`HKEY_CURRENT_USER`, yönetici izni gerekmez)
`.pcln` uzantısını venv'deki `pythonw.exe` + `main.py` ile ilişkilendirir;
konsol penceresi açılmaz, çıktılar `%LOCALAPPDATA%\PBRCELAIN\pbrcelain.log`
dosyasına yazılır. Proje klasörü taşınırsa komutu tekrar çalıştırın.
Kaldırmak için: `venv\Scripts\python register_file_association.py --unregister`

> Uygulamayı mutlaka venv içindeki Python ile başlatın. Sistem Python'unda
> CPU-only bir `torch` kuruluysa, **Device: GPU (CUDA)** seçimi artık sessizce
> CPU'ya düşmez; model yüklenirken hangi Python yorumlayıcısının ve hangi
> `torch` sürümünün kullanıldığını gösteren bir hata verir.

## Arayüz

Pencere iki ana bölüme ayrılır:

- **Project Section (sol, gri)** — Selected Map Preview, Map Type / Upload
  Image / Remove Image kontrolleri, aktif map türüne göre değişen dinamik
  parametre paneli (3 kolonlu, scroll destekli), ve altta **Generate Maps** /
  **Kill Process** butonları.
- **Preview Section (sağ, mor)** — gerçek zamanlı, OpenGL tabanlı bir sphere
  üzerinde materyal önizlemesi, altında sadece görüntüleme ayarlarını
  (Camera / Lighting / Display / Environment) içeren bağımsız bir panel.

## Proje dosyası (`.pcln`)

Uygulama proje tabanlı çalışır. `.pcln` dosyası bir zip container'dır:
kaynak görseller, üretilmiş (cache) map çıktıları ve tüm parametreler
içinde saklanır — proje tekrar açıldığında height map için ağır derinlik
modeli tekrar çalıştırılmaz (cache varsa doğrudan kullanılır).

Toolbar: **New Project / Open Project / Save / Save As / Export Maps / Exit**.

## Height map üretimi ve bilinen model artefaktları

Height panelindeki mantık, orijinal Height Map Generator ile aynıdır:

- Derinlik modeli aileleri:
  - **Depth Anything V2** (Small / Base / Large) — Hızlı, kararlı ve dengeli rölatif ters derinlik (disparity) tahmini.
  - **Marigold** (LCM / v1.1) — Difüzyon (denoising diffusion) tabanlı derinlik modeli; mikroskobik yüzey kabartıları, derzler ve organik dokularda kusursuz mikro-rölyef detayları sağlar.
- Monoküler derinlik modelleri perspektif sahnelerle eğitildiği için düz
  fotoğraflarda "bowl/dome" eğriliği artefaktı oluşturur — **Flatten
  curvature** ve **Edit Curve…** ile düzeltilir.
- **Chunk Estimation** (yüksek maliyetli): görüntüyü örtüşen parçalara
  bölüp her birini modele yerel çözünürlüğünde (Depth Anything: 518px, Marigold: 768px)
  ayrı ayrı vererek daha keskin detay üretir; N+1 model çalıştırması gerektirir.

## Normal map

Height map üretildiği anda **Strength**, **Invert Green (Y)** ve
**Pre-smooth** parametreleriyle otomatik türetilir (Normal panelinde
ayarlanır); Height henüz üretilmemişse panelde bir uyarı gösterilir.

## Roughness

Şimdilik yalnızca arayüzde bir placeholder olarak yer alır; kontroller
devre dışıdır ve **Generate Maps** bu haritayı atlar.

## 3D Preview

Gerçek zamanlı bir UV sphere üzerinde basitleştirilmiş bir Cook-Torrance /
GGX PBR shader'ı ile materyal gösterilir (albedo + normal + roughness).
Height map mevcutsa, görünüm açısına göre uyarlanmış katman sayısıyla çalışan
**Parallax Occlusion Mapping (POM)** de uygulanır; View Options içinden
açılıp kapatılabilir, derinliği ve kalite sınırı ayarlanabilir.
HDRI desteği tam image-based lighting değil, seçilen görselin ortalama
rengiyle ambient/arkaplan tonlamasından ibarettir — sade bir yaklaşıklama.
