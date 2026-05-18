# 🚀 دليل الـ Deployment — Avito ML API v3

## خيارات الـ Deploy

| الخيار | السعر | الصعوبة | مناسب لـ |
|---|---|---|---|
| **Render.com** | مجاني / 7$/شهر | ⭐ سهل | المشاريع الشخصية والـ demos |
| **Docker local** | مجاني | ⭐⭐ | التطوير والـ testing |
| **VPS (DigitalOcean/Hetzner)** | 5$/شهر | ⭐⭐⭐ | Production حقيقي |

---

## 1. Deploy على Render.com (الأسهل)

### خطوات:
1. عمل push للـ code على GitHub
2. روح لـ https://render.com → New → **Blueprint**
3. ربط الـ repo — Render يقرأ `render.yaml` تلقائياً
4. في Dashboard، أضف المتغيرات السرية:
   - `API_KEYS` ← ضع مفتاحك هنا (مثال: `prod-key-abc123`)
5. Deploy!

### URL ديالك:
```
https://avito-ml-api.onrender.com
```

### Test بعد الـ deploy:
```bash
# Health check
curl https://avito-ml-api.onrender.com/health

# Predict
curl -X POST https://avito-ml-api.onrender.com/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: prod-key-abc123" \
  -d '{"surface_m2": 120, "ville": "Casablanca", "type_bien": "appartement"}'
```

---

## 2. Deploy بـ Docker (محلي أو VPS)

```bash
# Clone الـ repo
git clone https://github.com/your-user/avito-ml .

# إنشاء ملف الـ environment
cp .env.example .env
# غيّر API_KEYS و DB_PASSWORD في .env

# بناء وتشغيل
make docker-build
make docker-run

# تحقق من الـ health
make docker-health

# تشغيل pipeline التدريب
make docker-train
```

---

## 3. CI/CD — GitHub Actions

### Secrets اللي تحتاج تضيفها في GitHub:
```
Settings → Secrets and variables → Actions → New repository secret
```

| Secret | كيفاش تجيبه |
|---|---|
| `DOCKERHUB_USERNAME` | Username ديالك في Docker Hub |
| `DOCKERHUB_TOKEN` | Docker Hub → Account Settings → Security → New Token |
| `RENDER_API_KEY` | Render → Account → API Keys |
| `RENDER_SERVICE_ID` | URL الـ service في Render: `srv-xxxxxxxxxxxx` |

### كيفاش يشتغل:
```
push → main
  ├── test    (pytest + coverage)
  ├── lint    (ruff)
  ├── build   (docker build + push to Docker Hub)
  └── deploy  (trigger Render deploy + smoke tests)
```

---

## 4. متغيرات البيئة

```env
# .env — نسخ من هنا للـ production
API_KEYS=prod-key-change-me          # غيّر هذا!
RATE_LIMIT_PER_MINUTE=60
CORS_ORIGINS=https://your-frontend.com

DB_HOST=localhost
DB_PORT=5433
DB_NAME=real_estate_db
DB_USER=postgres
DB_PASSWORD=strong-password-here    # غيّر هذا!

MODELS_DIR=models
LOG_LEVEL=INFO
```

---

## 5. Nginx (HTTPS في Production)

```bash
# تشغيل مع Nginx
docker compose --profile nginx up -d

# ضع شهادات SSL في nginx/certs/
# - nginx/certs/fullchain.pem
# - nginx/certs/privkey.pem

# مع Let's Encrypt (مجاني):
certbot certonly --standalone -d your-domain.com
```
