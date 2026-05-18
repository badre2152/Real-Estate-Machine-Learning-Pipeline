# 📋 دليل MLflow Registry — Avito ML Pipeline v3

## الفرق بين Tracking والـ Registry

```
Tracking  → يسجّل كل run (metrics, params, plots)
Registry  → يختار أحسن model ويحط فيه label رسمي
```

## Lifecycle ديال الـ Model

```
Training Run
     │
     ▼
register()        → version "None"   (مسجّل لكن غير مفعّل)
     │
     ▼
promote_to_staging()  → "Staging"    (تحت الاختبار)
     │
     ▼ (إذا النتائج كويسة)
promote_to_production() → "Production" (في الخدمة الفعلية)
     │
     ▼ (عند تجاوزه بـ model أحسن)
archive()             → "Archived"   (محفوظ لكن غير مستعمل)
```

## كيفاش يشتغل تلقائياً في الـ Pipeline

```python
# pipeline.py — بعد كل training:
result = auto_register_and_promote(
    run_id         = tracker.run_id,
    model_name     = "avito-regression",
    artifact_path  = "regression_model",
    primary_metric = "reg/R2",          # المقياس الرئيسي
    higher_is_better = True,
)

# النتيجة:
# {"version": "3", "promoted": True,  "reason": "New reg/R2 is better than production"}
# {"version": "4", "promoted": False, "reason": "New reg/R2 did not beat production"}
```

## API Endpoints الجديدة

```bash
# حالة الـ Registry
GET /v1/registry
Headers: X-API-Key: your-key

# Promote يدوي (rollback أو ترقية)
POST /v1/registry/promote?model_name=avito-regression&version=2&stage=Production
Headers: X-API-Key: your-key
```

## Rollback — كيفاش ترجع لـ version سابقة

```bash
# 1. شوف الـ versions المتاحة
make registry-status

# 2. ارجع لـ version 2 مثلاً
make registry-promote MODEL=avito-regression VERSION=2 STAGE=Production
```

## MLflow UI

```bash
make mlflow-ui
# افتح http://localhost:5000
# Models → avito-regression → ستشوف كل الـ versions والـ stages
```
