# Guide de contribution — ML Pipeline

Merci de contribuer ! Voici les règles à suivre.

---

## Workflow Git

```
main        ← Branche de production (stable, CI verte obligatoire)
develop     ← Branche d'intégration
feature/*   ← Nouvelles fonctionnalités (ex: feature/add-lightgbm)
fix/*       ← Corrections de bugs
```

**Toujours partir de `develop` :**

```bash
git checkout develop
git pull origin develop
git checkout -b feature/ma-nouvelle-feature
```

---

## Environnement de développement

```bash
# Cloner le repo
git clone https://github.com/badre2152/real-estate-pipeline
cd Machine_Learning_Pipeline

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Installer les dépendances de dev
make install-dev

# Copier et configurer l'env
cp .env.example .env
# Remplir les credentials PostgreSQL dans .env
```

---

## Tests

Tous les tests doivent passer avant de pusher :

```bash
make test          # Tests rapides
make test-cov      # Tests + couverture (doit être >= 75%)
```

Règles pour les nouveaux tests :
- Un test par comportement (pas par fonction)
- Noms explicites : `test_<quoi>_<condition>_<résultat attendu>`
- Utiliser des fixtures pytest pour les données partagées
- Ne jamais dépendre de la base de données dans les tests unitaires — utiliser des mocks

---

## Style de code

Le projet utilise **ruff** pour le linting :

```bash
make lint
```

Conventions :
- PEP 8 — longueur max : 100 caractères
- Docstrings sur toutes les fonctions publiques
- Type hints sur les arguments et retours
- Logging plutôt que print()

---

## Ajouter un nouveau modèle ML

1. Ajouter le modèle dans `get_regression_models()` ou `get_classification_models()`
2. Ajouter sa grille d'hyperparamètres dans `optimize_model()`
3. Vérifier que `evaluate_*()` gère correctement sa sortie
4. Ajouter un test dans `tests/`

---

## Structure d'un commit

```
type(scope): description courte

Corps optionnel (si nécessaire)

Refs: #issue-number
```

Types : `feat`, `fix`, `refactor`, `test`, `docs`, `ci`, `chore`

Exemples :
```
feat(regression): ajouter LightGBM comme modèle candidat
fix(prepare): corriger la division par zéro dans prix_par_m2
test(features): ajouter tests pour add_luxury_score
```

---

## Pull Request

- Titre clair et descriptif
- Description des changements
- Référence à l'issue concernée
- CI verte obligatoire (tests + lint)
- Au moins une review avant merge
