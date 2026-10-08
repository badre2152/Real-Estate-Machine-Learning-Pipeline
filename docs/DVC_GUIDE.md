# Guide DVC

DVC est présent dans ce dépôt avec `dvc.yaml` et `.dvc/config`. Il sert à décrire les étapes et les artefacts du pipeline, mais le workflow DVC complet n'est **pas encore confirmé comme exécutable**.

## État du pipeline

Le fichier `dvc.yaml` définit cinq étapes : `extract`, `validate`, `featurize`, `train` et `evaluate`.

| Étape | Rôle déclaré | Réserve |
| --- | --- | --- |
| `extract` | Extraction PostgreSQL vers `data/raw/obt.parquet` | La commande demande `--output`, mais `src/extract.py` ne prend actuellement pas cet argument en charge |
| `validate` | Validation des données et rapport JSON | Dépend de la sortie de l'extraction |
| `featurize` | Préparation des données et features | Dépend des fichiers Parquet et des signatures de fonctions |
| `train` | Exécution de `src/pipeline.py` | Le pipeline lit également PostgreSQL directement |
| `evaluate` | Lecture de `models/results.json` | Dépend des artefacts de training |

Ne pas présenter `dvc repro` comme un processus validé tant que la configuration et les commandes n'ont pas été alignées sur le code.

## Commandes de consultation

Depuis la racine du dépôt, après installation des dépendances :

```bash
dvc dag
dvc status
dvc metrics show
```

Ces commandes décrivent le workflow prévu et nécessitent un environnement DVC correctement configuré. Leur bon fonctionnement n'a pas été vérifié dans cette revue.

## Reproduction

```bash
dvc repro
```

Attention : cette commande est **susceptible d'échouer** à l'étape `extract` en raison de `--output`. Corriger le workflow avant de l'utiliser.

Le dépôt contient déjà `.dvc/config`, donc ne pas lancer `dvc init` une seconde fois sans besoin précis. Aucun stockage distant DVC opérationnel n'est configuré dans la version examinée. `dvc push` et `dvc pull` nécessitent d'abord un remote valide.

Les métriques citées dans les anciens exemples de cette documentation étaient illustratives, pas des performances mesurées et vérifiées du dépôt.
