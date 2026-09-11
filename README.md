# InsightTree — Adaptive Evidence-Driven Tree Retrieval

Implémentation Clean Architecture du workflow **Adaptive Evidence-Driven Tree Retrieval**.

## Flux

1. Ingestion de documents hétérogènes.
2. Q0 → Information Need.
3. Evidence Action Engine:
   - planification,
   - réutilisation du pool global,
   - retrieval dense/BM25,
   - outils déterministes,
   - validation,
   - mise à jour du pool.
4. Si Q0 est insuffisant:
   - création de l'Evidence Tree,
   - génération d'hypothèses,
   - génération de Q1–Q4 indépendantes,
   - exploration adaptative par priorité.
5. Pour chaque branche:
   - information need,
   - Evidence Action Engine,
   - validation,
   - force,
   - gain d'information,
   - mise à jour des hypothèses,
   - résolution ou génération d'enfants,
   - pruning.
6. Cross-branch reasoning si Q0 reste insuffisant.
7. Synthèse finale:
   - sélection des preuves fortes,
   - support/contradiction,
   - fiabilité des sources,
   - confiance/incertitude,
   - conclusion et citations.

## Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

Installer Ollama puis préparer les modèles configurés dans `.env`.

## Exécution

Le mode `session` est recommandé pour le prototype mémoire:

```bash
python main.py session
```

Puis:

```text
> ingest document.pdf rapport.docx donnees.xlsx
> ask Pourquoi les ventes ont-elles baissé ?
> quit
```

Les commandes `ingest` et `ask` séparées créent deux processus et les index mémoire ne persistent pas.

## Structure

```text
domain/
application/
config/
infrastructure/
tests/
main.py
requirements.txt
```

Le stockage vectoriel et le repository sont volontairement en mémoire pour le prototype. Ils peuvent être remplacés par Qdrant/pgvector/PostgreSQL sans modifier les services applicatifs, grâce aux ports.
