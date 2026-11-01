# Workflow-CI

Kriteria 3 submission Membangun Sistem Machine Learning (Dicoding).

Setiap ada perubahan di folder `MLProject/`, GitHub Actions akan:
1. Melatih ulang model lewat `mlflow run MLProject`
2. Menyimpan artefak MLflow (folder `mlruns/`) ke repository ini dan sebagai GitHub Artifact
3. Membangun Docker image dengan `mlflow models build-docker`
4. Push image ke Docker Hub

Secrets yang dibutuhkan (Settings > Secrets and variables > Actions):
- `DOCKER_HUB_USERNAME`
- `DOCKER_HUB_ACCESS_TOKEN`
