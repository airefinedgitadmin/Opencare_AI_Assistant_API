# care-plan-ai-api on Azure Container Apps

Every push to `main` runs `.github/workflows/deploy.yml`:

```
test (ruff + pytest, incl. migrations on a throwaway Postgres)
  ├─ migrate  → applies new migrations/*.sql to Supabase
  └─ build    → pushes ocareplanaiacr.azurecr.io/care-plan-ai-api:<commit sha>
       deploy → sets the app's secrets, rolls out the new revision, checks /healthz
```

Nothing is deployed unless the tests pass. Settings and secrets come from the
workflow and from GitHub, never by hand, so the app always matches `main`.

| Thing | Name |
|---|---|
| Subscription | `e393d081-6f39-4f62-89f3-795a97ce7a72` |
| Resource group | `rg-oc-careplan-ai` (already exists, `eastus`) |
| Container Registry | `ocareplanaiacr` |
| Container Apps Environment | `env-careplan-ai` |
| Container App | `careplan-ai-api` |

## One-time Azure setup

```bash
az account set --subscription "e393d081-6f39-4f62-89f3-795a97ce7a72"

# 1. Registry — stores the built image
az acr create --resource-group rg-oc-careplan-ai --name ocareplanaiacr --sku Basic

# 2. Build the first image (bootstraps something for the app to start from;
#    after this, GitHub Actions pushes new versions instead)
az acr build --registry ocareplanaiacr --resource-group rg-oc-careplan-ai --image care-plan-ai-api:latest .

# 3. Container Apps environment (shared hosting boundary)
az containerapp env create --name env-careplan-ai --resource-group rg-oc-careplan-ai --location eastus

# 4. The app itself — min-replicas 0 is the scale-to-zero setting that keeps cost near $0 when idle
az containerapp create \
  --name careplan-ai-api \
  --resource-group rg-oc-careplan-ai \
  --environment env-careplan-ai \
  --image ocareplanaiacr.azurecr.io/care-plan-ai-api:latest \
  --target-port 8000 \
  --ingress external \
  --min-replicas 0 \
  --max-replicas 2 \
  --cpu 0.5 --memory 1.0Gi

# 5. Enable the registry's admin account, and let the Container App pull with it.
az acr update --name ocareplanaiacr --admin-enabled true
ACR_USERNAME=$(az acr credential show --name ocareplanaiacr --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name ocareplanaiacr --query "passwords[0].value" -o tsv)
az containerapp registry set \
  --name careplan-ai-api --resource-group rg-oc-careplan-ai \
  --server ocareplanaiacr.azurecr.io \
  --username "$ACR_USERNAME" --password "$ACR_PASSWORD"
```

If `ocareplanaiacr` is already taken globally (ACR names are unique across
*all* of Azure), use `ocareplanaiacr2` consistently everywhere, including
`REGISTRY` in `deploy.yml`.

If this is the very first Container App in this subscription, run once
before step 1:
```bash
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
```

## Azure login for GitHub Actions (OIDC, no stored credentials)

The build and deploy jobs log in to Azure with a short-lived GitHub token, so
no Azure password is stored anywhere. Uses the managed identity
`careplan-ai-api-identity`.

The subject must match what GitHub sends exactly. This repository uses
GitHub's ID-based subject format (owner and repo IDs after `@`); if a login
fails with AADSTS700213, the error shows the subject to register.

```bash
# Trust GitHub Actions runs on this repo's main branch
az identity federated-credential create \
  --name github-main-ids \
  --identity-name careplan-ai-api-identity --resource-group rg-oc-careplan-ai \
  --issuer https://token.actions.githubusercontent.com \
  --subject "repo:airefinedgitadmin@235243447/Opencare_AI_Assistant_API@1340637424:ref:refs/heads/main" \
  --audiences api://AzureADTokenExchange

# Needs Owner / User Access Administrator: let it build images and update the app.
# Scoped to this one resource group, so the pipeline can't touch anything else.
az role assignment create \
  --assignee "$(az identity show --name careplan-ai-api-identity --resource-group rg-oc-careplan-ai --query principalId -o tsv)" \
  --role Contributor \
  --scope /subscriptions/e393d081-6f39-4f62-89f3-795a97ce7a72/resourceGroups/rg-oc-careplan-ai
```

## GitHub variables and secrets

**Repo → Settings → Secrets and variables → Actions.** These three go under
**Variables** (they aren't secret):

| Variable | Value |
|---|---|
| `AZURE_CLIENT_ID` | `az identity show --name careplan-ai-api-identity --resource-group rg-oc-careplan-ai --query clientId -o tsv` |
| `AZURE_TENANT_ID` | `az account show --query tenantId -o tsv` |
| `AZURE_SUBSCRIPTION_ID` | `e393d081-6f39-4f62-89f3-795a97ce7a72` |

No repository secrets are needed: the migrate job reads the database settings
from the Container App (below).

## App settings (on the Container App, not in GitHub)

The app's `CAREPLAN_AI_*` settings and secrets live on the Container App
itself; the pipeline only swaps the image, so they carry over on every
deploy. Set them from a `.env` file (same format as `.env.example`):

```powershell
.\scripts\set-container-app-env.ps1 -EnvFile .env
```

Secret values (`API_KEY`, `AZURE_OPENAI_API_KEY`, `DB_PASSWORD`,
`MEMORY_DB_URL`, `PINECONE_API_KEY`) go into the app's secret store; the rest
become plain env vars, and `CAREPLAN_AI_ENVIRONMENT` is set to `staging`.
`CAREPLAN_AI_API_KEY` must match `careplan-ai-api.key` in the Java backend's
config, or the two can't authenticate to each other.

Never put a real key in this file, `.env.example` or any other tracked file.

## Deploying

- **Normally:** push or merge to `main`.
- **Re-deploy without a code change:**
  Repo → Actions → *Test, build and deploy care-plan-ai-api* → Run workflow.
- **Changing a setting or rotating a secret:** update `.env` and re-run
  `scripts/set-container-app-env.ps1` (it restarts the app so new values apply).
- **Rollback:** Container Apps keeps old revisions. Either revert the commit
  on `main`, or point the app at an older image straight away:
  ```bash
  az containerapp update --name careplan-ai-api --resource-group rg-oc-careplan-ai \
    --image ocareplanaiacr.azurecr.io/care-plan-ai-api:<older commit sha>
  ```
  Migrations aren't rolled back, so a schema change has to stay compatible
  with the previous version of the code.

The deploy job's last step prints the live URL. To look it up yourself:
```bash
az containerapp show --name careplan-ai-api --resource-group rg-oc-careplan-ai \
  --query properties.configuration.ingress.fqdn -o tsv
```

## Cost

The app scales to zero, so it is billed only while handling requests. The
Container Apps free grant (per subscription, per month: 180,000 vCPU-seconds,
360,000 GiB-seconds and 2 million requests) covers about **100 hours of
active time** at 0.5 vCPU / 1 GiB. Past that, Azure **bills automatically at
pay-as-you-go rates**. It does not stop the app. Things that are **not**
covered by the free grant:

- **ACR Basic:** about $5/month whether used or not.
- **Azure OpenAI:** per token.
- **Log Analytics** (created with the environment): free up to 5 GB of
  ingestion a month.

Set a budget alert on `rg-oc-careplan-ai` (Cost Management → Budgets) so a
spike gets noticed.

## Optional, add later

- **Health probes** (readiness/liveness on `/healthz`): the default TCP check
  is enough to start with.
- **Raising `min-replicas`** if cold starts become a problem. Edit
  `--min-replicas` in `deploy.yml`. An always-on replica is billed (at a
  lower idle rate) around the clock, so the free grant runs out much faster.
