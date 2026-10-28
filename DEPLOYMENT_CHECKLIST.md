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

## Service principal for GitHub Actions (needs Owner / User Access Administrator)

The deploy job logs in to Azure to update the app. Creating the login needs a
role assignment, so someone with Owner (or User Access Administrator) on the
resource group runs this once:

```bash
az ad sp create-for-rbac \
  --name "github-careplan-ai-api" \
  --role Contributor \
  --scopes /subscriptions/e393d081-6f39-4f62-89f3-795a97ce7a72/resourceGroups/rg-oc-careplan-ai \
  --json-auth
```

The whole JSON it prints is the `AZURE_CREDENTIALS` secret. The scope is just
this one resource group, so the pipeline can't touch anything else.

## GitHub secrets

**Repo → Settings → Secrets and variables → Actions → New repository secret.**

| Secret | Value |
|---|---|
| `AZURE_CREDENTIALS` | JSON from the service principal step |
| `ACR_USERNAME` | `az acr credential show --name ocareplanaiacr --query username -o tsv` |
| `ACR_PASSWORD` | `az acr credential show --name ocareplanaiacr --query "passwords[0].value" -o tsv` |
| `SUPABASE_DB_URL` | Supabase → Connect → Session pooler (port 5432) URI, with the password filled in |
| `CAREPLAN_AI_API_KEY` | New random value (`openssl rand -hex 32`). Set the **same** value as `careplan-ai-api.key` in the Java backend's config, or the two can't authenticate to each other |
| `AZURE_OPENAI_API_KEY` | Azure portal → the OpenAI resource → Keys and Endpoint |
| `AZURE_OPENAI_ENDPOINT` | `https://<resource>.openai.azure.com` |
| `PINECONE_API_KEY` | Optional, only if `CAREPLAN_AI_LIBRARY_PROVIDER` is switched to `pinecone` |

Never put a real key in this file, `.env.example` or any other tracked file.

## Deploying

- **Normally:** push or merge to `main`.
- **Re-deploy without a code change** (for example after rotating a secret):
  Repo → Actions → *Test, build and deploy care-plan-ai-api* → Run workflow.
- **Changing a setting:** edit the `--set-env-vars` list in `deploy.yml`.
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
