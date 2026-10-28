"""Care plan style library endpoints (OPENCARE-51).

Called only by the Java backend (X-Api-Key):

* on care plan save / confirm / archive -> POST   (index or re-tag, within 5 min)
* on care plan delete                   -> DELETE
* System Admin "exclude from AI"        -> POST .../exclusion
* go-live backfill                      -> POST in batches of up to 100

With CAREPLAN_AI_LIBRARY_PROVIDER=none these succeed with `enabled: false`, so
the backend's save path never depends on the library being switched on.
"""

from fastapi import APIRouter, Depends

from careplan_ai.api.dependencies import require_api_key
from careplan_ai.library import service
from careplan_ai.library.models import LibraryBatchIn, LibraryExclusionIn, LibraryIndexResult

router = APIRouter(
    prefix="/library/homes/{home_id}/care-plans",
    tags=["care-plan-library"],
    dependencies=[Depends(require_api_key)],
)


@router.post("", response_model=LibraryIndexResult)
async def index_care_plans(home_id: int, payload: LibraryBatchIn) -> LibraryIndexResult:
    return await service.index_care_plans(home_id, payload.items, home_name=payload.home_name)


@router.delete("/{care_plan_id}", response_model=LibraryIndexResult)
async def remove_care_plan(home_id: int, care_plan_id: int) -> LibraryIndexResult:
    return await service.remove_care_plan(home_id, care_plan_id)


@router.post("/{care_plan_id}/exclusion", response_model=LibraryIndexResult)
async def set_exclusion(
    home_id: int, care_plan_id: int, payload: LibraryExclusionIn
) -> LibraryIndexResult:
    return await service.set_excluded(home_id, care_plan_id, payload.excluded)
