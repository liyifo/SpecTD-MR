import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class VisitMaskPlan:
    visit_index: int
    visit_node_id: int
    diag_nodes: List[int]
    proc_nodes: List[int]
    med_nodes: List[int]

    def has_targets(self) -> bool:
        return bool(self.diag_nodes or self.proc_nodes or self.med_nodes)


def _sample_nodes(nodes: List[int], ratio: float, minimum: int, rng: random.Random) -> List[int]:
    if not nodes or ratio <= 0:
        return []
    k = max(minimum, int(len(nodes) * ratio))
    k = min(k, len(nodes))
    return rng.sample(nodes, k)


def sample_visit_masks(visit_summaries,
                       mask_ratio: Dict[str, float],
                       min_masks: int = 1,
                       seed: Optional[int] = None) -> List[VisitMaskPlan]:
    rng = random.Random(seed)
    plans: List[VisitMaskPlan] = []
    diag_ratio = mask_ratio.get('diag', mask_ratio.get('concept', 0.15))
    proc_ratio = mask_ratio.get('proc', mask_ratio.get('concept', 0.15))
    med_ratio = mask_ratio.get('med', mask_ratio.get('concept', 0.15))

    for visit in visit_summaries:
        if getattr(visit, 'is_terminal', False):
            continue
        diag_nodes = _sample_nodes(visit.diag_nodes, diag_ratio, min_masks, rng)
        proc_nodes = _sample_nodes(visit.proc_nodes, proc_ratio, min_masks, rng)
        med_nodes = _sample_nodes(visit.med_nodes, med_ratio, min_masks, rng)
        plan = VisitMaskPlan(
            visit_index=visit.visit_index,
            visit_node_id=visit.visit_node_id,
            diag_nodes=diag_nodes,
            proc_nodes=proc_nodes,
            med_nodes=med_nodes,
        )
        if plan.has_targets():
            plans.append(plan)
    return plans


def iter_batches(plans: List[VisitMaskPlan], batch_size: int) -> Iterable[List[VisitMaskPlan]]:
    batch: List[VisitMaskPlan] = []
    for plan in plans:
        batch.append(plan)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
