
__all__ = [
    'DrugHypergraphPretrainer',
    'NodeFeatureInitializer',
    'build_drug_hypergraph',
    'VisitMaskPlan',
    'sample_visit_masks',
]


def __getattr__(name):
    if name in {'DrugHypergraphPretrainer', 'NodeFeatureInitializer'}:
        from .model import DrugHypergraphPretrainer, NodeFeatureInitializer
        return {
            'DrugHypergraphPretrainer': DrugHypergraphPretrainer,
            'NodeFeatureInitializer': NodeFeatureInitializer,
        }[name]
    if name == 'build_drug_hypergraph':
        from .data_builder import build_drug_hypergraph
        return build_drug_hypergraph
    if name in {'VisitMaskPlan', 'sample_visit_masks'}:
        from .masking import VisitMaskPlan, sample_visit_masks
        return {
            'VisitMaskPlan': VisitMaskPlan,
            'sample_visit_masks': sample_visit_masks,
        }[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
