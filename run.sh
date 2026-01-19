python -m pretraining.pretrain \
  --struct-emb-path data/MIMIC-III/emb/struct_deepwalk.pt \
  --text-emb-path data/MIMIC-III/emb/text_embeddings.pt \
  --logic-emb-path data/MIMIC-III/emb/logic_embeddings.pt \
  --records-path data/MIMIC-III/records_final.pkl \
  --voc-path data/MIMIC-III/voc_final.pkl \
  --batch-size 1 \
  --feature-dim 64 \
  --mask-mode vanilla \
  --mask-token-prob 0.8 \
  --mask-random-prob 0.1 \
  --device cuda:4 \
  --save-checkpoint \
  --dropout 0 \
  --weight-decay 0 \
  --num-layers 3 \
  --mlp-layers 2 \
  --text-pca-dim 128 \
  --project-node-embeds \
  --heads 4 \
  --warmup1-epochs 1 \
  --warmup2-epochs 1 \
  --epochs 1 \
  --warmup2-patience 10 \
  --full-patience 50 \
  --cluster-weight 2.0 \
  --alignment-weight 0.1


python -m recommender.downstream \
  --struct-emb-path data/MIMIC-III/emb/struct_deepwalk.pt \
  --text-emb-path data/MIMIC-III/emb/text_embeddings.pt \
  --logic-emb-path data/MIMIC-III/emb/logic_embeddings.pt \
  --pretrain-checkpoint pretrain_logs/xxxx/stage1_final.pt \
  --concept-cooccurrence-path data/MIMIC-III/concept_cooccurrence.pkl \
  --epochs 30 \
  --batch-size 32 \
  --seq-hidden-dim 128 \
  --readout-mode shared \
  --num-experts 5 \
  --lr-stage1 1e-3 \
  --lr 5e-4 \
  --device cuda:4



python -m pretraining.pretrain \
  --struct-emb-path data/MIMIC-IV/emb/struct_deepwalk.pt \
  --text-emb-path data/MIMIC-IV/emb/text_embeddings.pt \
  --logic-emb-path data/MIMIC-IV/emb/logic_embeddings.pt \
  --records-path data/MIMIC-IV/records_final.pkl \
  --voc-path data/MIMIC-IV/voc_final.pkl \
  --batch-size 1 \
  --feature-dim 64 \
  --mask-mode vanilla \
  --mask-token-prob 0.8 \
  --mask-random-prob 0.1 \
  --device cuda:5 \
  --save-checkpoint \
  --dropout 0 \
  --weight-decay 0 \
  --num-layers 3 \
  --mlp-layers 2 \
  --text-pca-dim 128 \
  --project-node-embeds \
  --heads 4 \
  --warmup1-epochs 1 \
  --warmup2-epochs 1 \
  --epochs 1 \
  --warmup2-patience 10 \
  --full-patience 50 \
  --cluster-weight 2.0 \
  --alignment-weight 0.1


python -m recommender.downstream \
  --struct-emb-path data/MIMIC-IV/emb/struct_deepwalk.pt \
  --text-emb-path data/MIMIC-IV/emb/text_embeddings.pt \
  --logic-emb-path data/MIMIC-IV/emb/logic_embeddings.pt \
  --pretrain-checkpoint pretrain_logs/xxxx/stage1_final.pt \
  --concept-cooccurrence-path data/MIMIC-IV/concept_cooccurrence.pkl \
  --epochs 30 \
  --batch-size 32 \
  --seq-hidden-dim 128 \
  --readout-mode shared \
  --num-experts 5 \
  --lr-stage1 1e-3 \
  --lr 5e-4 \
  --device cuda:5

