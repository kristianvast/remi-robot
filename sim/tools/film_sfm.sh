#!/usr/bin/env bash
# Structure-from-motion for the film frames on elegoo-sim-2 (COLMAP 4.2.1, CUDA, conda-forge).
# Usage on the VM: bash film_sfm.sh ~/film/frames_tm ~/film/sfm
# Runs both the global mapper (GLOMAP) and the incremental mapper on one sequential+loop match
# database and prints registration counts for each, so the better model can be chosen.
# SIFT_PEAK (default 0.00667, COLMAP's own) lowers the SIFT contrast threshold for flat frames.
set -euo pipefail
IMG="$1"
OUT="$2"
C=~/colmap-env/bin/colmap
VOCAB=~/film/vocab_tree_faiss_flickr100K_words256K.bin
mkdir -p "$OUT"
[ -s "$VOCAB" ] || curl -sSL -o "$VOCAB" \
  https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words256K.bin
t() { local s=$(date +%s); "$@"; echo "TIME $1 $2 $(( $(date +%s) - s ))s"; }
rm -f "$OUT/db.db"
t $C feature_extractor --database_path "$OUT/db.db" --image_path "$IMG" \
  --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV \
  --FeatureExtraction.use_gpu 1 --SiftExtraction.max_num_features 16000 \
  --SiftExtraction.peak_threshold "${SIFT_PEAK:-0.00667}"
t $C sequential_matcher --database_path "$OUT/db.db" \
  --SequentialMatching.overlap 20 --SequentialMatching.quadratic_overlap 1 \
  --SequentialMatching.loop_detection 1 --SequentialMatching.loop_detection_period 5 \
  --SequentialMatching.loop_detection_num_images 30 --SequentialMatching.vocab_tree_path "$VOCAB" \
  --FeatureMatching.use_gpu 1 --FeatureMatching.guided_matching 1
rm -rf "$OUT/global" "$OUT/incremental"
mkdir -p "$OUT/global" "$OUT/incremental"
t $C global_mapper --database_path "$OUT/db.db" --image_path "$IMG" --output_path "$OUT/global"
t $C mapper --database_path "$OUT/db.db" --image_path "$IMG" --output_path "$OUT/incremental"
for d in "$OUT"/global/* "$OUT"/incremental/*; do
  [ -d "$d" ] || continue
  echo "MODEL $d"
  $C model_analyzer --path "$d" 2>&1 | grep -E 'Registered images|Points|Mean reprojection|Mean track' || true
done
echo SFM_DONE
