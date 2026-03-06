python eval_vlm_multisegment.py \
  --episodes "$(ls dataset | grep '^2f_store_juheon_setup_single_object_pick_place' | paste -sd, -)" \
  --model gemini-3.1-flash-lite-preview \
  --video-fps 4 \
  --max-intervals 2 \


  python eval_vlm_multisegment.py \
  --episodes "$(ls dataset | grep '^2f_store_juheon_setup_single_object_pick_place' | paste -sd, -)" \
  --model gemini-3.1-flash-lite-preview \
  --video-fps 4 \
  --max-intervals 2 \
  --workers 6