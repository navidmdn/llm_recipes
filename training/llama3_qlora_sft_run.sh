WANDB_MODE=online WANDB_ENTITY=<your user name> WANDB_PROJECT=<your project name> python llama3_qlora_sft.py\
  --train_file data/train.json\
  --dev_dir data/\
  --output_dir outputs/llama3_qlora_test\
  --do_train\
  --do_eval\
  --model_id meta-llama/Meta-Llama-3-8B-Instruct\
  --use_peft\
  --per_device_train_batch_size 2\
  --per_device_eval_batch_size 4\
  --gradient_accumulation_steps 32\
  --eval_accumulation_steps 4\
  --num_train_epochs 2\
  --save_strategy steps\
  --eval_strategy steps\
  --save_total_limit 3\
  --metric_for_best_model "eval_set_val_loss"\
  --save_steps 5\
  --max_seq_length 2048\
  --logging_steps 2\
  --overwrite_output_dir\
  --report_to wandb\
  --load_best_model_at_end\
  --eval_steps 5\
  --max_new_tokens 1024\
  --learning_rate 2.0e-4\
  --lr_scheduler_type cosine\
  --warmup_ratio 0.1\
  --bf16\
  --lora_alpha 256\
  --lora_r 512\
  --flash_attention