# lmeva_on_mac
ローカル環境にある`https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/gguf.py`と差し替えて、下記のコマンドを実行する:
```bash
% poetry run python -m llama_cpp.server\
     --n_gpu_layers 25\
     --n_ctx 2048 \
     --model /path_to_model/model.gguf
```

```bash
% poetry run lm-eval
    --model gguf
    --model_args base_url=http://localhost:8000
    --tasks japanese_leaderboard
    --device mps
    --output_path path_to_out
    --log_sample
    --batch_size 1
    --limit 10
```
