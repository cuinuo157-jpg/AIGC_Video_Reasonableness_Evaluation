运行与批处理脚本：单视频评估、批量评估、可视化与结果汇总入口。

## Web 界面

- `python scripts/run_webui.py --host 127.0.0.1 --port 8080`

启动后访问 `http://127.0.0.1:8080`，可通过浏览器上传视频、选择检测维度并查看可视化结果。

## 单视频调试脚本

- `python scripts/debug_bio_anomaly.py --input <video> --save-vis`
- `python scripts/debug_dynamics.py --input <video> --subject --save-vis`
- `python scripts/debug_iris_tracking.py --input <video> --save-vis --save-video`
- `python scripts/debug_temporal_coherence.py --input <video> --save-det-vis`

## 多卡批量脚本（Dynamics）

已新增 `scripts/run_4gpu_batch.sh`，用于将视频目录自动分片到多张 GPU 并行执行 `debug_dynamics.py`。

- 脚本名为 `run_4gpu_batch.sh`，但 GPU 数量是可配置的。
- 通过 `--gpus` 指定卡列表，worker 数量 = 指定卡数量。

示例（4 卡）：

- `bash scripts/run_4gpu_batch.sh --input-dir /data/videos --gpus 0,1,2,3 --method raft --subject --offline --save-vis`

示例（6 卡）：

- `bash scripts/run_4gpu_batch.sh --input-dir /data/videos --gpus 0,1,2,3,4,5 --method raft --subject --offline`

可选参数：

- `--method raft|farneback`（默认 `raft`）
- `--max-frames <int>`（默认 `60`）
- `--max-side <int>`（默认 `512`）
- `--subject`、`--offline`、`--save-vis`

批处理输出目录：

- `outputs/dynamics_batch/<timestamp>/`
- 包含 `videos.txt`、`shard_*.txt`、`worker_*_gpu*.log`
