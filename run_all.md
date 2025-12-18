## -hohup実行-
cd /home/esakit25/work/tensaku
mkdir -p logs/nohup
ts=$(date +%Y%m%d_%H%M%S)

nohup bash -lc "cd /home/esakit25/work/tensaku && bash ./run_all.sh --mode prod \
  --qids Y14_1-2_1_3 --seeds 42,43,44 \
  --print-log-dir \
  --nohup-log logs/nohup/run_all_${ts}.log \
  --pid-file logs/nohup/run_all_${ts}.pid" \
  > logs/nohup/wrapper_${ts}.log 2>&1 &

## -追跡-
tail -F logs/nohup/wrapper_${ts}.log
tail -F logs/nohup/run_all_${ts}.log

## -停止-
kill "$(cat logs/nohup/run_all_*.pid)"

