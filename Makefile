# rk3588-ai targets. Run inside the rk3588 meta-repo, from this directory or with `make -C ai <target>`.
# Board settings (BOARD, BOARD_DIR, DEVICE) come from the meta-repo board.mk. Ported from legacy-final:Makefile.
.PHONY: help test train evaluate golden-test golden-update crosscheck export-npu complexity tables bench-gpu \
	benchmark-npu sync-board board-check bench-board bench-board-cpu bench-board-throughput \
	leaderboard prepare-data

include ../board.mk

PY           := ../.venv/bin/python
LOGTEE       := $(PY) scripts/logtee.py
RUN_DIR       = experiments/runs/$(RUN)
CORE_MASK    ?= AUTO
BOARD_IMAGES ?= data/ppe5/processed/images/test
comma        := ,
# Ultralytics must not pip-install into the venv (it silently changed onnx/onnxslim before); only make sync changes it
export YOLO_AUTOINSTALL = False

# On the board this repo is ~/$(BOARD_DIR)/ai and the venv ~/$(BOARD_DIR)/.venv (meta-repo sync-board-py, setup-board).
# The board has no git checkout: BOARD_PROV passes the PC repo states (--repo NAME:COMMIT:DIRTY).
BOARD_AI   = $(BOARD_DIR)/ai
BOARD_PY   = cd $(BOARD_AI) && ../.venv/bin/python -m
BOARD_PROV = --run-id $(RUN) $(shell $(PY) -m ai.core.provenance board-args)

help:
	@echo "Training and evaluation:"
	@echo "  make train CONFIG=ai/automation/configs/<c>.yaml [SEED=<n>]"
	@echo "  make evaluate RUN=<id> BACKEND=pt|onnx|rknn [PREC=fp16|int8|int8h5]"
	@echo "  make golden-test | golden-update | crosscheck [RUN=<id>] [WEIGHTS=<pt>]"
	@echo "Export and tier 1:"
	@echo "  make export-npu RUN=<id> PREC=fp16|int8|int8h5 [CALIB_FROM=<id>]"
	@echo "  make complexity RUN=<id> | bench-gpu RUN=<id> | tables DATASET=<ds>"
	@echo "  make benchmark-npu RUN=<id> PREC=<p> [CORE_MASK=AUTO] [SESSION=<n>] [SKIP_MEMORY=1]"
	@echo "Board (tier 2):"
	@echo "  make sync-board                copy ai/board, common, deploy scripts and the test images to the board"
	@echo "  make board-check [LOCK=1]      adb, rknn_server, frequency lock"
	@echo "  make bench-board RUN=<id> PREC=<p> [CORE_MASK=AUTO]"
	@echo "  make bench-board-cpu RUN=<id> [THREADS=4]"
	@echo "  make bench-board-throughput RUN=<id> PREC=<p> CORE_MASKS=CORE_0,CORE_1,CORE_2"
	@echo "Data and utilities:"
	@echo "  make prepare-data DATASET=ppe5 | leaderboard DATASET=<ds> | test"

test:
	$(PY) -m pytest -q tests

train:
	$(eval RID := $(shell $(PY) -m ai.automation.pipeline.run $(CONFIG) $(if $(SEED),--seed $(SEED)) --print-run-id 2>/dev/null | tail -n 1))
	@test -n "$(RID)" || $(PY) -m ai.automation.pipeline.run $(CONFIG) $(if $(SEED),--seed $(SEED)) --print-run-id
	$(LOGTEE) experiments/runs/$(RID)/logs/train -- $(PY) -m ai.automation.pipeline.run $(CONFIG) $(if $(SEED),--seed $(SEED))

evaluate:
	$(LOGTEE) $(RUN_DIR)/logs/evaluate_$(BACKEND)$(if $(PREC),_$(PREC)) -- $(PY) -m ai.automation.pipeline.evaluate $(RUN) --backend $(BACKEND) $(if $(PREC),--precision $(PREC) --board $(BOARD) --board-dir $(BOARD_DIR))

golden-test:
	$(LOGTEE) experiments/golden/logs/golden_test -- $(PY) -m ai.automation.pipeline.check_evaluator golden

golden-update:
	$(LOGTEE) experiments/golden/logs/golden_update -- $(PY) -m ai.automation.pipeline.check_evaluator golden --update

crosscheck:
	$(LOGTEE) $(if $(RUN),$(RUN_DIR)/logs/crosscheck,experiments/golden/logs/crosscheck) -- $(PY) -m ai.automation.pipeline.check_evaluator crosscheck $(if $(WEIGHTS),--weights $(WEIGHTS)) $(if $(RUN),--run-id $(RUN))

export-npu:
	RKNN_LOG_LEVEL=3 $(LOGTEE) $(RUN_DIR)/logs/export_$(PREC) -- $(PY) -m ai.automation.pipeline.export_npu $(RUN) --quantize $(if $(filter fp16,$(PREC)),16,8)$(if $(filter int8h5,$(PREC)), --recipe h5 --calib-from $(or $(CALIB_FROM),ppe5_yolov8n_baseline_100ep_s0))$(if $(and $(filter int8,$(PREC)),$(CALIB_FROM)), --calib-from $(CALIB_FROM))

complexity:
	$(LOGTEE) $(RUN_DIR)/logs/complexity -- $(PY) -m ai.automation.pipeline.complexity $(RUN)

tables:
	$(PY) -m ai.automation.report.tables $(DATASET)

bench-gpu:
	$(LOGTEE) $(RUN_DIR)/logs/bench_gpu -- $(PY) -m ai.automation.bench.bench_gpu $(RUN)

benchmark-npu:
	$(LOGTEE) $(RUN_DIR)/logs/tier1_$(PREC)_$(CORE_MASK)$(if $(SESSION),_sess$(SESSION)) -- $(PY) -m ai.automation.bench.benchmark_npu $(RUN) $(DEVICE) --precision $(PREC) --core-mask $(CORE_MASK) $(if $(SESSION),--session $(SESSION)) $(if $(SKIP_MEMORY),--skip-memory)

sync-board:
	$(MAKE) -C .. sync-board-py
	ssh $(BOARD) 'mkdir -p $(BOARD_AI)/$(BOARD_IMAGES)'
	rsync -aL --delete $(BOARD_IMAGES)/ $(BOARD):$(BOARD_AI)/$(BOARD_IMAGES)/

board-check:
	$(PY) -m ai.automation.bench.board_check $(BOARD) $(DEVICE) $(if $(LOCK),--require-lock)

bench-board:
	ssh $(BOARD) 'mkdir -p $(BOARD_AI)/$(RUN_DIR)/weights $(BOARD_AI)/$(RUN_DIR)/bench'
	rsync -a --relative $(RUN_DIR)/weights/./rknn_$(PREC)/*.rknn $(RUN_DIR)/weights/./rknn_$(PREC)/metadata.yaml $(BOARD):$(BOARD_AI)/$(RUN_DIR)/weights/
	$(LOGTEE) $(RUN_DIR)/logs/tier2_$(PREC)_$(CORE_MASK) -- ssh $(BOARD) '$(BOARD_PY) ai.board.bench.bench_npu --model $(RUN_DIR)/weights/rknn_$(PREC) --images $(BOARD_IMAGES) --core-mask $(CORE_MASK) --out $(RUN_DIR)/bench/tier2_$(PREC)_$(CORE_MASK).json $(BOARD_PROV)'
	mkdir -p $(RUN_DIR)/bench && scp $(BOARD):$(BOARD_AI)/$(RUN_DIR)/bench/tier2_$(PREC)_$(CORE_MASK).json $(RUN_DIR)/bench/

bench-board-cpu:
	ssh $(BOARD) 'mkdir -p $(BOARD_AI)/$(RUN_DIR)/weights $(BOARD_AI)/$(RUN_DIR)/bench' && scp $(RUN_DIR)/weights/best.onnx $(BOARD):$(BOARD_AI)/$(RUN_DIR)/weights/
	$(LOGTEE) $(RUN_DIR)/logs/tier2_cpu_onnx_t$(or $(THREADS),4) -- ssh $(BOARD) '$(BOARD_PY) ai.board.bench.bench_cpu_onnx --model $(RUN_DIR)/weights/best.onnx --images $(BOARD_IMAGES) --threads $(or $(THREADS),4) --out $(RUN_DIR)/bench/tier2_cpu_onnx_t$(or $(THREADS),4).json $(BOARD_PROV)'
	mkdir -p $(RUN_DIR)/bench && scp $(BOARD):$(BOARD_AI)/$(RUN_DIR)/bench/tier2_cpu_onnx_t$(or $(THREADS),4).json $(RUN_DIR)/bench/

bench-board-throughput:
	ssh $(BOARD) 'mkdir -p $(BOARD_AI)/$(RUN_DIR)/weights $(BOARD_AI)/$(RUN_DIR)/bench'
	rsync -a --relative $(RUN_DIR)/weights/./rknn_$(PREC)/*.rknn $(RUN_DIR)/weights/./rknn_$(PREC)/metadata.yaml $(BOARD):$(BOARD_AI)/$(RUN_DIR)/weights/
	$(LOGTEE) $(RUN_DIR)/logs/throughput_$(PREC)_$(subst $(comma),-,$(CORE_MASKS)) -- ssh $(BOARD) '$(BOARD_PY) ai.board.bench.bench_throughput --model $(RUN_DIR)/weights/rknn_$(PREC) --images $(BOARD_IMAGES) --core-masks $(CORE_MASKS) --out $(RUN_DIR)/bench/throughput_$(PREC)_$(subst $(comma),-,$(CORE_MASKS)).json $(BOARD_PROV)'
	mkdir -p $(RUN_DIR)/bench && scp $(BOARD):$(BOARD_AI)/$(RUN_DIR)/bench/throughput_$(PREC)_$(subst $(comma),-,$(CORE_MASKS)).json $(RUN_DIR)/bench/

leaderboard:
	cat experiments/leaderboards/$(DATASET).csv

prepare-data:
	$(PY) -m scripts.data.prepare_$(DATASET)
