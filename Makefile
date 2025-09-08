# 実験フォルダのルート
EXPERIMENTS_DIR=./experiments

# 新しい実験ディレクトリを作る
new:
	@# 既存のv*ディレクトリから最大のバージョン番号を取得
	@MAX_VERSION=$$(ls -d $(EXPERIMENTS_DIR)/v* 2>/dev/null | grep -o '[0-9]*$$' | sort -n | tail -n 1 || echo 0); \
	NEXT_VERSION=$$((MAX_VERSION + 1)); \
	NEW_DIR="$(EXPERIMENTS_DIR)/v$$NEXT_VERSION"; \
	PREV_DIR="$(EXPERIMENTS_DIR)/v$$MAX_VERSION"; \
	\
	if [ $$MAX_VERSION -eq 0 ]; then \
		echo "Error: No previous version found. Please create v1 manually first."; \
		exit 1; \
	fi; \
	\
	echo "Creating v$$NEXT_VERSION from v$$MAX_VERSION"; \
	cp -r "$$PREV_DIR" "$$NEW_DIR"; \
	echo "Created v$$NEXT_VERSION in $$NEW_DIR/"

# 最新バージョンを取得
LATEST_VERSION := $(shell ls -d $(EXPERIMENTS_DIR)/v* 2>/dev/null | grep -o '[0-9]*$$' | sort -n | tail -n 1)
# 実行対象の実験名（指定がない場合は最新バージョン）
EXP ?= v$(LATEST_VERSION)

# 実験を実行（今のEXPをボリュームとしてマウント）
run:
	@if [ -z "$(LATEST_VERSION)" ]; then \
		echo "Error: No version found in $(EXPERIMENTS_DIR)"; \
		exit 1; \
	fi
	@echo "Running experiment $(EXP)"
	docker run -it --rm \
	--memory=8g \
	--memory-swap=16g \
	-v $(shell pwd)/$(EXPERIMENTS_DIR)/$(EXP):/usr/src/app \
	-v $(shell pwd)/$(EXPERIMENTS_DIR)/$(EXP)/result:/usr/src/app/result \
	-v $(shell pwd)/data:/usr/src/app/global_data \
	--workdir /usr/src/app \
	--name python-ml-uchida python-ml python3 main.py

# モデルを確認
check:
	@if [ -z "$(LATEST_VERSION)" ]; then \
		echo "Error: No version found in $(EXPERIMENTS_DIR)"; \
		exit 1; \
	fi
	@echo "Checking model in $(EXP)"
	docker run -it --rm \
	-v $(shell pwd)/$(EXPERIMENTS_DIR)/$(EXP):/usr/src/app \
	-v $(shell pwd)/$(EXPERIMENTS_DIR)/$(EXP)/result:/usr/src/app/result \
	-v $(shell pwd)/data:/usr/src/app/global_data \
	--workdir /usr/src/app \
	--name python-ml-uchida python-ml python3 analysis/check_model.py $(ARGS)

# 結果を可視化
visualize:
	@if [ -z "$(LATEST_VERSION)" ]; then \
		echo "Error: No version found in $(EXPERIMENTS_DIR)"; \
		exit 1; \
	fi
	@echo "Visualizing results in $(EXP)"
	docker run -it --rm \
	-v $(shell pwd)/$(EXPERIMENTS_DIR)/$(EXP):/usr/src/app \
	-v $(shell pwd)/$(EXPERIMENTS_DIR)/$(EXP)/result:/usr/src/app/result \
	-v $(shell pwd)/data:/usr/src/app/global_data \
	--workdir /usr/src/app \
	--name python-ml-uchida python-ml python3 analysis/visualize.py $(ARGS)

# イメージをビルド
rebuild:
	docker rm -f python-ml-uchida 2>/dev/null || true
	docker rmi -f python-ml 2>/dev/null || true
	docker build -t python-ml .

# ========= Ablation 3-commands =========
# start: 作業用ワークスペース/ケースDIR作成（編集は ablation/<CASE>/work を対象）
# finish: work を実行→結果をケースDIRへ集約→差分要約→ワーク/ロックを掃除
# abort: 実行せずワーク/ロックを掃除

ABL_DIR=$(EXPERIMENTS_DIR)/$(EXP)/ablation/$(CASE)
ABL_WORK=$(ABL_DIR)/work
ABL_LOCK=$(ABL_DIR)/.lock

ablation-start:
	@if [ -z "$(EXP)" ] || [ -z "$(CASE)" ]; then \
		echo "Usage: make ablation-start EXP=v5 CASE=case_name"; \
		exit 1; \
	fi
	@if [ -f "$(ABL_LOCK)" ]; then \
		echo "Lock exists: $(ABL_LOCK). Run ablation-finish or ablation-abort first."; exit 1; \
	fi
	@echo "[Ablation] init $(EXP) $(CASE)"; \
	mkdir -p "$(ABL_DIR)" "$(ABL_WORK)"; \
	# ベース(本家)→work へコピー（result/ablationは除外）
	rsync -a --delete --exclude result --exclude ablation $(EXPERIMENTS_DIR)/$(EXP)/ $(ABL_WORK)/; \
	date '+%Y-%m-%d %H:%M:%S' > "$(ABL_LOCK)"; \
	if [ ! -f "$(ABL_DIR)/README.txt" ]; then \
		echo "Case: $(CASE)" > "$(ABL_DIR)/README.txt"; \
		echo "Started: $$(date '+%Y-%m-%d %H:%M:%S')" >> "$(ABL_DIR)/README.txt"; \
		echo "\n[Planned changes / Notes]" >> "$(ABL_DIR)/README.txt"; \
		echo "- " >> "$(ABL_DIR)/README.txt"; \
	fi; \
	echo "[Ablation] Ready. Edit here: $(ABL_WORK). Then run: make ablation-finish EXP=$(EXP) CASE=$(CASE)"

ablation-finish:
	@if [ -z "$(EXP)" ] || [ -z "$(CASE)" ]; then \
		echo "Usage: make ablation-finish EXP=v5 CASE=case_name"; \
		exit 1; \
	fi
	@if [ ! -f "$(ABL_LOCK)" ]; then \
		echo "No lock: $(ABL_LOCK). Run ablation-start first."; exit 1; \
	fi
	@echo "[Ablation] finishing $(EXP) $(CASE)"; \
	# work を実行し、成果物はケースDIRに保存
	docker run -it --rm \
		--memory=8g \
		--memory-swap=16g \
		-v $(shell pwd)/$(ABL_WORK):/usr/src/app \
		-v $(shell pwd)/$(ABL_DIR):/usr/src/app/result \
		-v $(shell pwd)/data:/usr/src/app/global_data \
		--workdir /usr/src/app \
		--name python-ml-uchida python-ml python3 main.py; \
	mkdir -p "$(ABL_DIR)"; \
	python3 tools/ablation_summarize.py --base "$(EXPERIMENTS_DIR)/$(EXP)" --edited "$(ABL_WORK)" --result_dir "$(ABL_DIR)" --out "$(ABL_DIR)"; \
	rm -rf "$(ABL_WORK)" "$(ABL_LOCK)"; \
	echo "[Ablation] Completed. Output: $(ABL_DIR)."

ablation-abort:
	@if [ -z "$(EXP)" ] || [ -z "$(CASE)" ]; then \
		echo "Usage: make ablation-abort EXP=v5 CASE=case_name"; \
		exit 1; \
	fi
	@if [ ! -f "$(ABL_LOCK)" ]; then \
		echo "No lock: $(ABL_LOCK). Nothing to abort."; exit 0; \
	fi
	@echo "[Ablation] aborting $(EXP) $(CASE)"; \
	rm -rf "$(ABL_WORK)" "$(ABL_LOCK)"; \
	echo "[Ablation] Cleaned up."
