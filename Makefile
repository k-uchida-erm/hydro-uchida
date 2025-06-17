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
	-v $(shell pwd)/$(EXPERIMENTS_DIR)/$(EXP):/usr/src/app \
	-v $(shell pwd)/$(EXPERIMENTS_DIR)/$(EXP)/result:/usr/src/app/result \
	-v $(shell pwd)/$(EXPERIMENTS_DIR)/$(EXP)/data:/usr/src/app/data \
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
	-v $(shell pwd)/$(EXPERIMENTS_DIR)/$(EXP)/data:/usr/src/app/data \
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
	-v $(shell pwd)/$(EXPERIMENTS_DIR)/$(EXP)/data:/usr/src/app/data \
	--workdir /usr/src/app \
	--name python-ml-uchida python-ml python3 analysis/visualize.py $(ARGS)

# イメージをビルド
rebuild:
	docker rm -f python-ml-uchida 2>/dev/null || true
	docker rmi -f python-ml 2>/dev/null || true
	docker build -t python-ml .
