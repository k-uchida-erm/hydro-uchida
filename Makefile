# 実験フォルダのルート
EXPERIMENTS_DIR=./experiments

# 実行対象の実験名（例: exp001）
EXP?=default

# 新しい実験ディレクトリを作る
new:
	mkdir -p $(EXPERIMENTS_DIR)/$(EXP)/src
	mkdir -p $(EXPERIMENTS_DIR)/$(EXP)/data
	mkdir -p $(EXPERIMENTS_DIR)/$(EXP)/result
	mkdir -p $(EXPERIMENTS_DIR)/$(EXP)/analysis
	touch $(EXPERIMENTS_DIR)/$(EXP)/main.py
	touch $(EXPERIMENTS_DIR)/$(EXP)/README.md
	touch $(EXPERIMENTS_DIR)/$(EXP)/src/__init__.py
	touch $(EXPERIMENTS_DIR)/$(EXP)/src/loader.py
	touch $(EXPERIMENTS_DIR)/$(EXP)/src/config.py
	touch $(EXPERIMENTS_DIR)/$(EXP)/src/model.py
	touch $(EXPERIMENTS_DIR)/$(EXP)/src/train.py
	touch $(EXPERIMENTS_DIR)/$(EXP)/analysis/check_model.py
	touch $(EXPERIMENTS_DIR)/$(EXP)/analysis/visualize.py
	@echo "Created experiment structure in $(EXPERIMENTS_DIR)/$(EXP)/"
	@echo "  - main.py"
	@echo "  - README.md"
	@echo "  - src/"
	@echo "    - __init__.py"
	@echo "    - loader.py"
	@echo "    - config.py"
	@echo "    - model.py"
	@echo "    - train.py"
	@echo "  - data/"
	@echo "  - result/"
	@echo "  - analysis/"
	@echo "    - check_model.py"
	@echo "    - visualize.py"

# 実験を実行（今のEXPをボリュームとしてマウント）
run:
	docker run -it --rm \
	-v $(shell pwd)/$(EXPERIMENTS_DIR)/$(EXP):/usr/src/app \
	-v $(shell pwd)/$(EXPERIMENTS_DIR)/$(EXP)/result:/usr/src/app/result \
	-v $(shell pwd)/$(EXPERIMENTS_DIR)/$(EXP)/data:/usr/src/app/data \
	--workdir /usr/src/app \
	--name python-ml-uchida python-ml python3 main.py

# モデルを確認
check:
	docker run -it --rm \
	-v $(shell pwd)/$(EXPERIMENTS_DIR)/$(EXP):/usr/src/app \
	-v $(shell pwd)/$(EXPERIMENTS_DIR)/$(EXP)/result:/usr/src/app/result \
	-v $(shell pwd)/$(EXPERIMENTS_DIR)/$(EXP)/data:/usr/src/app/data \
	--workdir /usr/src/app \
	--name python-ml-uchida python-ml python3 analysis/check_model.py $(ARGS)

# 結果を可視化
visualize:
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
