# 実験フォルダのルート
EXPERIMENTS_DIR=./experiments

# 実行対象の実験名（例: exp001）
EXP?=default

# 新しい実験ディレクトリを作る
new:
	mkdir -p $(EXPERIMENTS_DIR)/$(EXP)/src
	mkdir -p $(EXPERIMENTS_DIR)/$(EXP)/data
	touch $(EXPERIMENTS_DIR)/$(EXP)/main.py
	touch $(EXPERIMENTS_DIR)/$(EXP)/README.md
	touch $(EXPERIMENTS_DIR)/$(EXP)/src/__init__.py
	touch $(EXPERIMENTS_DIR)/$(EXP)/src/loader.py
	touch $(EXPERIMENTS_DIR)/$(EXP)/src/config.py
	touch $(EXPERIMENTS_DIR)/$(EXP)/src/model.py
	touch $(EXPERIMENTS_DIR)/$(EXP)/src/train.py
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

# 実験を実行（今のEXPをボリュームとしてマウント）
run:
	docker run -it --rm \
	-v $(shell pwd)/$(EXPERIMENTS_DIR)/$(EXP):/usr/src/app \
	--workdir /usr/src/app \
	--name python-ml-uchida python-ml python3 main.py

# イメージをビルド
rebuild:
	docker rm -f python-ml-uchida 2>/dev/null || true
	docker rmi -f python-ml 2>/dev/null || true
	docker build -t python-ml .
