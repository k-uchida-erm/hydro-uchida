# 実験フォルダのルート
EXPERIMENTS_DIR=./experiments

# 実行対象の実験名（例: exp001）
EXP?=default

# 新しい実験ディレクトリを作る
new:
	mkdir -p $(EXPERIMENTS_DIR)/$(EXP)
	touch $(EXPERIMENTS_DIR)/$(EXP)/main.py
	@echo "Created: $(EXPERIMENTS_DIR)/$(EXP)/main.py"

# 実験を実行（今のEXPをボリュームとしてマウント）
run:
	docker run -it --rm \
	-v $(shell pwd)/$(EXPERIMENTS_DIR)/$(EXP):/usr/src/app \
	--workdir /usr/src/app \
	--name python-ml-uchida python-ml python3 main.py

# イメージを再ビルド
rebuild:
	docker rmi -f python-ml 2>/dev/null || true
	docker build -t python-ml .
