run:
	docker-compose --env-file=.env up -d
envs:
	@while read -r line; do \
		echo 'export' "$$line"; \
	done < ./.env
format:
	python -m black ./
venv:
	ln -s $$(poetry env info -p) .venv
