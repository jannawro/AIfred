dep:
	pip-compile > requirements.txt
run:
	docker-compose --env-file=.env up -d
envs:
	@while read -r line; do \
		echo 'export' "$$line"; \
	done < ./.env
