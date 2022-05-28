.PHONY: _next_ next delete dot instance ssh stop

## create instance
instance:
	gcloud compute instances create ${INSTANCE_NAME} \
	  --zone=${ZONE} \
	  --image-family="common-cu113" \
	  --image-project=deeplearning-platform-release \
	  --maintenance-policy=TERMINATE \
	  --accelerator="type=nvidia-tesla-a100,count=1" \
	  --metadata="install-nvidia-driver=True" \
	  --boot-disk-type="pd-standard" \
	  --boot-disk-size="200GB" \
	  --machine-type="a2-highgpu-1g" \
	  --preemptible \
	  --service-account="vertex-ai@ai-lab-sandbox.iam.gserviceaccount.com" \
	  --network="outbound-internet" \
	  --subnet="subnet1"

## scp
scp:
	gcloud compute scp --recurse --zone=${ZONE} ./* ${INSTANCE_NAME}:~/dezero

## scp
scp-git:
	gcloud compute scp --recurse --zone=${ZONE} ./.gitignore ${INSTANCE_NAME}:~/dezero


## ssh
ssh:
	gcloud beta compute ssh --zone ${ZONE} ${INSTANCE_NAME} --project ${PROJECT_ID}

## stop
stop:
	gcloud compute instances stop ${INSTANCE_NAME} --zone ${ZONE}

## make next
_next_:
	make_next

## delete last
delete:
	delete_last

## make empty next
next:
	touch_next

## dot files
dot:
	dot sample.dot -T png -o sample.png


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')

