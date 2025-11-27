# Makefile for installing SymphonyLearn project.
# Usage:
#   make all


TORCH_SPEC ?= torch
UV_PIP_CMD ?= uv pip install
PYTORCH_BASE_URL ?= https://download.pytorch.org/whl/nightly

PROTOC_VERSION ?= 32.0
PROTOC_ZIP ?= protoc-$(PROTOC_VERSION)-linux-x86_64.zip
PROTOC_URL ?= https://github.com/protocolbuffers/protobuf/releases/download/v$(PROTOC_VERSION)/$(PROTOC_ZIP)
LOCAL_BIN ?= $(HOME)/.local/bin
EXPORT_LINE ?= export PATH=$$PATH:$(LOCAL_BIN)

.PHONY: all setup-env install-torch show-backend clean-protoc-zip

all: setup-env install-torch install-torchtt-ft

setup-env:
	echo "Setting up environment..."
	sudo apt-get update -y
	sudo apt-get install -y unzip curl ca-certificates gnupg wget build-essential pkg-config libssl-dev
	mkdir -p $(HOME)/.local
	wget -q -O $(PROTOC_ZIP) "$(PROTOC_URL)"
	unzip -o $(PROTOC_ZIP) -d $(HOME)/.local
	if ! grep -qx '$(EXPORT_LINE)' $(HOME)/.bashrc; then \
		echo '$(EXPORT_LINE)' >> $(HOME)/.bashrc; \
	fi
	. $(HOME)/.bashrc || true
	sudo apt-get update -y

	curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.noarmor.gpg | sudo tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null
	curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.tailscale-keyring.list | sudo tee /etc/apt/sources.list.d/tailscale.list >/dev/null
	sudo apt-get update -y
	sudo apt-get install -y tailscale

	curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh -s -- -y
	curl -LsSf https://astral.sh/uv/install.sh | sh
	source ~/.bashrc
	uv sync

install-torch:
	@backend=$$( \
		if command -v rocminfo >/dev/null 2>&1 || [ -x /opt/rocm/bin/rocminfo ]; then \
			printf 'rocm7.0'; \
		elif command -v nvidia-smi >/dev/null 2>&1; then \
			cuda_version=$$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: //' | sed 's/ .*//'); \
			if [ -n "$$cuda_version" ]; then \
				major=$${cuda_version%%.*}; \
				minor=$${cuda_version#*.}; \
				if [ "$$major" -gt 13 ]; then \
					printf 'cu130'; \
				elif [ "$$major" -eq 13 ]; then \
					printf 'cu130'; \
				elif [ "$$major" -eq 12 ] && [ "$$minor" -ge 9 ]; then \
					printf 'cu129'; \
				elif [ "$$major" -eq 12 ] && [ "$$minor" -ge 8 ]; then \
					printf 'cu128'; \
				else \
					printf 'cu128'; \
				fi; \
			else \
				printf 'cpu'; \
			fi; \
		else \
			printf 'cpu'; \
		fi \
	); \
	case "$$backend" in \
		cu128|cu129|cu130|rocm7.0|cpu) \
			index_url="$(PYTORCH_BASE_URL)/$$backend"; \
			;; \
		*) \
			echo "[make install-torch] Unknown backend $$backend" >&2; \
			exit 1; \
			;; \
	esac; \
	echo "[make install-torch] Backend: $$backend"; \
	echo "[make install-torch] Index URL: $$index_url"; \
	set -x; \
	$(UV_PIP_CMD) --pre $(TORCH_SPEC) --index-url "$$index_url" --force-reinstall; \
	set +x

install-torchtt-ft:
	$(UV_PIP_CMD) -r torchtitan/requirements.txt
	$(UV_PIP_CMD) ./torchtitan
	$(UV_PIP_CMD) ./torchft

show-backend:
	@make --no-print-directory -s install-torch UV_PIP_CMD="echo Would run: uv pip install" TORCH_SPEC="$(TORCH_SPEC)"

clean-protoc-zip:
	rm -f $(PROTOC_ZIP)