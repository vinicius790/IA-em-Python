# Instala o chart Helm da plataforma num cluster Kubernetes já existente.
# O Secret com a API_KEY é criado fora do Terraform (para a chave não ir para o state):
#   kubectl -n <namespace> create secret generic iris-ai-secrets --from-literal=API_KEY=<valor>

terraform {
  required_version = ">= 1.6.0"

  required_providers {
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.17"
    }
  }
}

variable "kubeconfig" {
  description = "Caminho do kubeconfig do cluster de destino."
  type        = string
  default     = "~/.kube/config"
}

variable "namespace" {
  description = "Namespace onde a plataforma será instalada."
  type        = string
  default     = "iris-ai"
}

variable "image_repository" {
  description = "Repositório da imagem construída a partir do Dockerfile deste projeto."
  type        = string
}

variable "image_tag" {
  description = "Tag da imagem; vazio usa o appVersion do chart."
  type        = string
  default     = ""
}

variable "existing_secret" {
  description = "Nome do Secret que contém API_KEY."
  type        = string
  default     = "iris-ai-secrets"
}

provider "helm" {
  kubernetes {
    config_path = var.kubeconfig
  }
}

resource "helm_release" "iris_ai" {
  name             = "iris-ai"
  namespace        = var.namespace
  create_namespace = true
  chart            = "${path.module}/../helm/iris-ai-platform"

  values = [
    yamlencode({
      image = {
        repository = var.image_repository
        tag        = var.image_tag
      }
      existingSecret = var.existing_secret
    })
  ]
}

output "release" {
  description = "Nome, namespace e versão do release instalado."
  value = {
    name      = helm_release.iris_ai.name
    namespace = helm_release.iris_ai.namespace
    version   = helm_release.iris_ai.version
  }
}
