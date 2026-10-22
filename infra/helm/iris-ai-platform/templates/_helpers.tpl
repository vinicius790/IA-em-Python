{{- define "iris-ai-platform.fullname" -}}
{{- printf "%s-api" .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "iris-ai-platform.labels" -}}
app.kubernetes.io/name: iris-ai-platform
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
