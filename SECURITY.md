# Segurança

## O que a aplicação faz

- **Autenticação por API key**: uma única dependência (`app/security.py`) protege todos os routers
  `/api`. A comparação é em tempo constante. Ficam públicos só `/healthz`, `/readyz`,
  `/api/v1/health`, `/api/v1/ready`, `/api/v1/metrics`, `/docs` e o console estático.
- **Falha fechada**: com `ENVIRONMENT` diferente de `development`, a aplicação não inicia sem `API_KEY`.
- **Sem SQL arbitrário**: não existe rota que execute consultas enviadas pelo cliente.
- **Limite de corpo**: requisições acima de `MAX_REQUEST_BYTES` recebem 413, contando os bytes
  recebidos (vale também para corpos `chunked`, sem `Content-Length`).
- **Rate limit** por IP de cliente, em memória, com descarte dos contadores vencidos.
- **Request ID**: o `X-Request-ID` do cliente só é aceito se tiver até 64 caracteres `[A-Za-z0-9._-]`.
- **Métricas** usam o template da rota, nunca a URL crua (cardinalidade limitada).
- **Artefatos**: só são desserializados depois de conferir o SHA-256 registrado. O catálogo guarda apenas
  o nome do arquivo, sempre resolvido dentro de `MODEL_DIR`. Com `ARTIFACT_SIGNING_KEY`, cada registro
  é assinado com HMAC-SHA256 e a assinatura é exigida.
- **Console**: dados da API entram na página só por `textContent`; a chave fica no `sessionStorage` da aba.
- **Erros 500** não devolvem detalhes internos; o erro vai para o log com o request id.

## Limites conhecidos

- Os artefatos são pickle (`joblib`). Sem `ARTIFACT_SIGNING_KEY`, quem consegue escrever em `MODEL_DIR`
  consegue trocar artefato e checksum juntos e executar código quando o artefato for carregado.
- Chaves de API estáticas (a mestra e as dos usuários, guardadas como SHA-256, com papéis `admin`,
  `operator` e `reader`), sem expiração nem OIDC. Para exposição pública, coloque a API atrás de um
  gateway com TLS e identidade (OIDC) e use rate limit distribuído.
- O rate limit é por processo e usa o IP visto pela aplicação; atrás de proxy, configure o uvicorn com
  `--proxy-headers --forwarded-allow-ips` para enxergar o IP real.

## Reportar vulnerabilidades

Use o recurso de *security advisories* do GitHub no repositório, ou abra uma issue sem detalhes
exploráveis pedindo um canal privado.

Nunca faça commit de `.env`, chaves, tokens, modelos proprietários ou dados pessoais.
