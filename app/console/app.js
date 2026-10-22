"use strict";

// Todo dado vindo da API entra na página por textContent (nunca innerHTML).
const $ = (selector) => document.querySelector(selector);
const CHAVE = "iris-api-key";
let apiKey = sessionStorage.getItem(CHAVE) || "";

async function api(path, { method = "GET", body } = {}) {
  const headers = {};
  if (apiKey) headers["X-API-Key"] = apiKey;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const response = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  const text = await response.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = text;
  }
  if (!response.ok) {
    const detail = data && data.detail !== undefined ? data.detail : text;
    const message = typeof detail === "string" ? detail : JSON.stringify(detail);
    throw new Error(response.status === 401 ? "API key ausente ou inválida." : `${response.status}: ${message}`);
  }
  return data;
}

function el(tag, text, attrs = {}) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = String(text);
  for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
  return node;
}

function avisar(message) {
  $("#aviso").textContent = message;
}

function mostrar(selector, value) {
  $(selector).textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function numero(value, casas = 3) {
  return typeof value === "number" ? value.toFixed(casas) : "n/d";
}

function preencherSelect(select, names, selected) {
  select.replaceChildren(...names.map((name) => el("option", name, { value: name })));
  if (names.includes(selected)) select.value = selected;
}

async function promover(model) {
  try {
    await api(`/api/v2/models/${encodeURIComponent(model.name)}/promote`, {
      method: "POST",
      body: { version: model.version, stage: "production", notes: "promovido pelo console" },
    });
    avisar(`${model.alias} promovido para production.`);
    await carregar();
  } catch (error) {
    avisar(`Promoção recusada: ${error.message}`);
  }
}

function linhaModelo(model) {
  const metrics = model.metrics || {};
  const row = el("tr");
  row.append(
    el("th", model.alias, { scope: "row" }),
    el("td", model.stage),
    el("td", numero(metrics.accuracy)),
    el("td", numero(metrics.f1_macro)),
    el("td", numero(metrics.latency_ms, 2)),
  );
  const action = el("td");
  if (model.stage !== "production") {
    const button = el("button", "Promover", { type: "button", "aria-label": `Promover ${model.alias} para production` });
    button.addEventListener("click", () => promover(model));
    action.append(button);
  }
  row.append(action);
  return row;
}

async function carregar() {
  try {
    const [platform, meta, models] = await Promise.all([
      api("/api/v2/platform"),
      api("/api/v3/meta"),
      api("/api/v1/models"),
    ]);
    $("#versao").textContent = `versão ${platform.version} · ${platform.environment}`;
    const cards = [
      ["Versões registradas", platform.model_count],
      ["Banco de dados", platform.database ? "online" : "offline"],
      ["Autenticação", platform.auth_required ? "API key exigida" : "aberta (development)"],
      ["Experimentos", platform.tracking_backend],
      ["Modelos disponíveis", meta.available_models.length],
    ];
    $("#cards").replaceChildren(
      ...cards.map(([title, value]) => {
        const card = el("div", undefined, { class: "card" });
        card.append(el("dt", title), el("dd", value));
        return card;
      }),
    );
    $("#modelos").replaceChildren(...models.map(linhaModelo));
    if (!models.length) {
      const row = el("tr");
      row.append(el("td", "Nenhuma versão registrada. Treine um modelo acima.", { colspan: "6" }));
      $("#modelos").append(row);
    }
    const registered = [...new Set(models.map((model) => model.name))];
    preencherSelect($("#modelo-treino"), meta.available_models, "xgboost");
    preencherSelect($("#modelo-predicao"), registered, registered[0]);
    preencherSelect($("#modelo-monitor"), registered, registered[0]);
    avisar("");
  } catch (error) {
    avisar(error.message);
  }
}

async function acompanharJob(jobId) {
  for (let attempt = 0; attempt < 120; attempt += 1) {
    const job = await api(`/api/v2/jobs/${jobId}`);
    if (job.status === "completed" || job.status === "failed") return job;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error("O job não terminou em 2 minutos; consulte /api/v2/jobs.");
}

$("#form-chave").addEventListener("submit", (event) => {
  event.preventDefault();
  apiKey = $("#chave").value.trim();
  // A chave fica só na sessão desta aba; some ao fechar o navegador.
  sessionStorage.setItem(CHAVE, apiKey);
  $("#chave").value = "";
  carregar();
});

$("#atualizar").addEventListener("click", carregar);

$("#form-treino").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const job = await api("/api/v2/training/async", {
      method: "POST",
      body: { model_name: $("#modelo-treino").value, cv_folds: 3 },
    });
    mostrar("#saida-treino", `Job ${job.job_id} em execução…`);
    const done = await acompanharJob(job.job_id);
    const metrics = (done.result && done.result.metrics) || {};
    mostrar("#saida-treino", {
      status: done.status,
      erro: done.error,
      accuracy: metrics.accuracy,
      f1_macro: metrics.f1_macro,
      latency_ms: metrics.latency_ms,
    });
    await carregar();
  } catch (error) {
    mostrar("#saida-treino", error.message);
  }
});

$("#dry-run").addEventListener("click", async () => {
  try {
    mostrar(
      "#saida-treino",
      await api("/api/v4/pipelines/dry-run", { method: "POST", body: { model_name: $("#modelo-treino").value } }),
    );
  } catch (error) {
    mostrar("#saida-treino", error.message);
  }
});

$("#form-predicao").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.target);
  const body = { model_name: $("#modelo-predicao").value };
  for (const [name, value] of form.entries()) body[name] = Number(value);
  try {
    const [prediction, explained] = await Promise.all([
      api("/api/v1/predict", { method: "POST", body }),
      api("/api/v2/explain", { method: "POST", body }),
    ]);
    $("#resultado").textContent =
      `Classe prevista: ${prediction.predicted_class} (confiança ${numero(prediction.confidence)}, ` +
      `${prediction.model_name} v${prediction.model_version}). ` +
      `Probabilidade média de base: ${numero(explained.explanation.base_value)}.`;
    $("#shap").replaceChildren(
      ...explained.explanation.contributions.map((item) => {
        const row = el("tr");
        const sign = item.shap_value >= 0 ? "+" : "";
        row.append(el("th", item.feature, { scope: "row" }), el("td", item.value), el("td", sign + numero(item.shap_value, 4)));
        return row;
      }),
    );
    $("#tabela-shap").hidden = false;
  } catch (error) {
    $("#resultado").textContent = error.message;
    $("#tabela-shap").hidden = true;
  }
});

$("#form-monitor").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    mostrar("#saida-monitor", await api(`/api/v2/monitoring/${encodeURIComponent($("#modelo-monitor").value)}`));
  } catch (error) {
    mostrar("#saida-monitor", error.message);
  }
});

carregar();
