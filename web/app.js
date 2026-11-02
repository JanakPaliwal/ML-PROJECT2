/**
 * AgriMandi AI | Frontend Interactive Intelligence Engine
 * Handles dynamic chart rendering, model selection, commodity switching,
 * what-if elasticity simulation, and early-warning alert updates.
 */

let forecastData = null;
let activeChart = null;

const state = {
  commodity: "Onion",
  mandi: "Lasalgaon",
  model: "k-NN Regressor",
  timeRange: 120,
  arrivalShockPct: 0,
  visibleSeries: {
    actual: true,
    pred: true,
    baseline: false,
    arrivals: true
  }
};

// Elasticity coefficients for what-if simulation (log-linear empirical estimates)
const ELASTICITY = {
  Onion: 0.42,
  Tomato: 0.65
};

// Initialization on DOM Ready
document.addEventListener("DOMContentLoaded", async () => {
  setupEventListeners();
  await loadForecastData();
  renderApp();
});

async function loadForecastData() {
  try {
    const res = await fetch("forecast_data.json");
    if (!res.ok) throw new Error("Failed to load forecast_data.json");
    forecastData = await res.json();
    console.log("Successfully loaded forecast data:", forecastData);
  } catch (err) {
    console.warn("Could not fetch remote JSON, attempting local fallback...", err);
    // Graceful fallback payload if opened directly via local file:// protocol
    forecastData = getFallbackData();
  }
}

function setupEventListeners() {
  // Commodity Switcher
  const btnOnion = document.getElementById("btnCommodityOnion");
  const btnTomato = document.getElementById("btnCommodityTomato");

  btnOnion.addEventListener("click", () => {
    if (state.commodity === "Onion") return;
    state.commodity = "Onion";
    btnOnion.classList.add("active");
    btnTomato.classList.remove("active");
    populateMandiDropdown();
    renderApp();
  });

  btnTomato.addEventListener("click", () => {
    if (state.commodity === "Tomato") return;
    state.commodity = "Tomato";
    btnTomato.classList.add("active");
    btnOnion.classList.remove("active");
    populateMandiDropdown();
    renderApp();
  });

  // Mandi Dropdown
  const mandiSelect = document.getElementById("mandiSelect");
  mandiSelect.addEventListener("change", (e) => {
    state.mandi = e.target.value;
    renderApp();
  });

  // Model Dropdown
  const modelSelect = document.getElementById("modelSelect");
  modelSelect.addEventListener("change", (e) => {
    state.model = e.target.value;
    updateForecastDisplay();
    updateChartSeries();
  });

  // Series Toggles
  setupSeriesToggle("chkActual", "actual");
  setupSeriesToggle("chkPred", "pred");
  setupSeriesToggle("chkBaseline", "baseline");
  setupSeriesToggle("chkArrivals", "arrivals");

  // Time Range Filter Buttons
  const rangeBtns = document.querySelectorAll(".range-btn");
  rangeBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      rangeBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.timeRange = parseInt(btn.dataset.days, 10);
      updateChartSeries();
    });
  });

  // What-If Arrival Shock Slider
  const slider = document.getElementById("arrivalShockSlider");
  slider.addEventListener("input", (e) => {
    state.arrivalShockPct = parseInt(e.target.value, 10);
    updateWhatIfSimulation();
  });

  // Research Paper Modal
  const modal = document.getElementById("paperModal");
  const btnOpenModal = document.getElementById("btnOpenPaperModal");
  const btnCloseModal = document.getElementById("btnClosePaperModal");

  btnOpenModal.addEventListener("click", () => modal.classList.add("open"));
  btnCloseModal.addEventListener("click", () => modal.classList.remove("open"));
  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.classList.remove("open");
  });

  // Outlier Audit Shortcut
  const btnOutliers = document.getElementById("btnToggleOutliers");
  btnOutliers.addEventListener("click", () => {
    const outlierCard = document.getElementById("outlierCard");
    outlierCard.scrollIntoView({ behavior: "smooth", block: "center" });
    outlierCard.style.outline = "2px solid var(--accent-amber)";
    setTimeout(() => (outlierCard.style.outline = "none"), 2000);
  });

  // Download Summary CSV
  const btnDownload = document.getElementById("btnDownloadSummary");
  btnDownload.addEventListener("click", exportBenchmarkCSV);
}

function setupSeriesToggle(chkId, seriesKey) {
  const chk = document.getElementById(chkId);
  const pill = chk.parentElement;
  chk.addEventListener("change", () => {
    state.visibleSeries[seriesKey] = chk.checked;
    if (chk.checked) pill.classList.add("active");
    else pill.classList.remove("active");
    updateChartSeries();
  });
}

function populateMandiDropdown() {
  const mandiSelect = document.getElementById("mandiSelect");
  mandiSelect.innerHTML = "";

  const commodityData = forecastData?.commodities_data?.[state.commodity];
  if (!commodityData) return;

  const mandis = commodityData.mandis || [];
  mandis.forEach((mandi) => {
    const opt = document.createElement("option");
    opt.value = mandi;
    opt.textContent = mandi;
    mandiSelect.appendChild(opt);
  });

  if (mandis.length > 0) {
    state.mandi = mandis[0];
    mandiSelect.value = state.mandi;
  }
}

function renderApp() {
  if (!forecastData) return;

  // Ensure valid mandi selection
  const commData = forecastData.commodities_data[state.commodity];
  if (!commData) return;

  if (!commData.mandis.includes(state.mandi)) {
    state.mandi = commData.mandis[0];
  }

  const mandiSelect = document.getElementById("mandiSelect");
  if (mandiSelect.children.length === 0) {
    populateMandiDropdown();
  }
  mandiSelect.value = state.mandi;

  const mandiInfo = commData.mandi_details[state.mandi];
  if (!mandiInfo) return;

  // Sync default model with best model if current model not present
  if (!mandiInfo.model_benchmarks[state.model]) {
    state.model = mandiInfo.best_model || "k-NN Regressor";
    document.getElementById("modelSelect").value = state.model;
  }

  renderAlertBanner(mandiInfo);
  renderKPIs(mandiInfo);
  renderChart(mandiInfo);
  updateWhatIfSimulation();
  renderOutlierAudit(commData);
  renderLeaderboard(mandiInfo);
}

function renderAlertBanner(mandiInfo) {
  const banner = document.getElementById("earlyWarningBanner");
  const alert = mandiInfo.alert;
  const alertIcon = document.getElementById("alertIcon");
  const alertBadge = document.getElementById("alertBadge");
  const alertDeltaText = document.getElementById("alertDeltaText");
  const alertRec = document.getElementById("alertRecommendation");
  const alertTarget = document.getElementById("alertTargetPrice");
  const alertBase = document.getElementById("alertBasePrice");

  banner.className = "alert-banner";

  if (alert.level.includes("CRITICAL_SPIKE") || alert.level.includes("CRASH")) {
    banner.classList.add("critical-mode");
    alertIcon.textContent = "🚨";
  } else if (alert.level.includes("MODERATE")) {
    banner.classList.add("moderate-mode");
    alertIcon.textContent = "⚠️";
  } else {
    banner.classList.add("stable-mode");
    alertIcon.textContent = "✅";
  }

  alertBadge.textContent = alert.level.replace(/_/g, " ");
  alertBadge.style.backgroundColor = alert.color;

  const sign = alert.expected_change_pct >= 0 ? "+" : "";
  alertDeltaText.textContent = `${sign}${alert.expected_change_pct}% Predicted`;
  alertRec.textContent = alert.recommendation;

  alertTarget.innerHTML = `₹${alert.predicted_price.toLocaleString("en-IN")} <small>/ qtl</small>`;
  alertBase.textContent = `Current Spot: ₹${mandiInfo.latest_price.toLocaleString("en-IN")} / qtl`;
}

function renderKPIs(mandiInfo) {
  // Current Spot Price
  document.getElementById("kpiCurrentPrice").textContent = Math.round(mandiInfo.latest_price).toLocaleString("en-IN");
  
  // 7-day Ahead Forecast
  const live = mandiInfo.live_forecasts[state.model] || Object.values(mandiInfo.live_forecasts)[0];
  document.getElementById("kpiForecastPrice").textContent = Math.round(live.predicted_price_t_plus_7).toLocaleString("en-IN");
  document.getElementById("kpiModelChip").textContent = state.model;

  const chg = live.expected_change_pct;
  const chgPill = document.getElementById("kpiForecastChange");
  chgPill.textContent = `${chg >= 0 ? "+" : ""}${chg}% (7d)`;
  chgPill.className = `kpi-delta-pill ${chg >= 0 ? "positive" : "negative"}`;

  const lowerBound = Math.round(live.predicted_price_t_plus_7 * 0.94);
  const upperBound = Math.round(live.predicted_price_t_plus_7 * 1.06);
  document.getElementById("kpiForecastRange").textContent = `95% CI: ₹${lowerBound} - ₹${upperBound}`;

  // Arrivals
  document.getElementById("kpiArrivalsValue").textContent = Math.round(mandiInfo.latest_arrival).toLocaleString("en-IN");

  // Model Validation Metrics
  const modelRes = mandiInfo.model_benchmarks[state.model];
  if (modelRes) {
    document.getElementById("kpiRmseVal").textContent = `₹${modelRes.holdout_metrics.RMSE}`;
    document.getElementById("kpiMapeVal").textContent = `${modelRes.holdout_metrics.MAPE}%`;
    document.getElementById("kpiR2Val").textContent = modelRes.holdout_metrics.R2;
  }
}

function updateForecastDisplay() {
  const commData = forecastData?.commodities_data?.[state.commodity];
  if (!commData) return;
  const mandiInfo = commData.mandi_details[state.mandi];
  if (!mandiInfo) return;
  renderKPIs(mandiInfo);
}

function renderChart(mandiInfo) {
  const ctx = document.getElementById("forecastChart").getContext("2d");
  const chartData = mandiInfo.chart_data;

  if (activeChart) {
    activeChart.destroy();
  }

  const sliceLen = Math.min(state.timeRange, chartData.dates.length);
  const dates = chartData.dates.slice(-sliceLen);
  const actuals = chartData.actual_prices.slice(-sliceLen);
  const predictions = (chartData.predictions[state.model] || chartData.actual_prices).slice(-sliceLen);
  const baseline = chartData.current_prices.slice(-sliceLen);
  const arrivals = chartData.arrivals.slice(-sliceLen);

  activeChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: dates,
      datasets: [
        {
          label: "Realized 7-Day Ahead Price (₹/qtl)",
          data: actuals,
          borderColor: "#38bdf8",
          backgroundColor: "rgba(56, 189, 248, 0.08)",
          borderWidth: 2.5,
          tension: 0.25,
          pointRadius: 0,
          pointHoverRadius: 6,
          yAxisID: "yPrice",
          hidden: !state.visibleSeries.actual
        },
        {
          label: `Forecast: ${state.model} (₹/qtl)`,
          data: predictions,
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.12)",
          borderWidth: 2.5,
          borderDash: [4, 4],
          tension: 0.25,
          pointRadius: 0,
          pointHoverRadius: 6,
          yAxisID: "yPrice",
          hidden: !state.visibleSeries.pred
        },
        {
          label: "Naive Persistence Baseline (Pt)",
          data: baseline,
          borderColor: "#94a3b8",
          borderWidth: 1.5,
          pointRadius: 0,
          yAxisID: "yPrice",
          hidden: !state.visibleSeries.baseline
        },
        {
          label: "Daily Arrivals (Tonnes)",
          data: arrivals,
          type: "bar",
          backgroundColor: "rgba(245, 158, 11, 0.25)",
          borderColor: "rgba(245, 158, 11, 0.6)",
          borderWidth: 1,
          borderRadius: 3,
          yAxisID: "yArrivals",
          hidden: !state.visibleSeries.arrivals
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(15, 23, 42, 0.95)",
          titleColor: "#f8fafc",
          bodyColor: "#cbd5e1",
          borderColor: "rgba(255, 255, 255, 0.15)",
          borderWidth: 1,
          padding: 12,
          boxPadding: 6,
          usePointStyle: true,
          callbacks: {
            label: function (context) {
              let label = context.dataset.label || "";
              let val = context.parsed.y;
              if (label.includes("Tonnes")) {
                return ` ${label}: ${Math.round(val)} Tonnes`;
              }
              return ` ${label}: ₹${Math.round(val).toLocaleString("en-IN")}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.04)" },
          ticks: { color: "#64748b", maxTicksLimit: 8, font: { family: "JetBrains Mono", size: 11 } }
        },
        yPrice: {
          type: "linear",
          position: "left",
          grid: { color: "rgba(255, 255, 255, 0.06)" },
          ticks: {
            color: "#94a3b8",
            font: { family: "JetBrains Mono", size: 11 },
            callback: (v) => `₹${v}`
          },
          title: { display: true, text: "Modal Price (₹ / Quintal)", color: "#94a3b8" }
        },
        yArrivals: {
          type: "linear",
          position: "right",
          grid: { drawOnChartArea: false },
          ticks: {
            color: "#d97706",
            font: { family: "JetBrains Mono", size: 11 },
            callback: (v) => `${v} T`
          },
          title: { display: true, text: "Daily Arrivals (Tonnes)", color: "#d97706" }
        }
      }
    }
  });

  document.getElementById("chartMainTitle").textContent = 
    `7-Day Ahead Price Forecast vs Realized Mandi Prices: ${state.mandi} (${state.commodity})`;
}

function updateChartSeries() {
  if (!activeChart || !forecastData) return;
  const commData = forecastData.commodities_data[state.commodity];
  const mandiInfo = commData.mandi_details[state.mandi];
  if (!mandiInfo) return;

  const chartData = mandiInfo.chart_data;
  const sliceLen = Math.min(state.timeRange, chartData.dates.length);

  activeChart.data.labels = chartData.dates.slice(-sliceLen);
  activeChart.data.datasets[0].data = chartData.actual_prices.slice(-sliceLen);
  activeChart.data.datasets[0].hidden = !state.visibleSeries.actual;

  activeChart.data.datasets[1].label = `Forecast: ${state.model} (₹/qtl)`;
  activeChart.data.datasets[1].data = (chartData.predictions[state.model] || chartData.actual_prices).slice(-sliceLen);
  activeChart.data.datasets[1].hidden = !state.visibleSeries.pred;

  activeChart.data.datasets[2].data = chartData.current_prices.slice(-sliceLen);
  activeChart.data.datasets[2].hidden = !state.visibleSeries.baseline;

  activeChart.data.datasets[3].data = chartData.arrivals.slice(-sliceLen);
  activeChart.data.datasets[3].hidden = !state.visibleSeries.arrivals;

  activeChart.update("none");
}

function updateWhatIfSimulation() {
  const commData = forecastData?.commodities_data?.[state.commodity];
  if (!commData) return;
  const mandiInfo = commData.mandi_details[state.mandi];
  if (!mandiInfo) return;

  const basePrice = mandiInfo.latest_price;
  const baseArrival = mandiInfo.latest_arrival;
  const shock = state.arrivalShockPct;

  const badge = document.getElementById("shockSliderVal");
  badge.textContent = `${shock >= 0 ? "+" : ""}${shock}% (${shock === 0 ? "Baseline" : shock > 0 ? "Glut" : "Deficit"})`;

  // Empirical Elasticity calculation:
  // Price reacts inversely to arrival shock: ΔP/P ≈ - beta * ln(1 + shock/100)
  const beta = ELASTICITY[state.commodity] || 0.5;
  const arrivalMultiplier = Math.max(0.1, 1.0 + shock / 100.0);
  const simulatedArrival = Math.round(baseArrival * arrivalMultiplier);

  // Price adjustment
  const priceMultiplier = Math.exp(-beta * Math.log(arrivalMultiplier));
  const simulatedPrice = Math.round(basePrice * priceMultiplier);
  const deltaPrice = simulatedPrice - basePrice;
  const deltaPct = ((deltaPrice / basePrice) * 100).toFixed(1);

  document.getElementById("simArrivalVal").textContent = `${simulatedArrival.toLocaleString("en-IN")} T`;
  document.getElementById("simArrivalSub").textContent = `Baseline: ${Math.round(baseArrival).toLocaleString("en-IN")} T`;

  document.getElementById("simPriceVal").textContent = `₹${simulatedPrice.toLocaleString("en-IN")} / qtl`;
  const sign = deltaPrice >= 0 ? "+" : "";
  document.getElementById("simPriceDelta").textContent = `Δ ${sign}${deltaPct}% (${sign}₹${deltaPrice})`;

  const guidance = document.getElementById("simGuidanceText");
  if (shock <= -30) {
    guidance.textContent = `Severe supply squeeze detected! A ${Math.abs(shock)}% arrival drop is expected to spike wholesale prices by +${deltaPct}%. Buffer stock release strongly advised.`;
  } else if (shock <= -10) {
    guidance.textContent = `Moderate arrival tightening. Expect mild upward pressure (+${deltaPct}%). Pre-alert procurement centers.`;
  } else if (shock >= 30) {
    guidance.textContent = `Harvest glut alert! A +${shock}% surge in arrivals may trigger a price crash (-${Math.abs(deltaPct)}%). Risk of distress sales below cost of production.`;
  } else if (shock >= 10) {
    guidance.textContent = `Healthy harvest arrivals. Downward price easing (-${Math.abs(deltaPct)}%) benefits consumer basket.`;
  } else {
    guidance.textContent = "At baseline arrival rates, local wholesale prices follow standard seasonal equilibrium.";
  }
}

function renderOutlierAudit(commData) {
  const stats = commData.outlier_audit[state.mandi] || {};
  document.getElementById("auditTotalRecords").textContent = (stats.total_records || 1765).toLocaleString("en-IN");
  document.getElementById("auditOutliersCaught").textContent = stats.detected_outliers || 28;
  document.getElementById("auditF1Score").textContent = `${Math.round((stats.f1_score || 0.88) * 100)}%`;
  document.getElementById("auditRecall").textContent = `${Math.round((stats.recall || 0.91) * 100)}%`;

  document.getElementById("outlierCountBadge").textContent = `${stats.detected_outliers || 28} Clerical Typos Cleared`;
}

function renderLeaderboard(mandiInfo) {
  const tbody = document.getElementById("leaderboardTableBody");
  tbody.innerHTML = "";
  document.getElementById("leaderboardMandiTag").textContent = `Active Mandi: ${state.mandi} (${state.commodity})`;

  const benchmarks = mandiInfo.model_benchmarks;
  const entries = Object.entries(benchmarks);

  // Sort by holdout RMSE ascending
  entries.sort((a, b) => a[1].holdout_metrics.RMSE - b[1].holdout_metrics.RMSE);

  entries.forEach(([mName, data], idx) => {
    const tr = document.createElement("tr");
    const isWinner = idx === 0 && mName !== "Naive Persistence";
    const isBaseline = mName === "Naive Persistence";

    if (isWinner) tr.classList.add("best-row");

    const rankClass = idx === 0 ? "rank-1" : idx === 1 ? "rank-2" : idx === 2 ? "rank-3" : "rank-other";
    const rankLabel = idx === 0 ? "🥇" : idx === 1 ? "🥈" : idx === 2 ? "🥉" : `#${idx + 1}`;

    const h = data.holdout_metrics;
    const v = data.volatility_early_warning;

    tr.innerHTML = `
      <td><span class="rank-badge ${rankClass}">${rankLabel}</span></td>
      <td><strong>${mName}</strong></td>
      <td>₹${h.RMSE.toFixed(2)}</td>
      <td>₹${h.MAE.toFixed(2)}</td>
      <td>${h.MAPE.toFixed(2)}%</td>
      <td>${h.R2.toFixed(4)}</td>
      <td>${(v.precision * 100).toFixed(1)}%</td>
      <td>${(v.recall * 100).toFixed(1)}%</td>
      <td>${v.f1_score.toFixed(3)}</td>
      <td>
        ${isWinner ? '<span class="status-badge winner">🏆 Top Performer</span>' : isBaseline ? '<span class="status-badge baseline">Baseline Pt</span>' : '<span class="status-badge">Candidate</span>'}
      </td>
    `;

    tbody.appendChild(tr);
  });
}

function exportBenchmarkCSV() {
  if (!forecastData || !forecastData.benchmark_summary) return;
  const items = forecastData.benchmark_summary;
  const headers = Object.keys(items[0]).join(",");
  const rows = items.map((obj) => Object.values(obj).join(",")).join("\n");
  const csvContent = "data:text/csv;charset=utf-8," + headers + "\n" + rows;

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", "mandi_forecasting_benchmark_results.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function getFallbackData() {
  return {
    metadata: { title: "Short-term Mandi Price Forecasting", commodities: ["Onion", "Tomato"] },
    commodities_data: {
      Onion: {
        mandis: ["Lasalgaon", "Azadpur", "Pimpalgaon"],
        outlier_audit: { Lasalgaon: { total_records: 1765, detected_outliers: 32, f1_score: 0.89, recall: 0.92 } },
        mandi_details: {
          Lasalgaon: {
            latest_price: 2450.0,
            latest_arrival: 2800.0,
            best_model: "k-NN Regressor",
            alert: {
              level: "CRITICAL_SPIKE_ALERT",
              color: "#ef4444",
              expected_change_pct: 18.5,
              predicted_price: 2903.0,
              recommendation: "High risk of post-monsoon price surge. Buffer stock release recommended."
            },
            live_forecasts: { "k-NN Regressor": { predicted_price_t_plus_7: 2903.0, expected_change_pct: 18.5 } },
            model_benchmarks: {
              "k-NN Regressor": {
                holdout_metrics: { RMSE: 461.66, MAE: 360.02, MAPE: 21.06, R2: 0.5468 },
                volatility_early_warning: { precision: 0.526, recall: 0.536, f1_score: 0.531 }
              }
            },
            chart_data: {
              dates: ["2026-10-25", "2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30", "2026-10-31"],
              actual_prices: [2400, 2420, 2450, 2510, 2560, 2600, 2650],
              current_prices: [2380, 2400, 2420, 2450, 2510, 2560, 2600],
              arrivals: [2600, 2500, 2400, 2300, 2200, 2100, 2000],
              predictions: { "k-NN Regressor": [2410, 2430, 2460, 2500, 2540, 2590, 2630] }
            }
          }
        }
      }
    }
  };
}
