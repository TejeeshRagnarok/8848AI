"use strict";
/*
 * Restaurant dashboard.  Talks only to /api/v1/dashboard/* with the HttpOnly
 * session cookie; the staff key is sent once at sign-in and never stored.
 * All text is inserted with textContent (never innerHTML) so menu or
 * customer text cannot inject markup.  Live updates: Server-Sent Events,
 * falling back to polling every 15 s if the stream drops.
 */
(() => {
  const API = "/api/v1/dashboard";
  const PAGE = 50;
  const state = {
    view: "orders", currency: "INR", orders: [], ordersTotal: 0, menu: [],
    lowStock: 5, openOrderId: null, events: null, pollTimer: null, refreshTimer: null,
  };
  const $ = (id) => document.getElementById(id);

  // ── Helpers ──────────────────────────────────────────────────────────────
  function el(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else if (value !== undefined && value !== null && value !== false) node.setAttribute(key, value === true ? "" : value);
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }
  const money = (paise) => {
    const sign = paise < 0 ? "-" : "";
    const abs = Math.abs(paise);
    const rupees = Math.floor(abs / 100).toLocaleString("en-IN");
    return `${state.currency === "INR" ? "₹" : state.currency + " "}${sign}${rupees}.${String(abs % 100).padStart(2, "0")}`;
  };
  const time = (iso) => new Date(iso).toLocaleString(undefined, { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
  function until(iso) {
    const mins = Math.round((new Date(iso) - Date.now()) / 60000);
    return mins > 0 ? `expires in ${mins} min` : "expired (releasing)";
  }
  function toast(message, bad = false) {
    const t = el("div", { class: "toast" + (bad ? " bad" : ""), text: message });
    $("toasts").append(t);
    setTimeout(() => t.remove(), bad ? 6000 : 3500);
  }
  function errorText(body, status) {
    if (body && typeof body.detail === "string") return body.detail;
    if (body && Array.isArray(body.detail)) return body.detail.map((d) => d.msg).join(" ");
    return `Request failed (${status}).`;
  }

  async function api(path, { method = "GET", body } = {}) {
    const headers = {};
    if (method !== "GET") headers["X-Requested-With"] = "dashboard";
    if (body !== undefined) headers["Content-Type"] = "application/json";
    const res = await fetch(API + path, {
      method, headers, credentials: "same-origin", body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (res.status === 401 && path !== "/session") { showLogin(); throw new Error("Please sign in again."); }
    const data = res.status === 204 ? null : await res.json().catch(() => null);
    if (!res.ok) throw new Error(errorText(data, res.status));
    return data;
  }

  // ── Badges ───────────────────────────────────────────────────────────────
  const SOURCE = {
    AI_VOICE: ["AI voice", "b-violet"], AI_CHAT: ["AI chat", "b-info"], WALK_IN_POS: ["Walk-in / POS", "b-warn"],
    API: ["API", ""], UNKNOWN: ["Unknown", ""],
  };
  const STATUS = {
    pending: ["Awaiting payment", "b-warn"], confirmed: ["Confirmed", "b-info"], preparing: ["Preparing", "b-violet"],
    ready: ["Ready", "b-ok"], completed: ["Completed", ""], cancelled: ["Cancelled", "b-bad"],
  };
  const PAYMENT = {
    pending: ["Pending", "b-warn"], succeeded: ["Paid (demo)", "b-ok"], failed: ["Failed", "b-bad"],
    expired: ["Expired", "b-bad"], cancelled: ["Cancelled", ""], refunded: ["Refunded (demo)", "b-info"],
  };
  const badge = (map, key) => { const [label, cls] = map[key] || [key || "–", ""]; return el("span", { class: `badge ${cls}`, text: label }); };

  function stockBadge(item) {
    if (!item.is_active) return el("span", { class: "badge", text: "Retired" });
    if (!item.is_available) return el("span", { class: "badge b-bad", text: "Switched off" });
    if (item.available_quantity === null) return el("span", { class: "badge b-ok", text: "Available" });
    if (item.available_quantity <= 0) return el("span", { class: "badge b-bad", text: "Sold out" });
    if (item.available_quantity <= state.lowStock) return el("span", { class: "badge b-warn", text: "Low stock" });
    return el("span", { class: "badge b-ok", text: "In stock" });
  }

  // ── Dialogs ──────────────────────────────────────────────────────────────
  /** Opens the dialog; onOk(form) returns a promise; errors stay in the dialog. */
  function openDialog({ title, body, okText = "Save", danger = false, onOk }) {
    const dialog = $("dialog");
    $("dialog-title").textContent = title;
    $("dialog-body").replaceChildren(...[].concat(body));
    $("dialog-error").hidden = true;
    const ok = $("dialog-ok");
    ok.textContent = okText;
    ok.className = "btn " + (danger ? "danger" : "primary");
    const form = $("dialog-form");
    form.onsubmit = async (event) => {
      if (event.submitter && event.submitter.value === "cancel") return;
      event.preventDefault();
      ok.disabled = true;
      try { await onOk(form); dialog.close(); }
      catch (err) { $("dialog-error").textContent = err.message; $("dialog-error").hidden = false; }
      finally { ok.disabled = false; }
    };
    dialog.showModal();
  }
  const confirmAction = (title, message, okText, run, danger = false) =>
    openDialog({ title, body: el("p", { text: message }), okText, danger, onOk: run });
  const field = (label, input) => el("label", {}, label, input);

  // ── Sign-in ──────────────────────────────────────────────────────────────
  function showLogin() {
    stopLive();
    $("app-view").hidden = true;
    $("drawer").hidden = true;
    $("login-view").hidden = false;
    $("login-key").value = "";
    $("login-key").focus();
  }
  async function showApp(info) {
    state.currency = info.currency;
    $("restaurant-name").textContent = info.restaurant_name;
    document.title = `${info.restaurant_name} · Dashboard`;
    $("login-view").hidden = true;
    $("app-view").hidden = false;
    await refreshAll();
    startLive();
  }
  $("login-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    $("login-error").hidden = true;
    try {
      const slug = $("login-slug").value.trim();
      const info = await api("/session", { method: "POST", body: { staff_key: $("login-key").value, restaurant_slug: slug || null } });
      $("login-key").value = "";
      await showApp(info);
    } catch (err) {
      $("login-error").textContent = err.message;
      $("login-error").hidden = false;
    }
  });
  $("sign-out").addEventListener("click", async () => {
    try { await api("/session", { method: "DELETE" }); } catch (_) { /* signed out anyway */ }
    showLogin();
  });

  // ── Live updates ─────────────────────────────────────────────────────────
  function setLive(mode) {
    $("live-dot").className = "dot" + (mode === "live" ? " on" : mode === "poll" ? " slow" : "");
    $("live-text").textContent = mode === "live" ? "Live" : mode === "poll" ? "Refreshing every 15 s" : "Offline";
  }
  function scheduleRefresh() {
    clearTimeout(state.refreshTimer);
    state.refreshTimer = setTimeout(() => refreshAll({ quiet: true }), 300);
  }
  function startLive() {
    stopLive();
    if (!("EventSource" in window)) return startPolling();
    const source = new EventSource(API + "/events", { withCredentials: true });
    state.events = source;
    source.onopen = () => { setLive("live"); clearInterval(state.pollTimer); state.pollTimer = null; };
    source.addEventListener("change", scheduleRefresh);
    source.onerror = () => { setLive("poll"); startPolling(); };
  }
  function startPolling() {
    if (state.pollTimer) return;
    state.pollTimer = setInterval(() => refreshAll({ quiet: true }), 15000);
  }
  function stopLive() {
    if (state.events) state.events.close();
    state.events = null;
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    setLive("off");
  }

  async function refreshAll({ quiet = false } = {}) {
    await Promise.all([loadSummary(), loadOrders({ quiet }), loadMenu({ quiet })]);
    if (state.openOrderId) openOrder(state.openOrderId, { quiet: true });
  }

  // ── Summary ──────────────────────────────────────────────────────────────
  async function loadSummary() {
    try {
      const s = await api("/summary");
      state.lowStock = s.low_stock_threshold;
      $("m-active").textContent = s.active_orders;
      $("m-today").textContent = s.orders_today;
      $("m-unpaid").textContent = s.awaiting_payment;
      $("m-low").textContent = s.low_stock;
      $("m-out").textContent = s.out_of_stock;
    } catch (_) { /* cards keep their last values */ }
  }

  // ── Orders ───────────────────────────────────────────────────────────────
  function orderQuery(offset) {
    const params = new URLSearchParams({ limit: PAGE, offset });
    const q = $("o-search").value.trim();
    if (q) params.set("q", q);
    if ($("o-status").value) params.set("status", $("o-status").value);
    if ($("o-source").value) params.set("source", $("o-source").value);
    if ($("o-active").checked) params.set("active_only", "true");
    return params.toString();
  }
  async function loadOrders({ quiet = false, append = false } = {}) {
    const stateBox = $("orders-state");
    if (!quiet && !append) { stateBox.className = "state"; stateBox.textContent = "Loading orders…"; }
    try {
      const page = await api("/orders?" + orderQuery(append ? state.orders.length : 0));
      state.orders = append ? state.orders.concat(page.items) : page.items.slice(0, Math.max(PAGE, quiet ? state.orders.length : PAGE));
      state.ordersTotal = page.total;
      renderOrders();
    } catch (err) {
      if (!quiet) { stateBox.className = "state error"; stateBox.textContent = err.message; }
    }
  }
  function renderOrders() {
    const rows = state.orders.map((o) => el("tr", { class: "clickable", tabindex: 0, onclick: () => openOrder(o.id),
        onkeydown: (e) => { if (e.key === "Enter") openOrder(o.id); } },
      el("td", {}, el("strong", { text: o.order_number })),
      el("td", { text: time(o.created_at) }),
      el("td", { text: o.customer_name || "Not provided" }),
      el("td", {}, badge(SOURCE, o.source)),
      el("td", { text: o.items.map((i) => `${i.quantity}× ${i.item_name}`).join(", ") }),
      el("td", { class: "num", text: money(o.total_paise) }),
      el("td", {}, o.payment ? badge(PAYMENT, o.payment.status) : el("span", { class: "muted", text: "–" })),
      el("td", {}, badge(STATUS, o.status)),
      el("td", { class: "small muted", text: o.status === "pending" && o.reservation_expires_at ? until(o.reservation_expires_at) : "" }),
    ));
    $("orders-body").replaceChildren(...rows);
    const box = $("orders-state");
    box.className = "state";
    box.textContent = rows.length ? "" : "No orders match these filters yet.";
    $("orders-count").textContent = state.ordersTotal ? `Showing ${state.orders.length} of ${state.ordersTotal}` : "";
    $("orders-more").hidden = state.orders.length >= state.ordersTotal;
  }
  let searchTimer;
  $("o-search").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => loadOrders(), 300); });
  for (const id of ["o-status", "o-source", "o-active"]) $(id).addEventListener("change", () => loadOrders());
  $("orders-more").addEventListener("click", () => loadOrders({ append: true }));

  async function openOrder(id, { quiet = false } = {}) {
    let o;
    try { o = await api("/orders/" + id); }
    catch (err) { if (!quiet) toast(err.message, true); return; }
    state.openOrderId = id;
    $("d-title").textContent = `Order ${o.order_number}`;
    const contact = o.customer_phone || "Not provided";
    const details = el("dl", { class: "kv" },
      el("dt", { text: "Placed" }), el("dd", { text: new Date(o.created_at).toLocaleString() }),
      el("dt", { text: "Channel" }), el("dd", {}, badge(SOURCE, o.source)),
      el("dt", { text: "Status" }), el("dd", {}, badge(STATUS, o.status)),
      el("dt", { text: "Payment" }), el("dd", {}, o.payment ? badge(PAYMENT, o.payment.status) : "–"),
      el("dt", { text: "Customer" }), el("dd", { text: o.customer_name || "Not provided" }),
      el("dt", { text: "Contact number" }), el("dd", { text: contact }),
      o.status === "pending" && o.reservation_expires_at ? [el("dt", { text: "Reservation" }), el("dd", { text: until(o.reservation_expires_at) })] : null,
      o.cancel_reason ? [el("dt", { text: "Cancelled because" }), el("dd", { text: o.cancel_reason.replace("_", " ") })] : null,
      o.notes ? [el("dt", { text: "Notes" }), el("dd", { text: o.notes })] : null,
    );
    const lines = el("table", { class: "lines" },
      el("tbody", {}, o.items.map((i) => el("tr", {},
        el("td", { text: `${i.quantity} × ${i.item_name}` }),
        el("td", { class: "num small muted", text: `${money(i.unit_price_paise)} each` }),
        el("td", { class: "num", text: money(i.line_total_paise) })))),
      el("tfoot", {}, el("tr", {}, el("td", { text: "Total" }), el("td"), el("td", { class: "num", text: money(o.total_paise) }))));
    const hints = [];
    if (o.source === "AI_VOICE") hints.push("Placed by the voice agent. The contact number was given or confirmed during the call; caller ID is not proof of identity.");
    if (o.payment) hints.push("Payments are DEMO only: no real money is charged or refunded.");
    $("d-body").replaceChildren(details, lines, ...hints.map((h) => el("p", { class: "hint", text: h })));
    $("d-actions").replaceChildren(...orderActions(o));
    $("drawer").hidden = false;
  }
  $("d-close").addEventListener("click", () => { $("drawer").hidden = true; state.openOrderId = null; });

  function orderActions(o) {
    const act = (label, cls, run, confirm) => el("button", { class: `btn ${cls}`, text: label, onclick: () => {
      const go = async () => { await run(); toast(`${o.order_number}: ${label.toLowerCase()} done.`); await refreshAll({ quiet: true }); };
      if (confirm) confirmAction(confirm.title, confirm.message, label, go, confirm.danger);
      else go().catch((err) => toast(err.message, true));
    } });
    const status = (s) => () => api(`/orders/${o.id}/status`, { method: "POST", body: { status: s } });
    const buttons = [];
    if (o.status === "pending" && o.payment && o.payment.status === "pending") {
      buttons.push(act("Mark paid (demo)", "primary", () => api(`/orders/${o.id}/mark-paid`, { method: "POST" }),
        { title: "Record payment", message: `Record ${money(o.total_paise)} as paid for ${o.order_number}? This is a demo payment; the reserved stock becomes sold.` }));
    }
    if (o.status === "confirmed") buttons.push(act("Start preparing", "primary", status("preparing")));
    if (o.status === "preparing") buttons.push(act("Mark ready", "primary", status("ready")));
    if (o.status === "ready") buttons.push(act("Complete", "primary", status("completed")));
    if (!["completed", "cancelled"].includes(o.status)) {
      const effect = o.status === "pending" ? "Its reserved stock is released and the pending payment cancelled."
        : o.status === "confirmed" ? "Its stock goes back on hand and the demo payment is refunded."
        : "Preparation has started, so the stock stays used; the demo payment is refunded.";
      buttons.push(act("Cancel order", "danger", () => api(`/orders/${o.id}/cancel`, { method: "POST" }),
        { title: `Cancel ${o.order_number}?`, message: effect, danger: true }));
    }
    return buttons.length ? buttons : [el("span", { class: "muted small", text: "No further actions for this order." })];
  }

  // ── Walk-in sale ─────────────────────────────────────────────────────────
  $("walkin-open").addEventListener("click", async () => {
    await loadMenu({ quiet: true });
    const sellable = state.menu.filter((i) => i.is_active && i.orderable);
    if (!sellable.length) { toast("Nothing on the menu is available to sell right now.", true); return; }
    const qty = new Map();
    const list = el("div", { class: "pick-list" }, sellable.map((i) => {
      const input = el("input", { type: "number", min: 0, max: Math.min(100, i.available_quantity ?? 100), value: 0, "aria-label": `Quantity of ${i.name}` });
      qty.set(i.id, input);
      return el("div", { class: "pick" },
        el("span", { text: i.name }),
        el("span", { class: "muted small", text: `${money(i.price_paise)}${i.available_quantity !== null ? ` · ${i.available_quantity} left` : ""}` }),
        input);
    }));
    const name = el("input", { maxlength: 100, placeholder: "Optional" });
    const phone = el("input", { maxlength: 20, placeholder: "Optional" });
    openDialog({
      title: "Record walk-in sale", okText: "Create order",
      body: [list, field("Customer name", name), field("Contact number", phone),
        el("p", { class: "muted small", text: "Prices and stock are checked on the server. The order is labelled Walk-in / POS; mark it paid once the customer pays." })],
      onOk: async () => {
        const items = [...qty].map(([id, input]) => ({ menu_item_id: id, quantity: Number(input.value) })).filter((l) => l.quantity > 0);
        if (!items.length) throw new Error("Add at least one item.");
        const order = await api("/orders/walk-in", { method: "POST", body: {
          items, customer_name: name.value.trim() || null, customer_phone: phone.value.trim() || null } });
        toast(`${order.order_number} created: ${money(order.total_paise)}.`);
        await refreshAll({ quiet: true });
        openOrder(order.id);
      },
    });
  });

  // ── Menu & inventory ─────────────────────────────────────────────────────
  async function loadMenu({ quiet = false } = {}) {
    const box = $("menu-state");
    if (!quiet) { box.className = "state"; box.textContent = "Loading menu…"; }
    try { state.menu = await api("/menu?include_inactive=true"); renderMenu(); }
    catch (err) { if (!quiet) { box.className = "state error"; box.textContent = err.message; } }
  }
  function renderMenu() {
    const term = $("m-search").value.trim().toLowerCase();
    const showRetired = $("m-retired").checked;
    const items = state.menu.filter((i) => (showRetired || i.is_active) &&
      (!term || i.name.toLowerCase().includes(term) || (i.category || "").toLowerCase().includes(term)));
    const dash = (v) => (v === null || v === undefined ? "–" : v);
    $("menu-body").replaceChildren(...items.map((i) => el("tr", { class: i.is_active ? "" : "retired" },
      el("td", {}, el("strong", { text: i.name }), i.description ? el("div", { class: "item-desc", text: i.description }) : null),
      el("td", { text: i.category || "–" }),
      el("td", { class: "num", text: money(i.price_paise) }),
      el("td", { class: "num" }, el("strong", { text: i.available_quantity === null ? "Not tracked" : i.available_quantity })),
      el("td", { class: "num", text: dash(i.on_hand) }),
      el("td", { class: "num", text: dash(i.reserved) }),
      el("td", { class: "num", text: dash(i.sold) }),
      el("td", {}, stockBadge(i)),
      el("td", {}, el("div", { class: "actions" }, itemActions(i))),
    )));
    const box = $("menu-state");
    box.className = "state";
    box.textContent = items.length ? "" : (state.menu.length ? "No items match." : "No menu items yet. Add the first one.");
  }
  $("m-search").addEventListener("input", renderMenu);
  $("m-retired").addEventListener("change", renderMenu);

  function itemActions(i) {
    const btn = (label, onclick, cls = "ghost") => el("button", { class: `btn small ${cls}`, text: label, onclick });
    if (!i.is_active) {
      return [btn("Restore", () => confirmAction(`Restore ${i.name}?`, "It returns to the menu and the voice agent can offer it again.", "Restore",
        async () => { await api(`/menu/${i.id}/activate`, { method: "POST" }); toast(`${i.name} restored.`); await refreshAll({ quiet: true }); }))];
    }
    return [
      btn("Stock", () => stockDialog(i), "primary"),
      btn("Edit", () => itemDialog(i)),
      btn(i.is_available ? "Switch off" : "Switch on", () => {
        const on = !i.is_available;
        confirmAction(`${on ? "Switch on" : "Switch off"} ${i.name}?`,
          on ? "Customers and the voice agent can order it again (if in stock)." : "It stays on the menu but nobody can order it until you switch it back on.",
          on ? "Switch on" : "Switch off",
          async () => { await api(`/menu/${i.id}`, { method: "PATCH", body: { is_available: on } }); toast(`${i.name} switched ${on ? "on" : "off"}.`); await refreshAll({ quiet: true }); });
      }),
      btn("Retire", () => confirmAction(`Retire ${i.name}?`,
        "It is removed from the menu, the voice agent and new orders. Past orders and the stock history are kept, and you can restore it later.",
        "Retire", async () => { await api(`/menu/${i.id}/deactivate`, { method: "POST" }); toast(`${i.name} retired.`); await refreshAll({ quiet: true }); }, true)),
    ];
  }

  function stockDialog(i) {
    const change = el("input", { type: "number", required: true, step: 1, min: -1000000, max: 1000000, placeholder: "e.g. 20 or -3" });
    const preset = el("select", {}, ["Delivery received", "Stock count correction", "Waste / spoiled", "Damaged", "Other"].map((r) => el("option", { text: r })));
    const note = el("input", { maxlength: 150, placeholder: "Optional details" });
    const now = i.available_quantity === null ? "Stock not tracked yet: a positive number starts tracking."
      : `On hand ${i.on_hand}, reserved for unpaid orders ${i.reserved}, sellable ${i.available_quantity}.`;
    openDialog({
      title: `Adjust stock: ${i.name}`, okText: "Apply",
      body: [el("p", { class: "muted small", text: now }), field("Change (+ add, − remove)", change), field("Reason", preset), field("Details", note)],
      onOk: async () => {
        const n = Number(change.value);
        if (!Number.isInteger(n) || n === 0) throw new Error("Enter a whole number other than 0.");
        const reason = note.value.trim() ? `${preset.value}: ${note.value.trim()}` : preset.value;
        const item = await api(`/menu/${i.id}/stock`, { method: "POST", body: { change: n, reason } });
        toast(`${item.name}: ${n > 0 ? "+" : ""}${n}. Sellable now ${item.available_quantity}.`);
        await refreshAll({ quiet: true });
      },
    });
  }

  function itemDialog(i) {
    const editing = Boolean(i);
    const name = el("input", { required: true, maxlength: 200, value: editing ? i.name : "" });
    const category = el("input", { maxlength: 100, value: editing ? i.category || "" : "" });
    const description = el("textarea", { maxlength: 2000, rows: 3 });
    description.value = editing ? i.description || "" : "";
    const price = el("input", { type: "number", required: true, min: 0.01, step: 0.01, value: editing ? (i.price_paise / 100).toFixed(2) : "" });
    const stock = el("input", { type: "number", min: 0, step: 1, placeholder: "Leave empty to not track stock" });
    openDialog({
      title: editing ? `Edit ${i.name}` : "Add menu item", okText: editing ? "Save changes" : "Add item",
      body: [field("Name", name), field("Category", category), field("Description", description),
        field(`Price (${state.currency})`, price), editing ? null : field("Starting stock", stock)].filter(Boolean),
      onOk: async () => {
        const paise = Math.round(Number(price.value) * 100);
        if (!Number.isInteger(paise) || paise <= 0) throw new Error("Enter a price greater than 0.");
        const body = { name: name.value.trim(), category: category.value.trim() || null,
          description: description.value.trim() || null, price_paise: paise };
        if (editing) {
          await api(`/menu/${i.id}`, { method: "PATCH", body });
          toast(`${body.name} updated.`);
        } else {
          if (stock.value !== "") body.initial_stock = Number(stock.value);
          await api("/menu", { method: "POST", body });
          toast(`${body.name} added to the menu.`);
        }
        await refreshAll({ quiet: true });
      },
    });
  }
  $("item-new").addEventListener("click", () => itemDialog(null));

  // ── Navigation and start-up ──────────────────────────────────────────────
  for (const button of document.querySelectorAll(".nav-item")) {
    button.addEventListener("click", () => {
      state.view = button.dataset.view;
      for (const b of document.querySelectorAll(".nav-item")) b.classList.toggle("active", b === button);
      $("orders-view").hidden = state.view !== "orders";
      $("menu-view").hidden = state.view !== "menu";
      $("drawer").hidden = true;
      state.openOrderId = null;
    });
  }
  setInterval(() => { if (state.view === "orders" && state.orders.length) renderOrders(); }, 30000); // expiry countdowns

  api("/session").then(showApp).catch(showLogin);
})();
