#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import shutil
import tempfile
import threading
import time
import traceback
import uuid
import warnings
from dataclasses import dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PAGE_HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>nvdiffrast Web Viewer</title>
  <style>
    :root {
      --bg: #f6efe2;
      --bg-soft: #fffaf1;
      --panel: rgba(255, 250, 241, 0.88);
      --ink: #1f2933;
      --muted: #5e6b74;
      --accent: #b95c31;
      --accent-strong: #8e3f1c;
      --accent-soft: rgba(185, 92, 49, 0.12);
      --line: rgba(31, 41, 51, 0.12);
      --shadow: 0 24px 70px rgba(59, 36, 26, 0.12);
      --preview-a: #10212e;
      --preview-b: #183548;
      --preview-line: rgba(255, 255, 255, 0.08);
      --ok: #2f7d5a;
      --warn: #a44a2f;
      --radius: 26px;
    }

    * {
      box-sizing: border-box;
    }

    html, body {
      margin: 0;
      min-height: 100%;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(36, 115, 91, 0.16), transparent 34%),
        radial-gradient(circle at right 10%, rgba(185, 92, 49, 0.18), transparent 32%),
        linear-gradient(180deg, #f9f2e7 0%, #f3ebde 100%);
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
    }

    body {
      padding: 32px 20px 40px;
    }

    .shell {
      max-width: 1500px;
      margin: 0 auto;
      display: grid;
      gap: 22px;
      animation: rise-in 420ms ease-out both;
    }

    .hero,
    .panel {
      background: var(--panel);
      border: 1px solid rgba(255, 255, 255, 0.55);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }

    .hero {
      padding: 28px;
      display: grid;
      gap: 8px;
      overflow: hidden;
      position: relative;
    }

    .hero::after {
      content: "";
      position: absolute;
      inset: auto -5% -28% auto;
      width: 280px;
      height: 280px;
      border-radius: 999px;
      background: radial-gradient(circle, rgba(185, 92, 49, 0.2) 0%, rgba(185, 92, 49, 0) 68%);
      pointer-events: none;
    }

    .eyebrow {
      margin: 0;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font-size: 11px;
      color: var(--accent-strong);
      font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    }

    h1 {
      margin: 0;
      font-size: clamp(2rem, 3.4vw, 3.75rem);
      line-height: 0.95;
      max-width: 60ch;
    }

    .hero p {
      margin: 0;
      max-width: 58ch;
      color: var(--muted);
      font-size: 1.02rem;
      line-height: 1.55;
    }

    .layout {
      display: grid;
      gap: 22px;
      grid-template-columns: minmax(320px, 430px) minmax(0, 1fr);
      align-items: start;
    }

    .panel {
      padding: 22px;
      display: grid;
      gap: 18px;
    }

    .controls {
      animation: drift-in 480ms ease-out both;
    }

    .preview {
      animation: drift-in 560ms ease-out both;
    }

    .section {
      display: grid;
      gap: 12px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--line);
    }

    .section:last-child {
      padding-bottom: 0;
      border-bottom: 0;
    }

    .section-head {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
    }

    .section-head h2 {
      margin: 0;
      font-size: 1rem;
      font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      letter-spacing: 0.04em;
    }

    .hint {
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.45;
    }

    label {
      display: grid;
      gap: 8px;
      font-size: 0.95rem;
      color: var(--ink);
    }

    select,
    input[type="file"],
    button,
    .mode-button {
      font: inherit;
    }

    select,
    input[type="file"] {
      width: 100%;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid rgba(31, 41, 51, 0.16);
      background: rgba(255, 255, 255, 0.92);
      color: var(--ink);
    }

    .upload-row,
    .toolbar,
    .stats {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }

    .upload-row > *:first-child {
      flex: 1 1 220px;
    }

    .button,
    .mode-button {
      border: 1px solid transparent;
      border-radius: 999px;
      padding: 10px 15px;
      background: rgba(255, 255, 255, 0.78);
      color: var(--ink);
      cursor: pointer;
      transition: transform 160ms ease, background 160ms ease, border-color 160ms ease, color 160ms ease, box-shadow 160ms ease;
    }

    .button:hover,
    .mode-button:hover {
      transform: translateY(-1px);
      border-color: rgba(185, 92, 49, 0.3);
      box-shadow: 0 10px 24px rgba(104, 52, 29, 0.12);
    }

    .button.primary,
    .mode-button.active {
      background: linear-gradient(135deg, var(--accent), var(--accent-strong));
      color: #fff9f2;
      box-shadow: 0 12px 26px rgba(142, 63, 28, 0.28);
    }

    .button.secondary {
      background: rgba(16, 33, 46, 0.06);
      border-color: rgba(16, 33, 46, 0.08);
    }

    .button:disabled {
      opacity: 0.6;
      cursor: wait;
      transform: none;
    }

    .mode-grid {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    }

    .slider-block {
      display: grid;
      gap: 8px;
    }

    .slider-block.disabled {
      opacity: 0.58;
    }

    .slider-line {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: center;
    }

    .slider-line span {
      font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 0.88rem;
      color: var(--muted);
      min-width: 76px;
      text-align: right;
    }

    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }

    .color-row {
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 12px;
      align-items: center;
    }

    input[type="color"] {
      width: 56px;
      height: 40px;
      padding: 4px;
      border: 1px solid rgba(31, 41, 51, 0.16);
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.92);
      cursor: pointer;
    }

    .color-value {
      font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      color: var(--muted);
      font-size: 0.88rem;
    }

    .preview-shell {
      position: relative;
      min-height: 620px;
      border-radius: calc(var(--radius) - 6px);
      overflow: hidden;
      background:
        linear-gradient(90deg, transparent 48%, var(--preview-line) 50%, transparent 52%) 0 0 / 42px 42px,
        linear-gradient(transparent 48%, var(--preview-line) 50%, transparent 52%) 0 0 / 42px 42px,
        radial-gradient(circle at top, rgba(255, 255, 255, 0.08), transparent 36%),
        linear-gradient(160deg, var(--preview-a), var(--preview-b));
      border: 1px solid rgba(255, 255, 255, 0.12);
      user-select: none;
      touch-action: none;
      cursor: grab;
    }

    .preview-shell.dragging {
      cursor: grabbing;
    }

    .preview-shell img {
      display: block;
      width: 100%;
      height: 100%;
      min-height: 620px;
      object-fit: contain;
    }

    .overlay {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      padding: 18px;
      text-align: center;
      color: rgba(255, 249, 242, 0.92);
      background: linear-gradient(180deg, rgba(11, 20, 30, 0.18), rgba(11, 20, 30, 0.45));
      opacity: 0;
      pointer-events: none;
      transition: opacity 160ms ease;
    }

    .overlay.visible {
      opacity: 1;
    }

    .overlay-card {
      max-width: 420px;
      padding: 18px 20px;
      border-radius: 20px;
      background: rgba(10, 18, 27, 0.7);
      border: 1px solid rgba(255, 255, 255, 0.12);
      backdrop-filter: blur(10px);
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 9px 12px;
      border-radius: 999px;
      background: rgba(16, 33, 46, 0.08);
      color: var(--muted);
      font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 0.84rem;
    }

    .status-pill.ok {
      background: rgba(47, 125, 90, 0.12);
      color: var(--ok);
    }

    .status-pill.warn {
      background: rgba(164, 74, 47, 0.12);
      color: var(--warn);
    }

    .stats {
      justify-content: space-between;
    }

    .stats strong {
      font-weight: 600;
      color: var(--ink);
    }

    .mono {
      font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    }

    @keyframes rise-in {
      from {
        opacity: 0;
        transform: translateY(16px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    @keyframes drift-in {
      from {
        opacity: 0;
        transform: translateY(20px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    @media (max-width: 1120px) {
      .layout {
        grid-template-columns: 1fr;
      }

      .preview-shell,
      .preview-shell img {
        min-height: 460px;
      }
    }

    @media (max-width: 640px) {
      body {
        padding: 20px 14px 28px;
      }

      .hero,
      .panel {
        padding: 18px;
      }

      .preview-shell,
      .preview-shell img {
        min-height: 360px;
      }

      .slider-line {
        grid-template-columns: 1fr;
      }

      .slider-line span {
        text-align: left;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <p class="eyebrow">CUDA / nvdiffrast / Local Viewer</p>
      <h1>Orbit mesh review in the browser.</h1>
    </section>

    <section class="layout">
      <aside class="panel controls">
        <section class="section">
          <div class="section-head">
            <h2>Mesh Source</h2>
            <span class="hint" id="meshCountLabel"></span>
          </div>

          <label>
            Example mesh or uploaded mesh
            <select id="meshSelect"></select>
          </label>

          <div class="upload-row">
            <input id="uploadInput" type="file" accept=".glb,.gltf">
            <button class="button primary" id="uploadButton" type="button">Upload Mesh</button>
          </div>
          <div class="hint">upload `glb` file</div>
        </section>

        <section class="section">
          <div class="section-head">
            <h2>Render Mode</h2>
            <span class="hint mono" id="modeLabel"></span>
          </div>
          <div class="mode-grid" id="modeGrid"></div>
        </section>

        <section class="section">
          <div class="section-head">
            <h2>Camera</h2>
            <span class="hint">drag to orbit / wheel to dolly</span>
          </div>

          <label class="slider-block">
            Elev
            <div class="slider-line">
              <input id="elevInput" type="range" min="-85" max="85" step="1" value="15">
              <span id="elevValue">15.0°</span>
            </div>
          </label>

          <label class="slider-block">
            Azim
            <div class="slider-line">
              <input id="azimInput" type="range" min="-180" max="180" step="1" value="35">
              <span id="azimValue">35.0°</span>
            </div>
          </label>

          <label class="slider-block">
            Camera Distance
            <div class="slider-line">
              <input id="distanceInput" type="range" min="0.70" max="3.00" step="0.05" value="1.15">
              <span id="distanceValue">1.15x</span>
            </div>
          </label>

          <label class="slider-block">
            Camera Type
            <select id="cameraInput">
              <option value="perspective">perspective</option>
              <option value="orthographic">orthographic</option>
            </select>
          </label>

          <label class="slider-block">
            FOV
            <div class="slider-line">
              <input id="fovInput" type="range" min="15" max="90" step="1" value="45">
              <span id="fovValue">45.0°</span>
            </div>
          </label>

          <label class="slider-block">
            JPEG Background
            <div class="color-row">
              <input id="backgroundColorInput" type="color" value="#8c8c8c">
              <span class="color-value" id="backgroundColorValue">#8c8c8c</span>
            </div>
          </label>

          <div class="toolbar">
            <button class="button secondary" id="resetButton" type="button">Reset View</button>
            <button class="button secondary" id="renderButton" type="button">Render Now</button>
          </div>
        </section>
      </aside>

      <section class="panel preview">
        <div class="stats">
          <div class="status-pill" id="statusPill">idle</div>
          <div class="toolbar">
            <span class="status-pill ok" id="meshLabel">mesh: -</span>
            <span class="status-pill" id="renderInfo">render: -</span>
          </div>
        </div>

        <div class="preview-shell" id="previewShell">
          <img id="previewImage" alt="Rendered mesh preview">
          <div class="overlay visible" id="overlay">
            <div class="overlay-card">
              <strong>Viewer is ready.</strong>
              <p id="overlayText">...initializing...</p>
            </div>
          </div>
        </div>
      </section>
    </section>
  </main>

  <script>
    window.__VIEWER_STATE__ = __INITIAL_VIEWER_STATE__;
    const defaults = { elev: 15, azim: 35, distance: 1.15, camera: "perspective", fov: 45, mode: "beauty", background: "#bcbcbc" };
    const apiBase = new URL(".", window.location.href);
    const state = {
      meshId: null,
      mode: defaults.mode,
      pendingMode: null,
      rendering: false,
      previewTimer: null,
      finalTimer: null,
      drag: null,
      imageUrl: null,
      modes: [],
    };

    const meshSelect = document.getElementById("meshSelect");
    const uploadInput = document.getElementById("uploadInput");
    const uploadButton = document.getElementById("uploadButton");
    const modeGrid = document.getElementById("modeGrid");
    const modeLabel = document.getElementById("modeLabel");
    const meshCountLabel = document.getElementById("meshCountLabel");
    const elevInput = document.getElementById("elevInput");
    const azimInput = document.getElementById("azimInput");
    const distanceInput = document.getElementById("distanceInput");
    const cameraInput = document.getElementById("cameraInput");
    const fovInput = document.getElementById("fovInput");
    const backgroundColorInput = document.getElementById("backgroundColorInput");
    const elevValue = document.getElementById("elevValue");
    const azimValue = document.getElementById("azimValue");
    const distanceValue = document.getElementById("distanceValue");
    const fovValue = document.getElementById("fovValue");
    const backgroundColorValue = document.getElementById("backgroundColorValue");
    const resetButton = document.getElementById("resetButton");
    const renderButton = document.getElementById("renderButton");
    const previewShell = document.getElementById("previewShell");
    const previewImage = document.getElementById("previewImage");
    const overlay = document.getElementById("overlay");
    const overlayText = document.getElementById("overlayText");
    const statusPill = document.getElementById("statusPill");
    const meshLabel = document.getElementById("meshLabel");
    const renderInfo = document.getElementById("renderInfo");

    function setStatus(message, tone = "") {
      statusPill.textContent = message;
      statusPill.className = "status-pill" + (tone ? " " + tone : "");
    }

    function showOverlay(message, keepVisible = false) {
      overlayText.textContent = message;
      overlay.classList.toggle("visible", keepVisible);
    }

    function hideOverlay() {
      overlay.classList.remove("visible");
    }

    function wrapAngle(value) {
      let result = Number(value);
      while (result > 180) result -= 360;
      while (result < -180) result += 360;
      return result;
    }

    function syncLabels() {
      elevValue.textContent = `${Number(elevInput.value).toFixed(1)}°`;
      azimValue.textContent = `${Number(azimInput.value).toFixed(1)}°`;
      distanceValue.textContent = `${Number(distanceInput.value).toFixed(2)}x`;
      const orthographic = cameraInput.value === "orthographic";
      fovInput.disabled = orthographic;
      fovInput.closest(".slider-block")?.classList.toggle("disabled", orthographic);
      fovValue.textContent = orthographic ? "ignored" : `${Number(fovInput.value).toFixed(1)}°`;
      backgroundColorValue.textContent = backgroundColorInput.value.toLowerCase();
      modeLabel.textContent = state.mode;
    }

    function applyBackgroundColor(value) {
      const normalized = String(value || defaults.background).toLowerCase();
      backgroundColorInput.value = normalized;
      previewShell.style.background = normalized;
      syncLabels();
    }

    function currentPayload() {
      return {
        mesh_id: state.meshId,
        render_mode: state.mode,
        elev: Number(elevInput.value),
        azim: Number(azimInput.value),
        distance_scale: Number(distanceInput.value),
        camera: cameraInput.value,
        fov: Number(fovInput.value),
        background_hex: backgroundColorInput.value,
      };
    }

    function setMode(mode) {
      state.mode = mode;
      for (const button of modeGrid.querySelectorAll(".mode-button")) {
        button.classList.toggle("active", button.dataset.mode === mode);
      }
      syncLabels();
    }

    function buildModeButtons(modes) {
      modeGrid.innerHTML = "";
      for (const mode of modes) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "mode-button";
        button.dataset.mode = mode;
        button.textContent = mode;
        button.addEventListener("click", () => {
          setMode(mode);
          queueFinalRender(0);
        });
        modeGrid.appendChild(button);
      }
      setMode(state.mode);
    }

    function setMeshes(meshes, preferredId = null) {
      const current = preferredId ?? state.meshId;
      meshSelect.innerHTML = "";
      for (const mesh of meshes) {
        const option = document.createElement("option");
        option.value = mesh.id;
        option.textContent = `${mesh.name} (${mesh.kind})`;
        meshSelect.appendChild(option);
      }
      if (current && meshes.some((mesh) => mesh.id === current)) {
        meshSelect.value = current;
      }
      state.meshId = meshSelect.value || null;
      meshCountLabel.textContent = `${meshes.length} mesh option(s)`;
      const selectedText = meshSelect.options[meshSelect.selectedIndex]?.textContent ?? "-";
      meshLabel.textContent = `mesh: ${selectedText}`;
    }

    async function fetchState() {
      const response = await fetch(new URL("api/state", apiBase), { cache: "no-store" });
      if (!response.ok) {
        throw new Error("Failed to load viewer state.");
      }
      return response.json();
    }

    function setPendingMode(mode) {
      if (state.pendingMode === "final") {
        return;
      }
      state.pendingMode = mode;
    }

    function queueFinalRender(delay = 0, { cancelPreview = true } = {}) {
      if (cancelPreview) {
        window.clearTimeout(state.previewTimer);
      }
      queueRender("final", delay);
    }

    async function renderNow(mode = "final") {
      if (!state.meshId) {
        setStatus("mesh required", "warn");
        showOverlay("Select example mesh or upload mesh.", true);
        return;
      }
      if (state.rendering) {
        setPendingMode(mode);
        return;
      }
      if (mode === "final") {
        window.clearTimeout(state.previewTimer);
      }

      state.rendering = true;
      state.pendingMode = null;
      uploadButton.disabled = true;
      renderButton.disabled = true;
      if (!state.imageUrl) {
        setStatus(mode === "preview" ? "previewing" : "rendering", "");
        showOverlay(mode === "preview" ? "Rendering low-res preview..." : "Rendering....Please wait", true);
      } else {
        hideOverlay();
        setStatus(mode === "preview" ? "previewing" : "refining", "");
      }

      const startedAt = performance.now();
      try {
        const response = await fetch(new URL("api/render", apiBase), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...currentPayload(), preview: mode === "preview" }),
        });
        if (!response.ok) {
          const error = await response.json().catch(() => ({ error: "Render failed." }));
          throw new Error(error.error || "Render failed.");
        }

        if (mode === "preview" && state.pendingMode === "final") {
          return;
        }

        const blob = await response.blob();
        if (state.imageUrl) {
          URL.revokeObjectURL(state.imageUrl);
        }
        state.imageUrl = URL.createObjectURL(blob);
        previewImage.src = state.imageUrl;
        hideOverlay();
        setStatus(mode === "preview" ? "preview ready" : "ready", "ok");
        const ms = response.headers.get("X-Render-Time-Ms");
        const resolution = response.headers.get("X-Render-Resolution");
        const elapsed = ms ? `${Number(ms).toFixed(1)} ms` : `${(performance.now() - startedAt).toFixed(1)} ms`;
        renderInfo.textContent = `render: ${mode}${resolution ? ` ${resolution}px` : ""} ${elapsed}`;
        meshLabel.textContent = `mesh: ${meshSelect.options[meshSelect.selectedIndex]?.textContent ?? "-"}`;
      } catch (error) {
        setStatus("render failed", "warn");
        showOverlay(error.message || "Render failed.", true);
        renderInfo.textContent = `render: ${error.message || "error"}`;
      } finally {
        state.rendering = false;
        uploadButton.disabled = false;
        renderButton.disabled = false;
        if (state.pendingMode) {
          const nextMode = state.pendingMode;
          state.pendingMode = null;
          renderNow(nextMode);
        }
      }
    }

    function queueRender(mode = "final", delay = 140) {
      const timerKey = mode === "preview" ? "previewTimer" : "finalTimer";
      window.clearTimeout(state[timerKey]);
      state[timerKey] = window.setTimeout(() => {
        renderNow(mode);
      }, delay);
    }

    function queueInteractiveRender() {
      queueRender("preview", 45);
      queueFinalRender(240, { cancelPreview: false });
    }

    async function uploadMesh() {
      const file = uploadInput.files?.[0];
      if (!file) {
        setStatus("select a file", "warn");
        return;
      }

      const formData = new FormData();
      formData.append("file", file);
      uploadButton.disabled = true;
      setStatus("uploading", "");
      showOverlay("uploading...", true);

      try {
        const response = await fetch(new URL("api/upload", apiBase), {
          method: "POST",
          body: formData,
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Upload failed.");
        }
        const viewerState = await fetchState();
        setMeshes(viewerState.meshes, payload.mesh.id);
        uploadInput.value = "";
        setStatus("upload ready", "ok");
        queueFinalRender(0);
      } catch (error) {
        setStatus("upload failed", "warn");
        showOverlay(error.message || "Upload failed.", true);
      } finally {
        uploadButton.disabled = false;
      }
    }

    function updateSliderValue(input, value) {
      input.value = String(value);
      syncLabels();
    }

    function resetView() {
      updateSliderValue(elevInput, defaults.elev);
      updateSliderValue(azimInput, defaults.azim);
      updateSliderValue(distanceInput, defaults.distance);
      cameraInput.value = defaults.camera;
      updateSliderValue(fovInput, defaults.fov);
      applyBackgroundColor(defaults.background);
      queueFinalRender(0);
    }

    function onControlInput() {
      syncLabels();
      queueInteractiveRender();
    }

    meshSelect.addEventListener("change", () => {
      state.meshId = meshSelect.value;
      meshLabel.textContent = `mesh: ${meshSelect.options[meshSelect.selectedIndex]?.textContent ?? "-"}`;
      queueFinalRender(0);
    });
    uploadButton.addEventListener("click", uploadMesh);
    resetButton.addEventListener("click", resetView);
    renderButton.addEventListener("click", () => queueFinalRender(0));
    elevInput.addEventListener("input", onControlInput);
    azimInput.addEventListener("input", onControlInput);
    distanceInput.addEventListener("input", onControlInput);
    cameraInput.addEventListener("change", onControlInput);
    fovInput.addEventListener("input", onControlInput);
    backgroundColorInput.addEventListener("input", () => {
      applyBackgroundColor(backgroundColorInput.value);
      queueFinalRender(0);
    });

    previewShell.addEventListener("pointerdown", (event) => {
      previewShell.classList.add("dragging");
      previewShell.setPointerCapture(event.pointerId);
      state.drag = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        elev: Number(elevInput.value),
        azim: Number(azimInput.value),
      };
    });

    previewShell.addEventListener("pointermove", (event) => {
      if (!state.drag || state.drag.pointerId !== event.pointerId) {
        return;
      }
      const nextAzim = wrapAngle(state.drag.azim - (event.clientX - state.drag.startX) * 0.65);
      const nextElev = Math.max(-85, Math.min(85, state.drag.elev + (event.clientY - state.drag.startY) * 0.55));
      updateSliderValue(azimInput, nextAzim);
      updateSliderValue(elevInput, nextElev);
      queueInteractiveRender();
    });

    function endDrag(pointerId) {
      if (state.drag && state.drag.pointerId === pointerId) {
        previewShell.classList.remove("dragging");
        state.drag = null;
        queueFinalRender(0);
      }
    }

    previewShell.addEventListener("pointerup", (event) => endDrag(event.pointerId));
    previewShell.addEventListener("pointercancel", (event) => endDrag(event.pointerId));
    previewShell.addEventListener("wheel", (event) => {
      event.preventDefault();
      const current = Number(distanceInput.value);
      const next = Math.max(0.7, Math.min(3.0, current + Math.sign(event.deltaY) * 0.08));
      updateSliderValue(distanceInput, next.toFixed(2));
      queueInteractiveRender();
    }, { passive: false });

    async function boot() {
      syncLabels();
      try {
        const payload = window.__VIEWER_STATE__ || await fetchState();
        state.modes = payload.render_modes;
        defaults.elev = payload.defaults.elev;
        defaults.azim = payload.defaults.azim;
        defaults.distance = payload.defaults.distance_scale;
        defaults.camera = payload.defaults.camera || "perspective";
        defaults.fov = payload.defaults.fov;
        defaults.mode = payload.defaults.render_mode;
        defaults.background = payload.defaults.background_hex || "#bcbcbc";
        updateSliderValue(elevInput, defaults.elev);
        updateSliderValue(azimInput, defaults.azim);
        updateSliderValue(distanceInput, defaults.distance);
        cameraInput.value = defaults.camera;
        updateSliderValue(fovInput, defaults.fov);
        applyBackgroundColor(defaults.background);
        state.mode = defaults.mode;
        buildModeButtons(payload.render_modes);
        setMeshes(payload.meshes, payload.defaults.mesh_id);
        setStatus("ready", "ok");
        showOverlay("Init Rendering.", true);
        queueFinalRender(0);
      } catch (error) {
        setStatus("init failed", "warn");
        showOverlay(error.message || "Failed to initialize viewer.", true);
      }
    }

    boot();
  </script>
</body>
</html>
"""

MAX_UPLOAD_BYTES = 512 * 1024 * 1024
SUPPORTED_UPLOAD_SUFFIXES = {".glb", ".gltf"}
DEFAULT_BACKGROUND_HEX = "#bcbcbc"
VIEWER_JPG_QUALITY = 90


def _parse_background_hex(value: str | None) -> tuple[str, tuple[int, int, int]]:
    text = (value or DEFAULT_BACKGROUND_HEX).strip().lower()
    if len(text) != 7 or not text.startswith("#"):
        raise ValueError("background_hex must be a #RRGGBB color.")
    try:
        rgb = tuple(int(text[index:index + 2], 16) for index in (1, 3, 5))
    except ValueError as exc:
        raise ValueError("background_hex must be a #RRGGBB color.") from exc
    return text, rgb


@dataclass(frozen=True)
class MeshEntry:
    id: str
    name: str
    path: Path
    kind: str

    def to_public(self) -> dict[str, str]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
        }


@dataclass
class RendererState:
    resolution: int
    renderer: Any
    active_mesh_id: str | None = None
    active_assets: Any = None


class ViewerBackend:
    def __init__(self, args: argparse.Namespace):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment issue
            raise RuntimeError("PyTorch is required to run the web viewer.") from exc

        try:
            from nvdiffrast_mesh_renderer.config import CAMERA_CHOICES, RENDER_MODE_CHOICES, config_from_args
            from nvdiffrast_mesh_renderer.lifecycle import is_cuda_failure
            from nvdiffrast_mesh_renderer.renderer import SceneRenderer
        except ImportError as exc:  # pragma: no cover - environment issue
            raise RuntimeError(
                "Renderer imports failed. Install this package and its nvdiffrast dependency first."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this viewer. `torch.cuda.is_available()` returned false.")

        self._torch = torch
        self._config_from_args = config_from_args
        self._SceneRenderer = SceneRenderer
        self._is_cuda_failure = is_cuda_failure
        self.camera_choices = tuple(CAMERA_CHOICES)
        self.render_modes = tuple(RENDER_MODE_CHOICES)
        self.repo_root = Path(__file__).resolve().parent
        self.example_dir = (self.repo_root / args.example_dir).resolve()
        self.upload_dir = Path(args.upload_dir).resolve()
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.resolution = int(args.resolution)
        self.preview_resolution = max(64, min(int(args.preview_resolution), self.resolution))
        self.env_map = str(Path(args.env_map).expanduser().resolve()) if args.env_map else ""
        if self.env_map and not Path(self.env_map).is_file():
            raise RuntimeError(f"Environment map not found: {self.env_map}")
        self._lock = threading.RLock()
        self._meshes: dict[str, MeshEntry] = {}
        self._example_ids: list[str] = []
        self._renderer_states: dict[int, RendererState] = {}

        self.defaults = {
            "elev": 15.0,
            "azim": 35.0,
            "distance_scale": 1.15,
            "camera": "perspective",
            "fov": 45.0,
            "render_mode": "beauty",
            "background_hex": DEFAULT_BACKGROUND_HEX,
        }
        self._load_example_meshes()
        self.default_mesh_id = self._pick_default_mesh_id()
        self._get_renderer_state(self.resolution, clear_cuda=False)
        if self.preview_resolution != self.resolution:
            self._get_renderer_state(self.preview_resolution, clear_cuda=False)

    def _load_example_meshes(self) -> None:
        if not self.example_dir.is_dir():
            raise RuntimeError(f"Example mesh directory not found: {self.example_dir}")
        mesh_paths = sorted(
            path for path in self.example_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_UPLOAD_SUFFIXES
        )
        if len(mesh_paths) == 0:
            raise RuntimeError(f"No example meshes found under {self.example_dir}")
        for path in mesh_paths:
            mesh_id = f"example:{path.name}"
            self._meshes[mesh_id] = MeshEntry(
                id=mesh_id,
                name=path.name,
                path=path.resolve(),
                kind="example",
            )
            self._example_ids.append(mesh_id)

    def _pick_default_mesh_id(self) -> str:
        preferred_name = "varco3d_pbr_model.glb"
        for mesh_id in self._example_ids:
            if self._meshes[mesh_id].path.name == preferred_name:
                return mesh_id
        return self._example_ids[0]

    def _build_config(self, *, input_path: Path, resolution: int) -> Any:
        args = argparse.Namespace(
            input=str(input_path),
            output=str(self.repo_root / "outputs" / "web_viewer.png"),
            resolution=int(resolution),
            elev=self.defaults["elev"],
            azim=self.defaults["azim"],
            elev_start=None,
            elev_end=None,
            elev_step=None,
            azim_start=None,
            azim_end=None,
            azim_step=None,
            camera=self.defaults["camera"],
            fov=self.defaults["fov"],
            distance=None,
            distance_scale=self.defaults["distance_scale"],
            env_map=self.env_map,
            env_usage="light",
            env_light_intensity=0.35,
            env_background_intensity=1.0,
            env_diffuse_samples=16,
            background="transparent",
            light_intensity=1.15,
            exposure=1.0,
            tonemap="aces",
            cull_mode="auto",
            render_mode=self.defaults["render_mode"],
            wireframe_opacity=0.7,
            wireframe_thickness_px=0.45,
            double_sided_depth_peels=4,
            normalize_depth=True,
            render_all=False,
            render_all_batch_size=4,
            canonical_six_views=False,
            multi_view_chunk_size=4,
            geometry_preprocess_device="auto",
            geometry_cuda_threshold_faces=100000,
            geometry_cuda_threshold_vertices=100000,
            texture_map_max_size=0,
            benchmark_runs=None,
            benchmark_warmup_runs=None,
            no_antialias=False,
            display=False,
            print_progress=False,
        )
        return self._config_from_args(args)

    def _mesh_path(self, mesh_id: str) -> Path:
        entry = self._meshes.get(mesh_id)
        if entry is None:
            raise ValueError(f"Unknown mesh id: {mesh_id}")
        return entry.path

    def _discard_active_assets(self, renderer_state: RendererState, *, release_cuda: bool) -> None:
        renderer_state.active_assets = None
        renderer_state.active_mesh_id = None
        gc.collect()
        renderer_state.renderer.clear_texture_cache()
        if release_cuda and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _create_renderer_state(self, resolution: int, *, clear_cuda: bool) -> RendererState:
        if clear_cuda:
            gc.collect()
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        config = self._build_config(input_path=self._mesh_path(self.default_mesh_id), resolution=resolution)
        return RendererState(
            resolution=resolution,
            renderer=self._SceneRenderer(config, device=self.device),
        )

    def _get_renderer_state(self, resolution: int, *, clear_cuda: bool = True) -> RendererState:
        renderer_state = self._renderer_states.get(resolution)
        if renderer_state is None:
            renderer_state = self._create_renderer_state(resolution, clear_cuda=clear_cuda)
            self._renderer_states[resolution] = renderer_state
        return renderer_state

    def _recreate_renderer(self, resolution: int, *, clear_cuda: bool = True) -> RendererState:
        current = self._renderer_states.get(resolution)
        if current is not None:
            self._discard_active_assets(current, release_cuda=False)
        renderer_state = self._create_renderer_state(resolution, clear_cuda=clear_cuda)
        self._renderer_states[resolution] = renderer_state
        return renderer_state

    def _get_assets(self, renderer_state: RendererState, mesh_id: str):
        if renderer_state.active_mesh_id == mesh_id and renderer_state.active_assets is not None:
            return renderer_state.active_assets
        mesh_path = self._mesh_path(mesh_id)
        self._discard_active_assets(renderer_state, release_cuda=False)
        renderer_state.active_assets = renderer_state.renderer.prepare_assets(mesh_path)
        renderer_state.active_mesh_id = mesh_id
        return renderer_state.active_assets

    def state_payload(self) -> dict[str, Any]:
        return {
            "meshes": [self._meshes[mesh_id].to_public() for mesh_id in self._example_ids] + [
                entry.to_public() for mesh_id, entry in self._meshes.items() if mesh_id not in set(self._example_ids)
            ],
            "render_modes": list(self.render_modes),
            "defaults": {
                "mesh_id": self.default_mesh_id,
                **self.defaults,
            },
            "rendering": {
                "resolution": self.resolution,
                "preview_resolution": self.preview_resolution,
            },
        }

    def upload_mesh(self, filename: str, fileobj) -> MeshEntry:
        original_name = Path(filename or "").name
        suffix = Path(original_name).suffix.lower()
        if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
            raise ValueError("Only .glb and .gltf uploads are supported.")
        mesh_id = f"upload:{uuid.uuid4().hex}"
        target_name = f"{uuid.uuid4().hex}{suffix}"
        target_path = self.upload_dir / target_name
        with target_path.open("wb") as handle:
            shutil.copyfileobj(fileobj, handle)
        entry = MeshEntry(
            id=mesh_id,
            name=original_name or target_name,
            path=target_path.resolve(),
            kind="upload",
        )
        self._meshes[mesh_id] = entry
        return entry

    def render_image(
        self,
        *,
        mesh_id: str,
        render_mode: str,
        elev: float,
        azim: float,
        distance_scale: float,
        camera: str,
        fov: float,
        background_hex: str = DEFAULT_BACKGROUND_HEX,
        preview: bool = False,
    ) -> tuple[bytes, str, float, int]:
        if render_mode not in self.render_modes:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        if camera not in self.camera_choices:
            raise ValueError(f"Unsupported camera type: {camera}")
        if not (0.5 <= distance_scale <= 4.0):
            raise ValueError("distance_scale must stay between 0.5 and 4.0.")
        if not (10.0 <= fov <= 100.0):
            raise ValueError("fov must stay between 10 and 100 degrees.")
        _, background_rgb = _parse_background_hex(background_hex)
        resolution = self.preview_resolution if preview else self.resolution
        started = time.perf_counter()
        with self._lock:
            renderer_state = self._get_renderer_state(resolution)
            effective_config = replace(
                renderer_state.renderer.config,
                input=str(self._mesh_path(mesh_id)),
                elev=float(elev),
                azim=float(azim),
                camera=str(camera),
                fov=float(fov),
                distance=None,
                distance_scale=float(distance_scale),
                render_mode=render_mode,
            )
            try:
                assets = self._get_assets(renderer_state, mesh_id)
                prepared = renderer_state.renderer.prepare_view(assets, effective_config)
                image = renderer_state.renderer.render_prepared(prepared, render_mode=render_mode)
            except Exception as exc:
                if not self._is_cuda_failure(exc):
                    raise
                renderer_state = self._recreate_renderer(resolution, clear_cuda=True)
                effective_config = replace(
                    renderer_state.renderer.config,
                    input=str(self._mesh_path(mesh_id)),
                    elev=float(elev),
                    azim=float(azim),
                    camera=str(camera),
                    fov=float(fov),
                    distance=None,
                    distance_scale=float(distance_scale),
                    render_mode=render_mode,
                )
                assets = self._get_assets(renderer_state, mesh_id)
                prepared = renderer_state.renderer.prepare_view(assets, effective_config)
                image = renderer_state.renderer.render_prepared(prepared, render_mode=render_mode)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        image_bytes, content_type = self._encode_response_image(
            image,
            background_rgb=background_rgb,
        )
        return image_bytes, content_type, elapsed_ms, resolution

    def _encode_response_image(
        self,
        image,
        *,
        background_rgb: tuple[int, int, int],
    ) -> tuple[bytes, str]:
        from nvdiffrast_mesh_renderer.image_io import encode_jpg_bytes

        return encode_jpg_bytes(image, jpg_quality=VIEWER_JPG_QUALITY, background_rgb=background_rgb), "image/jpeg"

    def close(self) -> None:
        for renderer_state in self._renderer_states.values():
            self._discard_active_assets(renderer_state, release_cuda=False)


class ViewerHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, RequestHandlerClass, backend: ViewerBackend):
        super().__init__(server_address, RequestHandlerClass)
        self.backend = backend


class ViewerRequestHandler(BaseHTTPRequestHandler):
    server: ViewerHTTPServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_html(self._render_page())
            return
        if parsed.path == "/api/state":
            self._send_json(self.server.backend.state_payload())
            return
        if parsed.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Not found.")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/upload":
            self._handle_upload()
            return
        if parsed.path == "/api/render":
            self._handle_render()
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Not found.")

    def _handle_upload(self) -> None:
        length = self.headers.get("Content-Length")
        if length is not None and int(length) > MAX_UPLOAD_BYTES:
            self._send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Upload is too large.")
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import cgi
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        if "file" not in form:
            self._send_error(HTTPStatus.BAD_REQUEST, "Multipart form must include a `file` field.")
            return
        field = form["file"]
        if isinstance(field, list):
            field = field[0]
        filename = getattr(field, "filename", "") or ""
        if not filename:
            self._send_error(HTTPStatus.BAD_REQUEST, "Uploaded file is missing a filename.")
            return
        try:
            entry = self.server.backend.upload_mesh(filename, field.file)
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except Exception as exc:
            traceback.print_exc()
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return
        self._send_json({"mesh": entry.to_public()})

    def _handle_render(self) -> None:
        try:
            payload = self._read_json()
            image_bytes, content_type, elapsed_ms, resolution = self.server.backend.render_image(
                mesh_id=str(payload["mesh_id"]),
                render_mode=str(payload["render_mode"]),
                elev=float(payload["elev"]),
                azim=float(payload["azim"]),
                distance_scale=float(payload["distance_scale"]),
                camera=str(payload.get("camera", "perspective")),
                fov=float(payload["fov"]),
                background_hex=str(payload.get("background_hex", DEFAULT_BACKGROUND_HEX)),
                preview=bool(payload.get("preview", False)),
            )
        except KeyError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, f"Missing required field: {exc.args[0]}")
            return
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except Exception as exc:
            traceback.print_exc()
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(image_bytes)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Render-Time-Ms", f"{elapsed_ms:.3f}")
        self.send_header("X-Render-Resolution", str(resolution))
        self.end_headers()
        self.wfile.write(image_bytes)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _render_page(self) -> str:
        initial_state = json.dumps(
            self.server.backend.state_payload(),
            ensure_ascii=False,
        ).replace("<", "\\u003c")
        return PAGE_HTML.replace("__INITIAL_VIEWER_STATE__", initial_state, 1)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        del format, args


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local browser-based viewer backed by nvdiffrast_mesh_renderer.")
    parser.add_argument("--host", default="127.0.0.1", help="Host/interface to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument("--resolution", type=int, default=1024, help="Square render resolution")
    parser.add_argument("--preview-resolution", type=int, default=128, help="Low-res preview size used while camera controls are moving")
    parser.add_argument("--example-dir", default="example_meshes", help="Directory containing bundled example meshes")
    parser.add_argument("--env-map", default="", help="Optional HDR/EXR environment map path. Disabled by default for compatibility.")
    parser.add_argument(
        "--upload-dir",
        default=str(Path(tempfile.gettempdir()) / "nvdiffrast_web_viewer_uploads"),
        help="Directory to store uploaded meshes",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    try:
        backend = ViewerBackend(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    server = ViewerHTTPServer((args.host, args.port), ViewerRequestHandler, backend)
    url = f"http://{args.host}:{args.port}"
    print(f"nvdiffrast web viewer listening on {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down viewer.")
    finally:
        backend.close()
        server.server_close()


if __name__ == "__main__":
    main()
