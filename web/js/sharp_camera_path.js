import { app } from "../../../scripts/app.js";

function chainCallback(object, property, callback) {
    if (!object) {
        return;
    }
    if (property in object && object[property]) {
        const original = object[property];
        object[property] = function () {
            const result = original.apply(this, arguments);
            return callback.apply(this, arguments) ?? result;
        };
    } else {
        object[property] = callback;
    }
}

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function defaultPath() {
    return [
        { t: 0.0, x: -0.45, y: 0.00, z: 0.00 },
        { t: 0.35, x: 0.18, y: -0.06, z: 0.20 },
        { t: 0.70, x: 0.45, y: 0.04, z: 0.10 },
        { t: 1.0, x: -0.45, y: 0.00, z: 0.00 },
    ];
}

function parsePath(value) {
    try {
        const data = JSON.parse(value || "[]");
        const source = Array.isArray(data) ? data : data.keyframes;
        if (!Array.isArray(source)) {
            return defaultPath();
        }
        const points = source.map((p) => ({
            t: Number(p.t),
            x: Number(p.x),
            y: Number(p.y),
            z: Number(p.z),
        })).filter((p) => Number.isFinite(p.t) && Number.isFinite(p.x)
            && Number.isFinite(p.y) && Number.isFinite(p.z));
        return points.length >= 2 ? points.sort((a, b) => a.t - b.t) : defaultPath();
    } catch {
        return defaultPath();
    }
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function writePath(node, points) {
    const widget = findWidget(node, "path_json");
    if (!widget) {
        return;
    }
    const normalized = points
        .map((p) => ({
            t: Number(clamp(p.t, 0, 1).toFixed(4)),
            x: Number(clamp(p.x, -2, 2).toFixed(4)),
            y: Number(clamp(p.y, -2, 2).toFixed(4)),
            z: Number(clamp(p.z, -2, 2).toFixed(4)),
        }))
        .sort((a, b) => a.t - b.t);
    if (normalized.length > 0) {
        normalized[0].t = 0;
        normalized[normalized.length - 1].t = 1;
    }
    widget.value = JSON.stringify(normalized, null, 2);
    widget.callback?.(widget.value);
    node.setDirtyCanvas?.(true, true);
}

function widgetValue(node, name, fallback) {
    const widget = findWidget(node, name);
    return widget ? widget.value : fallback;
}

function setWidgetValue(node, name, value) {
    const widget = findWidget(node, name);
    if (!widget) {
        return;
    }
    widget.value = value;
    widget.callback?.(value);
}

function hideBackingWidget(widget) {
    widget.hidden = true;
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
}

function resizeCanvasForDisplay(canvas) {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width || canvas.clientWidth || 320));
    const height = Math.max(1, Math.round(rect.height || canvas.clientHeight || 160));
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const bufferWidth = Math.round(width * dpr);
    const bufferHeight = Math.round(height * dpr);
    if (canvas.width !== bufferWidth || canvas.height !== bufferHeight) {
        canvas.width = bufferWidth;
        canvas.height = bufferHeight;
    }
    return { width, height, dpr };
}

function drawEditor(canvas, points, activeIndex, isRecording, mode, target) {
    const { width, height, dpr } = resizeCanvasForDisplay(canvas);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#111827";
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "#253247";
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
        ctx.beginPath();
        ctx.moveTo((width * i) / 4, 0);
        ctx.lineTo((width * i) / 4, height);
        ctx.moveTo(0, (height * i) / 4);
        ctx.lineTo(width, (height * i) / 4);
        ctx.stroke();
    }
    const toCanvas = (p) => {
        const depth = Number(p.z || 0);
        return {
            x: width / 2 + (Number(p.x || 0) + depth * 0.22) * width * 0.24,
            y: height / 2 - (Number(p.y || 0) - depth * 0.12) * height * 0.24,
            s: 1 + depth * 0.18,
        };
    };
    const cube = [
        { x: -0.35, y: -0.35, z: -0.2 }, { x: 0.35, y: -0.35, z: -0.2 },
        { x: 0.35, y: 0.35, z: -0.2 }, { x: -0.35, y: 0.35, z: -0.2 },
        { x: -0.25, y: -0.25, z: 0.35 }, { x: 0.25, y: -0.25, z: 0.35 },
        { x: 0.25, y: 0.25, z: 0.35 }, { x: -0.25, y: 0.25, z: 0.35 },
    ].map(toCanvas);
    const edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
    ctx.strokeStyle = "#475569";
    ctx.lineWidth = 1.5;
    for (const [a, b] of edges) {
        ctx.beginPath();
        ctx.moveTo(cube[a].x, cube[a].y);
        ctx.lineTo(cube[b].x, cube[b].y);
        ctx.stroke();
    }
    const targetPoint = toCanvas(target);
    ctx.strokeStyle = "#facc15";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(targetPoint.x - 8, targetPoint.y);
    ctx.lineTo(targetPoint.x + 8, targetPoint.y);
    ctx.moveTo(targetPoint.x, targetPoint.y - 8);
    ctx.lineTo(targetPoint.x, targetPoint.y + 8);
    ctx.stroke();
    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px sans-serif";
    ctx.fillText(`3D path ${mode}`, 8, 15);

    ctx.lineWidth = 2;
    ctx.strokeStyle = isRecording ? "#22c55e" : "#38bdf8";
    ctx.beginPath();
    points.forEach((p, index) => {
        const c = toCanvas(p);
        if (index === 0) {
            ctx.moveTo(c.x, c.y);
        } else {
            ctx.lineTo(c.x, c.y);
        }
    });
    ctx.stroke();
    points.forEach((p, index) => {
        const c = toCanvas(p);
        ctx.fillStyle = index === activeIndex ? "#f97316" : "#e2e8f0";
        ctx.beginPath();
        ctx.arc(c.x, c.y, (index === activeIndex ? 5 : 4) * Math.max(0.65, c.s), 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#cbd5e1";
        ctx.font = "10px sans-serif";
        ctx.fillText(String(index + 1), c.x + 6, c.y - 6);
    });
}

function installCameraPathEditor(node) {
    const pathWidget = findWidget(node, "path_json");
    if (!pathWidget || findWidget(node, "sharp_camera_path_editor")) {
        return;
    }
    hideBackingWidget(pathWidget);
    const root = document.createElement("div");
    root.style.display = "flex";
    root.style.flexDirection = "column";
    root.style.gap = "6px";
    root.style.padding = "6px";
    root.style.background = "#0f172a";
    root.style.border = "1px solid #334155";
    root.style.borderRadius = "6px";
    root.style.boxSizing = "border-box";
    root.style.height = "100%";
    root.style.minHeight = "256px";
    root.style.overflow = "hidden";

    const header = document.createElement("button");
    header.textContent = "Close Path Editor";
    header.style.padding = "4px 8px";
    header.style.borderRadius = "4px";
    header.style.border = "1px solid #475569";
    header.style.background = "#1e293b";
    header.style.color = "#e2e8f0";
    header.style.textAlign = "left";
    header.style.lineHeight = "16px";
    header.style.flex = "0 0 28px";
    root.appendChild(header);

    const panel = document.createElement("div");
    panel.style.display = "flex";
    panel.style.flexDirection = "column";
    panel.style.gap = "6px";
    panel.style.flex = "1 1 auto";
    panel.style.minHeight = "0";
    root.appendChild(panel);

    const canvas = document.createElement("canvas");
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.minHeight = "120px";
    canvas.style.flex = "1 1 auto";
    canvas.style.cursor = "crosshair";
    canvas.style.borderRadius = "4px";
    panel.appendChild(canvas);

    const modeRow = document.createElement("div");
    modeRow.style.display = "grid";
    modeRow.style.gridTemplateColumns = "1fr 1fr";
    modeRow.style.gap = "4px";
    modeRow.style.flex = "0 0 24px";
    panel.appendChild(modeRow);
    const modeSelect = document.createElement("select");
    for (const name of ["orbit", "pan", "dolly", "anchor"]) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        modeSelect.appendChild(opt);
    }
    modeRow.appendChild(modeSelect);
    const anchorSelect = document.createElement("select");
    for (const name of ["center", "foreground", "background", "left", "right", "top", "bottom", "custom"]) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = `anchor ${name}`;
        anchorSelect.appendChild(opt);
    }
    anchorSelect.value = widgetValue(node, "target_anchor", "center");
    anchorSelect.onchange = () => {
        setWidgetValue(node, "target_anchor", anchorSelect.value);
        refresh();
    };
    modeRow.appendChild(anchorSelect);

    const controls = document.createElement("div");
    controls.style.display = "grid";
    controls.style.gridTemplateColumns = "repeat(4, 1fr)";
    controls.style.gap = "4px";
    controls.style.flex = "0 0 58px";
    panel.appendChild(controls);

    const zInput = document.createElement("input");
    zInput.type = "range";
    zInput.min = "-1";
    zInput.max = "1";
    zInput.step = "0.01";
    zInput.value = "0";
    zInput.style.flex = "0 0 18px";
    panel.appendChild(zInput);

    let points = parsePath(pathWidget.value);
    let activeIndex = 0;
    let recording = false;
    let recordStart = 0;

    const getTarget = () => ({
        x: Number(widgetValue(node, "target_x", 0)),
        y: Number(widgetValue(node, "target_y", 0)),
        z: Number(widgetValue(node, "target_z", 0)),
    });
    const refresh = () => drawEditor(canvas, points, activeIndex, recording, modeSelect.value, getTarget());
    const sync = () => {
        writePath(node, points);
        refresh();
    };
    const button = (label, fn) => {
        const el = document.createElement("button");
        el.textContent = label;
        el.style.padding = "5px 4px";
        el.style.borderRadius = "4px";
        el.style.border = "1px solid #475569";
        el.style.background = "#1e293b";
        el.style.color = "#e2e8f0";
        el.style.minWidth = "0";
        el.style.lineHeight = "16px";
        el.onclick = fn;
        controls.appendChild(el);
        return el;
    };

    const recordButton = button("Record", () => {
        recording = !recording;
        recordButton.textContent = recording ? "Stop" : "Record";
        recordStart = performance.now();
        if (recording) {
            points = [];
        }
        refresh();
    });
    button("Add", () => {
        const last = points[activeIndex] || { t: 0.5, x: 0, y: 0, z: 0 };
        points.push({ ...last, t: points.length ? 1 : 0 });
        activeIndex = points.length - 1;
        sync();
    });
    button("Clear", () => {
        points = defaultPath();
        activeIndex = 0;
        sync();
    });
    button("Mirror", () => {
        points = points.map((p) => ({ ...p, x: -p.x }));
        sync();
    });
    button("Anchor", () => {
        modeSelect.value = "anchor";
        setWidgetValue(node, "target_anchor", "custom");
        anchorSelect.value = "custom";
        refresh();
    });

    function pointerToPoint(event) {
        const rect = canvas.getBoundingClientRect();
        const rawX = clamp(((event.clientX - rect.left) / rect.width - 0.5) / 0.24, -2, 2);
        const rawY = clamp((0.5 - (event.clientY - rect.top) / rect.height) / 0.24, -2, 2);
        const current = points[activeIndex] || { t: 0.5, x: 0, y: 0, z: 0 };
        if (modeSelect.value === "dolly") {
            return { x: current.x, y: current.y, z: clamp(rawY, -2, 2) };
        }
        return { x: rawX, y: rawY, z: Number(zInput.value) };
    }

    function updateFromPointer(event) {
        const p = pointerToPoint(event);
        if (modeSelect.value === "anchor") {
            setWidgetValue(node, "target_anchor", "custom");
            setWidgetValue(node, "target_x", p.x);
            setWidgetValue(node, "target_y", p.y);
            setWidgetValue(node, "target_z", p.z);
            anchorSelect.value = "custom";
            refresh();
            return;
        }
        if (recording) {
            const elapsed = clamp((performance.now() - recordStart) / 4000, 0, 1);
            const previous = points[points.length - 1];
            if (!previous || Math.abs(previous.x - p.x) + Math.abs(previous.y - p.y) > 0.03) {
                points.push({ t: elapsed, ...p });
            }
            if (elapsed >= 1) {
                recording = false;
                recordButton.textContent = "Record";
            }
            sync();
            return;
        }
        if (!points.length) {
            points = defaultPath();
        }
        points[activeIndex] = { ...points[activeIndex], ...p };
        sync();
    }

    canvas.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        canvas.setPointerCapture(event.pointerId);
        updateFromPointer(event);
    });
    canvas.addEventListener("pointermove", (event) => {
        if (event.buttons) {
            event.preventDefault();
            updateFromPointer(event);
        }
    });
    canvas.addEventListener("wheel", (event) => {
        event.preventDefault();
        zInput.value = String(clamp(Number(zInput.value) - event.deltaY * 0.001, -1, 1));
        refresh();
    }, { passive: false });

    pathWidget.callback = ((original) => function () {
        original?.apply(this, arguments);
        points = parsePath(pathWidget.value);
        activeIndex = Math.min(activeIndex, points.length - 1);
        refresh();
    })(pathWidget.callback);

    const editorWidget = node.addDOMWidget("sharp_camera_path_editor", "sharp_camera_path", root, {
        serialize: false,
        hideOnZoom: false,
    });
    let expanded = true;
    const setExpanded = (value) => {
        expanded = Boolean(value);
        panel.style.display = expanded ? "flex" : "none";
        root.style.height = expanded ? "100%" : "32px";
        root.style.minHeight = expanded ? "256px" : "32px";
        header.textContent = expanded ? "Close Path Editor" : "Open Path Editor";
        node.setSize([Math.max(node.size[0], 380), node.computeSize()[1]]);
        node.setDirtyCanvas?.(true, true);
        refresh();
    };
    header.onclick = () => setExpanded(!expanded);
    editorWidget.computeSize = (width) => [width, expanded ? 264 : 40];
    if (window.ResizeObserver) {
        const observer = new ResizeObserver(() => refresh());
        observer.observe(root);
        observer.observe(canvas);
    }
    node.setSize([Math.max(node.size[0], 380), node.computeSize()[1]]);
    refresh();
}

function scheduleCameraPathEditor(node) {
    if (!node || node.type !== "VHS_SHARP_CameraPath") {
        return;
    }
    let attempts = 0;
    const retry = () => {
        if (findWidget(node, "sharp_camera_path_editor")) {
            return;
        }
        installCameraPathEditor(node);
        if (!findWidget(node, "sharp_camera_path_editor") && attempts < 30) {
            attempts += 1;
            setTimeout(retry, 100);
        }
    };
    setTimeout(retry, 0);
}

app.registerExtension({
    name: "VideoHelperSuite.SHARP.CameraPath",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "VHS_SHARP_CameraPath") {
            return;
        }
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
            scheduleCameraPathEditor(this);
        });
        chainCallback(nodeType.prototype, "onConfigure", function () {
            scheduleCameraPathEditor(this);
        });
    },
    async setup() {
        for (const node of app.graph?._nodes || []) {
            scheduleCameraPathEditor(node);
        }
    },
});
