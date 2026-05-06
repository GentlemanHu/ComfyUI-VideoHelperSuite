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

function drawEditor(canvas, points, activeIndex, isRecording) {
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#111827";
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
        ctx.beginPath();
        ctx.moveTo((width * i) / 4, 0);
        ctx.lineTo((width * i) / 4, height);
        ctx.moveTo(0, (height * i) / 4);
        ctx.lineTo(width, (height * i) / 4);
        ctx.stroke();
    }
    ctx.strokeStyle = "#94a3b8";
    ctx.beginPath();
    ctx.moveTo(width / 2, 0);
    ctx.lineTo(width / 2, height);
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    const toCanvas = (p) => ({
        x: width / 2 + p.x * width * 0.24,
        y: height / 2 - p.y * height * 0.24,
    });
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
        ctx.arc(c.x, c.y, index === activeIndex ? 5 : 4, 0, Math.PI * 2);
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
    const root = document.createElement("div");
    root.style.display = "flex";
    root.style.flexDirection = "column";
    root.style.gap = "6px";
    root.style.padding = "6px";
    root.style.background = "#0f172a";
    root.style.border = "1px solid #334155";
    root.style.borderRadius = "6px";
    root.style.boxSizing = "border-box";

    const canvas = document.createElement("canvas");
    canvas.width = 320;
    canvas.height = 180;
    canvas.style.width = "100%";
    canvas.style.height = "180px";
    canvas.style.cursor = "crosshair";
    canvas.style.borderRadius = "4px";
    root.appendChild(canvas);

    const controls = document.createElement("div");
    controls.style.display = "grid";
    controls.style.gridTemplateColumns = "repeat(4, 1fr)";
    controls.style.gap = "4px";
    root.appendChild(controls);

    const zInput = document.createElement("input");
    zInput.type = "range";
    zInput.min = "-1";
    zInput.max = "1";
    zInput.step = "0.01";
    zInput.value = "0";
    root.appendChild(zInput);

    let points = parsePath(pathWidget.value);
    let activeIndex = 0;
    let recording = false;
    let recordStart = 0;

    const refresh = () => drawEditor(canvas, points, activeIndex, recording);
    const sync = () => {
        writePath(node, points);
        refresh();
    };
    const button = (label, fn) => {
        const el = document.createElement("button");
        el.textContent = label;
        el.style.padding = "4px";
        el.style.borderRadius = "4px";
        el.style.border = "1px solid #475569";
        el.style.background = "#1e293b";
        el.style.color = "#e2e8f0";
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

    function pointerToPoint(event) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: clamp(((event.clientX - rect.left) / rect.width - 0.5) / 0.24, -2, 2),
            y: clamp((0.5 - (event.clientY - rect.top) / rect.height) / 0.24, -2, 2),
            z: Number(zInput.value),
        };
    }

    function updateFromPointer(event) {
        const p = pointerToPoint(event);
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
    editorWidget.computeSize = (width) => [width, 260];
    node.setSize([Math.max(node.size[0], 360), node.computeSize()[1] + 260]);
    refresh();
}

app.registerExtension({
    name: "VideoHelperSuite.SHARP.CameraPath",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "VHS_SHARP_CameraPath") {
            return;
        }
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
            installCameraPathEditor(this);
        });
    },
});
