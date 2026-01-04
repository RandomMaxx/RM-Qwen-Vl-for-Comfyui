import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Qwen3VL.Interact",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Qwen3VL_ImageLoader") {
            const getWidget = (n, name) => n.widgets?.find(w => w.name === name);

            // Debounce helper
            const debounce = (func, delay) => {
                let t;
                return (...args) => {
                    clearTimeout(t);
                    t = setTimeout(() => func.apply(this, args), delay);
                };
            };

            // Helper to update the "count" display widget
            const updateCountDisplay = (node, current, total) => {
                let wCount = getWidget(node, "count_display"); // Using specific name to avoid conflicts
                if (!wCount) {
                    wCount = node.addWidget("text", "count_display", "0/0", () => {}, { serialize: false });
                }
                
                // Enforce Integer types for display
                const c = parseInt(current);
                const t = parseInt(total);
                
                wCount.value = `${isNaN(c) ? 0 : c}/${isNaN(t) ? 0 : t}`;
                
                if (wCount.inputEl) {
                    wCount.inputEl.readOnly = true;
                    wCount.inputEl.style.textAlign = "center";
                    wCount.inputEl.style.opacity = "0.7";
                    wCount.inputEl.style.cursor = "default";
                }
            };

            // --- MAIN LOGIC: Update File List & Widgets ---
            nodeType.prototype.updateFileList = async function() {
                const wDir = getWidget(this, "directory");
                const wFile = getWidget(this, "filename");
                const wFilter = getWidget(this, "filter");
                const wIndex = getWidget(this, "index");
                const wRecur = getWidget(this, "recursive");
                const wQuick = getWidget(this, "quick_select");

                if (!wDir || !wFile) return;

                const params = new URLSearchParams({ 
                    path: wDir.value, 
                    filter: wFilter.value || "", 
                    recursive: wRecur.value, 
                    t: Date.now() 
                });

                try {
                    const res = await fetch(`/qwen/files?${params}`);
                    if (res.ok) {
                        const data = await res.json();
                        const files = data.files || [];
                        
                        if (files.length > 0) {
                            wFile.options.values = files;
                            wIndex.options.max = files.length;
                            
                            // Clamp input index
                            if (wIndex.value > files.length) wIndex.value = files.length;
                            if (wIndex.value < 1) wIndex.value = 1;
                            
                            wFile.value = files[wIndex.value - 1];
                            updateCountDisplay(this, wIndex.value, files.length);
                        } else {
                            wFile.options.values = ["(No matches)"];
                            wFile.value = "(No matches)";
                            updateCountDisplay(this, 0, 0);
                        }
                    }
                } catch (e) { console.error(e); }
                this.triggerPreview();
                this.updateHistoryDropdown(wQuick);
            };

            // Helper to refresh History Dropdown via API
            nodeType.prototype.updateHistoryDropdown = async function(wQuick) {
                if (!wQuick) return;
                // UI update logic for history if needed
            };

            nodeType.prototype.triggerPreview = function() {
                const wDir = getWidget(this, "directory");
                const wFile = getWidget(this, "filename");
                if (!wFile || !wFile.value || wFile.value.startsWith("(")) return;
                
                const params = new URLSearchParams({ path: wDir.value, filename: wFile.value, t: Date.now() });
                const img = new Image();
                img.onload = () => { this.imgs = [img]; this.setDirtyCanvas(true, true); };
                img.src = `/qwen/live_preview?${params}`;
            };

            // --- EXECUTION LOGIC (Visual Update Only) ---
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function() {
                onExecuted?.apply(this, arguments);
                const wIndex = getWidget(this, "index");
                const wFile = getWidget(this, "filename");
                
                if (!wFile || !wFile.options || !wFile.options.values) return;
                const count = wFile.options.values.length;

                // Update visual count
                const currentVal = wIndex.value;
                updateCountDisplay(this, currentVal, count);
            };

            // --- INITIALIZATION ---
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                const wDir = getWidget(this, "directory");
                const wFilter = getWidget(this, "filter");
                const wIndex = getWidget(this, "index");
                const wFile = getWidget(this, "filename");
                const wQuick = getWidget(this, "quick_select");

                // Initialize Count Display
                if (wIndex) updateCountDisplay(this, wIndex.value, 0);

                // 1. Directory Change -> Update List
                if (wDir) {
                    const debouncedUpdate = debounce(() => this.updateFileList(), 500);
                    wDir.callback = () => debouncedUpdate();
                }

                // 2. Quick Select -> Update Directory
                if (wQuick) {
                    wQuick.callback = (val) => {
                         if (val && val !== "--- Local Input Folders ---" && val !== "(Select to Auto-Fill)") {
                            if (wDir) {
                                wDir.value = val;
                                this.updateFileList();
                            }
                         }
                    };
                }

                // 3. Index Change -> Update Filename
                if (wIndex) {
                    wIndex.callback = (v) => {
                        const f = wFile.options.values;
                        if (f && f.length > 0 && !f[0].startsWith("(")) {
                            const safeIndex = Math.max(1, Math.min(v, f.length));
                            if (safeIndex !== v) wIndex.value = safeIndex; 
                            
                            wFile.value = f[safeIndex - 1];
                            updateCountDisplay(this, safeIndex, f.length);
                            this.triggerPreview();
                        }
                    };
                }

                // 4. Filter Change
                let t;
                if (wFilter) {
                    wFilter.callback = () => { clearTimeout(t); t = setTimeout(() => this.updateFileList(), 500); };
                }

                // 5. Filename Change -> Update Index
                if (wFile) {
                    wFile.callback = (v) => {
                        const idx = wFile.options.values.indexOf(v);
                        if (idx !== -1) { 
                            wIndex.value = idx + 1; 
                            updateCountDisplay(this, idx + 1, wFile.options.values.length);
                            this.triggerPreview(); 
                        }
                    };
                }

                // Initial Load
                setTimeout(() => this.updateFileList(), 100);
            };
        }
    }
});