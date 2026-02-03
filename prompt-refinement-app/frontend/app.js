/**
 * Prompt Refinement Factory - Frontend Application
 */

// API Base URL
const API_BASE = '';

// State
let currentTrace = null;
let loadedTraces = [];
let currentMode = 'planning';
let currentPromptVersion = 'v2';  // Default to V2 (string-based task descriptions)

// DOM Elements
const elements = {
    healthStatus: document.getElementById('health-status'),
    traceSelect: document.getElementById('trace-select'),
    loadTraceBtn: document.getElementById('load-trace-btn'),
    callSelection: document.getElementById('call-selection'),
    callSelect: document.getElementById('call-select'),
    imageGrid: document.getElementById('image-grid'),
    beforeImg: document.getElementById('before-img'),
    afterImg: document.getElementById('after-img'),
    systemPrompt: document.getElementById('system-prompt'),
    userPrompt: document.getElementById('user-prompt'),
    loadCurrentSystemBtn: document.getElementById('load-current-system'),
    loadCurrentUserBtn: document.getElementById('load-current-user'),
    runBtn: document.getElementById('run-btn'),
    unloadBtn: document.getElementById('unload-btn'),
    loadedModelStatus: document.getElementById('loaded-model-status'),
    exportBtn: document.getElementById('export-btn'),
    outputContainer: document.getElementById('output-container'),
    comparisonSection: document.getElementById('comparison-section'),
    comparisonResult: document.getElementById('comparison-result'),
    saveVersionBtn: document.getElementById('save-version-btn'),
    loadVersionsBtn: document.getElementById('load-versions-btn'),
    versionsList: document.getElementById('versions-list'),
    saveModal: document.getElementById('save-modal'),
    versionId: document.getElementById('version-id'),
    versionName: document.getElementById('version-name'),
    versionNotes: document.getElementById('version-notes'),
    saveConfirmBtn: document.getElementById('save-confirm-btn'),
    saveCancelBtn: document.getElementById('save-cancel-btn'),
    originalModal: document.getElementById('original-modal'),
    originalOutput: document.getElementById('original-output'),
    closeOriginalBtn: document.getElementById('close-original-btn'),
    backendSelect: document.getElementById('backend-select'),
    backendStatus: document.getElementById('backend-status'),
};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await checkHealth();
    await loadTraceList();
    await loadPromptVersions();
    setupEventListeners();
    // Auto-load V2 prompts on startup
    await loadCurrentPrompts('system');
    await loadCurrentPrompts('user');
});

// Event Listeners
function setupEventListeners() {
    // Mode selection
    document.querySelectorAll('input[name="mode"]').forEach(radio => {
        radio.addEventListener('change', async (e) => {
            currentMode = e.target.value;
            updateImageLabels();
            if (currentTrace) {
                populateCallSelect();
            }
            // Auto-reload prompts for new mode
            await loadCurrentPrompts('system');
            await loadCurrentPrompts('user');
        });
    });

    // Prompt version selection
    document.querySelectorAll('input[name="prompt-version"]').forEach(radio => {
        radio.addEventListener('change', async (e) => {
            currentPromptVersion = e.target.value;
            updatePromptVersionHint();
            // Auto-reload prompts for new version
            await loadCurrentPrompts('system');
            await loadCurrentPrompts('user');
        });
    });

    // Load trace button
    elements.loadTraceBtn.addEventListener('click', loadSelectedTrace);

    // Call selection
    elements.callSelect.addEventListener('change', loadSelectedCall);

    // Load current prompts
    elements.loadCurrentSystemBtn.addEventListener('click', () => loadCurrentPrompts('system'));
    elements.loadCurrentUserBtn.addEventListener('click', () => loadCurrentPrompts('user'));

    // Run button
    elements.runBtn.addEventListener('click', runModels);

    // Export button
    elements.exportBtn.addEventListener('click', exportForClaude);

    // Unload button
    elements.unloadBtn.addEventListener('click', unloadModel);

    // Version buttons
    elements.saveVersionBtn.addEventListener('click', showSaveModal);
    elements.loadVersionsBtn.addEventListener('click', loadPromptVersions);
    elements.saveConfirmBtn.addEventListener('click', saveVersion);
    elements.saveCancelBtn.addEventListener('click', hideSaveModal);

    // Close original modal
    elements.closeOriginalBtn.addEventListener('click', () => {
        elements.originalModal.style.display = 'none';
    });

    // Close modals on outside click
    elements.saveModal.addEventListener('click', (e) => {
        if (e.target === elements.saveModal) hideSaveModal();
    });
    elements.originalModal.addEventListener('click', (e) => {
        if (e.target === elements.originalModal) {
            elements.originalModal.style.display = 'none';
        }
    });
}

// API Calls
async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
            ...options,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error(`API Error (${endpoint}):`, error);
        throw error;
    }
}

// Health Check
async function checkHealth() {
    try {
        const data = await apiCall('/api/health');
        const statusDot = elements.healthStatus.querySelector('.status-dot');
        const statusText = elements.healthStatus.querySelector('.status-text');

        statusDot.classList.remove('ok', 'error');
        statusDot.classList.add('ok');
        statusText.textContent = data.mock_mode ? 'Mock Mode' : 'Connected';

        // Update loaded model status
        updateLoadedModelStatus();

        // Check vLLM availability
        checkBackendStatus();
    } catch (error) {
        const statusDot = elements.healthStatus.querySelector('.status-dot');
        const statusText = elements.healthStatus.querySelector('.status-text');

        statusDot.classList.remove('ok');
        statusDot.classList.add('error');
        statusText.textContent = 'Error';
    }
}

// Check backend server status (vLLM and TensorRT-LLM)
async function checkBackendStatus() {
    if (!elements.backendStatus) return;

    try {
        // Check both backends in parallel
        const [vllmData, tensorrtData] = await Promise.all([
            apiCall('/api/vllm/status').catch(() => ({ available: false })),
            apiCall('/api/tensorrt/status').catch(() => ({ available: false })),
        ]);

        const available = [];
        if (vllmData.available) {
            available.push('vLLM');
        }
        if (tensorrtData.available) {
            available.push('TensorRT');
        }

        if (available.length > 0) {
            elements.backendStatus.textContent = `Available: ${available.join(', ')}`;
            elements.backendStatus.classList.remove('error');
            elements.backendStatus.classList.add('ok');
        } else {
            elements.backendStatus.textContent = 'Servers not running';
            elements.backendStatus.classList.remove('ok');
            elements.backendStatus.classList.add('error');
        }
    } catch (error) {
        elements.backendStatus.textContent = 'Not available';
        elements.backendStatus.classList.remove('ok');
        elements.backendStatus.classList.add('error');
    }
}

// Update loaded model status
async function updateLoadedModelStatus() {
    try {
        const data = await apiCall('/api/models');
        if (data.current_loaded) {
            elements.loadedModelStatus.textContent = `Loaded: ${data.current_loaded}`;
        } else {
            elements.loadedModelStatus.textContent = 'No model loaded';
        }
    } catch (error) {
        elements.loadedModelStatus.textContent = '';
    }
}

// Unload model
async function unloadModel() {
    elements.unloadBtn.disabled = true;
    elements.unloadBtn.textContent = 'Unloading...';

    try {
        const result = await apiCall('/api/unload', { method: 'POST' });
        alert(`Model unloaded: ${result.unloaded_model || 'none'}\nGPU memory freed.`);
        updateLoadedModelStatus();
    } catch (error) {
        alert(`Failed to unload: ${error.message}`);
    } finally {
        elements.unloadBtn.disabled = false;
        elements.unloadBtn.textContent = 'Unload Model & Free GPU';
    }
}

// Load Trace List
async function loadTraceList() {
    try {
        const traces = await apiCall('/api/traces');
        loadedTraces = traces;

        elements.traceSelect.innerHTML = '<option value="">-- Select a trace --</option>';

        traces.forEach(trace => {
            const option = document.createElement('option');
            option.value = trace.id;
            option.textContent = `${trace.id} (P:${trace.num_planning}, M:${trace.num_monitoring})`;
            elements.traceSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to load traces:', error);
    }
}

// Load Selected Trace
async function loadSelectedTrace() {
    const traceId = elements.traceSelect.value;
    if (!traceId) return;

    try {
        elements.loadTraceBtn.disabled = true;
        elements.loadTraceBtn.textContent = 'Loading...';

        currentTrace = await apiCall(`/api/traces/${traceId}`);

        // Show call selection
        elements.callSelection.style.display = 'block';
        populateCallSelect();

    } catch (error) {
        alert(`Failed to load trace: ${error.message}`);
    } finally {
        elements.loadTraceBtn.disabled = false;
        elements.loadTraceBtn.textContent = 'Load';
    }
}

// Populate Call Select
function populateCallSelect() {
    elements.callSelect.innerHTML = '';

    const calls = currentMode === 'planning' ? currentTrace.planning : currentTrace.monitoring;

    if (calls.length === 0) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = `No ${currentMode} calls in this trace`;
        elements.callSelect.appendChild(option);
        return;
    }

    calls.forEach((call, index) => {
        const option = document.createElement('option');
        option.value = index;
        const label = currentMode === 'planning'
            ? `#${call.call_number}: ${call.task_selected || 'N/A'}`
            : `#${call.call_number}: ${call.task_name} (${call.success ? 'SUCCESS' : 'FAIL'})`;
        option.textContent = label;
        elements.callSelect.appendChild(option);
    });

    // Load first call
    loadSelectedCall();
}

// Load Selected Call
function loadSelectedCall() {
    const index = parseInt(elements.callSelect.value);
    if (isNaN(index)) return;

    const calls = currentMode === 'planning' ? currentTrace.planning : currentTrace.monitoring;
    const call = calls[index];

    if (!call) return;

    // Don't override prompts - keep the latest production prompts
    // User can view original trace prompts via "View Original" if needed

    // Load images only
    loadCallImages(call);
}

// Load Call Images
function loadCallImages(call) {
    const traceId = currentTrace.id;

    // Reset images
    elements.beforeImg.style.display = 'none';
    elements.afterImg.style.display = 'none';

    if (currentMode === 'monitoring') {
        // Before/After images
        if (call.before_image) {
            const filename = call.before_image.split('/').pop();
            elements.beforeImg.src = `/api/images/${traceId}/${filename}`;
            elements.beforeImg.style.display = 'block';
        }
        if (call.after_image) {
            const filename = call.after_image.split('/').pop();
            elements.afterImg.src = `/api/images/${traceId}/${filename}`;
            elements.afterImg.style.display = 'block';
        }
    } else {
        // Planning: central/head image
        if (call.images) {
            const imgKey = call.images.central || call.images.head || Object.values(call.images)[0];
            if (imgKey) {
                const filename = imgKey.split('/').pop();
                elements.beforeImg.src = `/api/images/${traceId}/${filename}`;
                elements.beforeImg.style.display = 'block';
            }
        }
    }
}

// Update Image Labels
function updateImageLabels() {
    const beforePlaceholder = document.querySelector('#image-before span');
    const afterPlaceholder = document.querySelector('#image-after span');

    if (currentMode === 'monitoring') {
        beforePlaceholder.textContent = 'BEFORE';
        afterPlaceholder.textContent = 'AFTER';
    } else {
        beforePlaceholder.textContent = 'SCENE';
        afterPlaceholder.textContent = '(N/A)';
    }
}

// Load Current Prompts
async function loadCurrentPrompts(type) {
    try {
        const data = await apiCall(`/api/current-prompts/${currentMode}?version=${currentPromptVersion}`);

        if (type === 'system') {
            elements.systemPrompt.value = data.system_prompt;
        } else {
            elements.userPrompt.value = data.user_prompt;
        }

        // Show which version was loaded
        console.log(`Loaded ${currentMode} ${type} prompt (${data.version || currentPromptVersion})`);
    } catch (error) {
        alert(`Failed to load current prompts: ${error.message}`);
    }
}

// Update Prompt Version Hint
function updatePromptVersionHint() {
    const hintEl = document.getElementById('prompt-version-hint');
    if (!hintEl) return;

    if (currentPromptVersion === 'v1') {
        hintEl.textContent = "V1: 'left_orange_plate' | Coded task names for known objects only";
    } else {
        hintEl.textContent = "V2: 'Use left arm to pick up...' | String descriptions, supports unseen objects";
    }
}

// Run Models
async function runModels() {
    const selectedModels = Array.from(document.querySelectorAll('input[name="model"]:checked'))
        .map(cb => cb.value);

    if (selectedModels.length === 0) {
        alert('Please select at least one model');
        return;
    }

    const systemPrompt = elements.systemPrompt.value;
    const userPrompt = elements.userPrompt.value;

    if (!systemPrompt || !userPrompt) {
        alert('Please enter system and user prompts');
        return;
    }

    // Build images object
    const images = {};
    if (elements.beforeImg.style.display !== 'none' && elements.beforeImg.src) {
        images[currentMode === 'monitoring' ? 'before' : 'central'] = elements.beforeImg.src;
    }
    if (elements.afterImg.style.display !== 'none' && elements.afterImg.src) {
        images['after'] = elements.afterImg.src;
    }

    // Get generation controls
    const useSamplingInput = document.getElementById('use-sampling');
    const maxTokensInput = document.getElementById('max-tokens');
    const disableThinkingInput = document.getElementById('disable-thinking');
    const imageResolutionInput = document.getElementById('image-resolution');
    const useSampling = useSamplingInput ? useSamplingInput.checked : false;
    const maxTokens = maxTokensInput ? parseInt(maxTokensInput.value) : 128;
    const disableThinking = disableThinkingInput ? disableThinkingInput.checked : false;
    const imageResolution = imageResolutionInput ? imageResolutionInput.value : 'small';
    const backend = elements.backendSelect ? elements.backendSelect.value : 'transformers';

    // Show loading state
    elements.runBtn.disabled = true;
    elements.runBtn.textContent = 'Running...';
    elements.outputContainer.innerHTML = '<div class="spinner" style="margin: 2rem auto;"></div>';

    try {
        if (selectedModels.length === 1) {
            // Single model run
            const result = await apiCall('/api/run', {
                method: 'POST',
                body: JSON.stringify({
                    model: selectedModels[0],
                    mode: currentMode,
                    system_prompt: systemPrompt,
                    user_prompt: userPrompt,
                    images: images,
                    max_tokens: maxTokens,
                    disable_thinking: disableThinking,
                    use_sampling: useSampling,
                    image_resolution: imageResolution,
                    backend: backend,
                }),
            });

            displaySingleResult(selectedModels[0], result);
        } else {
            // Compare multiple models
            const result = await apiCall('/api/compare', {
                method: 'POST',
                body: JSON.stringify({
                    models: selectedModels,
                    mode: currentMode,
                    system_prompt: systemPrompt,
                    user_prompt: userPrompt,
                    images: images,
                    max_tokens: maxTokens,
                    disable_thinking: disableThinking,
                    use_sampling: useSampling,
                    image_resolution: imageResolution,
                    backend: backend,
                }),
            });

            displayComparisonResults(result);
        }
    } catch (error) {
        elements.outputContainer.innerHTML = `
            <div class="output-placeholder">
                <p class="error-text">Error: ${error.message}</p>
            </div>
        `;
    } finally {
        elements.runBtn.disabled = false;
        elements.runBtn.textContent = 'Run Selected Models';
        updateLoadedModelStatus();
    }
}

// Display Single Result
function displaySingleResult(model, result) {
    elements.outputContainer.innerHTML = '';
    elements.comparisonSection.style.display = 'none';

    const card = createOutputCard(model, result);
    elements.outputContainer.appendChild(card);
}

// Display Comparison Results
function displayComparisonResults(data) {
    elements.outputContainer.innerHTML = '';

    // Display each model's output
    for (const [model, result] of Object.entries(data.results)) {
        const card = createOutputCard(model, result);
        elements.outputContainer.appendChild(card);
    }

    // Display comparison
    if (data.comparison) {
        elements.comparisonSection.style.display = 'block';
        displayComparison(data.comparison);
    }
}

// Create Output Card
function createOutputCard(model, result) {
    const card = document.createElement('div');
    card.className = `output-card ${result.success ? 'success' : 'error'}`;

    const modelNames = {
        '4B-Instruct': 'Qwen 4B-Instruct',
        '4B-Thinking': 'Qwen 4B-Thinking',
        '2B-Instruct': 'Qwen 2B-Instruct',
        '2B-Thinking': 'Qwen 2B-Thinking',
    };

    // Build metrics display
    let metricsHtml = '';
    if (result.metrics) {
        const m = result.metrics;
        metricsHtml = `
            <div class="metrics-row">
                <span title="Generation time">Gen: ${m.generation_ms?.toFixed(0) || '?'}ms</span>
                <span title="Output tokens">Tokens: ${m.output_tokens || '?'}</span>
                <span title="Tokens per second">Speed: ${m.tokens_per_second?.toFixed(1) || '?'} tok/s</span>
            </div>
        `;
    }

    card.innerHTML = `
        <div class="output-card-header">
            <h4>${modelNames[model] || model}</h4>
            <span class="latency">${result.latency_ms ? result.latency_ms.toFixed(0) + 'ms' : 'N/A'}</span>
        </div>
        ${metricsHtml}
        <div class="output-tabs">
            <button class="output-tab active" data-tab="parsed">Parsed</button>
            <button class="output-tab" data-tab="raw">Raw</button>
        </div>
        <div class="output-card-body">
            <pre class="output-parsed">${formatJSON(result.parsed_output)}</pre>
            <pre class="output-raw" style="display:none;">${escapeHtml(result.raw_response || 'N/A')}</pre>
        </div>
        ${result.error ? `<p class="error-text">Error: ${escapeHtml(result.error)}</p>` : ''}
    `;

    // Tab switching
    card.querySelectorAll('.output-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            card.querySelectorAll('.output-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            const tabName = tab.dataset.tab;
            card.querySelector('.output-parsed').style.display = tabName === 'parsed' ? 'block' : 'none';
            card.querySelector('.output-raw').style.display = tabName === 'raw' ? 'block' : 'none';
        });
    });

    return card;
}

// Display Comparison
function displayComparison(comparison) {
    elements.comparisonResult.innerHTML = '';

    if (comparison.match) {
        elements.comparisonResult.innerHTML = `
            <div class="match-indicator match">
                <strong>All models agree</strong> - No discrepancies found
            </div>
        `;
    } else {
        let html = `
            <div class="match-indicator mismatch">
                <strong>Discrepancies found</strong> (Reference: ${comparison.reference_model})
            </div>
        `;

        comparison.discrepancies.forEach(d => {
            html += `
                <div class="discrepancy">
                    <span class="discrepancy-field">${d.field}</span>
                    <div><strong>${d.reference.model}:</strong> ${formatValue(d.reference.value)}</div>
                    <div><strong>${d.compared.model}:</strong> ${formatValue(d.compared.value)}</div>
                </div>
            `;
        });

        elements.comparisonResult.innerHTML = html;
    }
}

// Load Prompt Versions
async function loadPromptVersions() {
    try {
        const versions = await apiCall('/api/prompts');

        if (versions.length === 0) {
            elements.versionsList.innerHTML = '<p class="hint">No saved versions. Save your first version above.</p>';
            return;
        }

        elements.versionsList.innerHTML = '';

        versions.forEach(v => {
            const card = document.createElement('div');
            card.className = 'version-card';
            card.innerHTML = `
                <h4>${v.name || v.version_id}</h4>
                <div class="version-meta">
                    <div>${v.mode} prompt</div>
                    <div>${v.version_id}</div>
                </div>
            `;
            card.addEventListener('click', () => loadVersion(v.version_id));
            elements.versionsList.appendChild(card);
        });
    } catch (error) {
        console.error('Failed to load versions:', error);
    }
}

// Load Version
async function loadVersion(versionId) {
    try {
        const version = await apiCall(`/api/prompts/${versionId}`);

        elements.systemPrompt.value = version.system_prompt || '';
        elements.userPrompt.value = version.user_prompt || '';

        // Set mode
        document.querySelector(`input[name="mode"][value="${version.mode}"]`).checked = true;
        currentMode = version.mode;
        updateImageLabels();

        alert(`Loaded version: ${versionId}`);
    } catch (error) {
        alert(`Failed to load version: ${error.message}`);
    }
}

// Show Save Modal
function showSaveModal() {
    // Generate default version ID
    const timestamp = new Date().toISOString().slice(0, 10).replace(/-/g, '');
    elements.versionId.value = `${currentMode}_v${timestamp}`;
    elements.versionName.value = '';
    elements.versionNotes.value = '';
    elements.saveModal.style.display = 'flex';
}

// Hide Save Modal
function hideSaveModal() {
    elements.saveModal.style.display = 'none';
}

// Save Version
async function saveVersion() {
    const versionId = elements.versionId.value.trim();
    const name = elements.versionName.value.trim();

    if (!versionId) {
        alert('Please enter a version ID');
        return;
    }

    try {
        await apiCall('/api/prompts', {
            method: 'POST',
            body: JSON.stringify({
                version_id: versionId,
                name: name || versionId,
                mode: currentMode,
                system_prompt: elements.systemPrompt.value,
                user_prompt: elements.userPrompt.value,
                notes: elements.versionNotes.value,
            }),
        });

        hideSaveModal();
        await loadPromptVersions();
        alert(`Saved version: ${versionId}`);
    } catch (error) {
        alert(`Failed to save version: ${error.message}`);
    }
}

// Export for Claude Code
async function exportForClaude() {
    const systemPrompt = elements.systemPrompt.value;
    const userPrompt = elements.userPrompt.value;

    if (!systemPrompt || !userPrompt) {
        alert('Please enter system and user prompts');
        return;
    }

    // Build images object
    const images = {};
    if (elements.beforeImg.style.display !== 'none' && elements.beforeImg.src) {
        // Extract the API path from the full URL
        const url = new URL(elements.beforeImg.src);
        images[currentMode === 'monitoring' ? 'before' : 'central'] = url.pathname;
    }
    if (elements.afterImg.style.display !== 'none' && elements.afterImg.src) {
        const url = new URL(elements.afterImg.src);
        images['after'] = url.pathname;
    }

    if (Object.keys(images).length === 0) {
        alert('Please load a trace with images first');
        return;
    }

    elements.exportBtn.disabled = true;
    elements.exportBtn.textContent = 'Exporting...';

    try {
        const result = await apiCall('/api/export-for-claude', {
            method: 'POST',
            body: JSON.stringify({
                model: '4B-Instruct',  // Placeholder, not used for export
                mode: currentMode,
                system_prompt: systemPrompt,
                user_prompt: userPrompt,
                images: images,
            }),
        });

        // Show export result
        const exportInfo = `
Export created successfully!

Directory: ${result.export_dir}

Files:
- ${result.files.prompt}
${result.files.images.map(img => '- ' + img).join('\n')}

To use with Claude Code, run:
${result.usage}

Or simply:
claude "Read the prompt.md file and images in ${result.export_dir} and provide the JSON output"
        `.trim();

        alert(exportInfo);
        console.log('Export result:', result);

    } catch (error) {
        alert(`Export failed: ${error.message}`);
    } finally {
        elements.exportBtn.disabled = false;
        elements.exportBtn.textContent = 'Export for Claude Code';
    }
}

// Utility Functions
function formatJSON(obj) {
    if (!obj || Object.keys(obj).length === 0) {
        return 'No parsed output';
    }
    return JSON.stringify(obj, null, 2);
}

function formatValue(val) {
    if (val === null || val === undefined) return 'null';
    if (typeof val === 'object') return JSON.stringify(val);
    return String(val);
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
